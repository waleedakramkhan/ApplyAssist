"""ApplyAssist Pipeline Orchestrator.

Runs pipeline stages in sequence or concurrently (streaming mode).

Usage (via CLI):
    applyassist run                        # all stages, sequential
    applyassist run --stream               # all stages, concurrent
    applyassist run discover enrich        # specific stages
    applyassist run score tailor cover     # LLM-only stages
    applyassist run --dry-run              # preview without executing
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from applyassist.config import load_env, ensure_dirs
from applyassist.database import init_db, get_connection, get_stats

log = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGE_ORDER = ("discover", "enrich", "score", "tailor", "cover", "pdf")

STAGE_META: dict[str, dict] = {
    "discover": {"desc": "Job discovery (JobSpy + Workday + smart extract)"},
    "enrich":   {"desc": "Detail enrichment (full descriptions + apply URLs)"},
    "score":    {"desc": "LLM scoring (fit 1-10)"},
    "tailor":   {"desc": "Resume tailoring (LLM + validation)"},
    "cover":    {"desc": "Cover letter generation"},
    "pdf":      {"desc": "PDF conversion (tailored resumes + cover letters)"},
}

# Upstream dependency: a stage only finishes when its upstream is done AND
# it has no remaining pending work.
_UPSTREAM: dict[str, str | None] = {
    "discover": None,
    "enrich":   "discover",
    "score":    "enrich",
    "tailor":   "score",
    "cover":    "tailor",
    "pdf":      "cover",
}


# ---------------------------------------------------------------------------
# Individual stage runners
# ---------------------------------------------------------------------------

def _run_discover(workers: int = 1) -> dict:
    """Stage: Job discovery — JobSpy, Workday, and smart-extract scrapers."""
    stats: dict = {"jobspy": None, "workday": None, "smartextract": None}

    # JobSpy
    console.print("  [cyan]JobSpy full crawl...[/cyan]")
    try:
        from applyassist.discovery.jobspy import run_discovery
        run_discovery()
        stats["jobspy"] = "ok"
    except Exception as e:
        log.error("JobSpy crawl failed: %s", e)
        console.print(f"  [red]JobSpy error:[/red] {e}")
        stats["jobspy"] = f"error: {e}"

    # Workday corporate scraper
    console.print("  [cyan]Workday corporate scraper...[/cyan]")
    try:
        from applyassist.discovery.workday import run_workday_discovery
        run_workday_discovery(workers=workers)
        stats["workday"] = "ok"
    except Exception as e:
        log.error("Workday scraper failed: %s", e)
        console.print(f"  [red]Workday error:[/red] {e}")
        stats["workday"] = f"error: {e}"

    # Smart extract
    console.print("  [cyan]Smart extract (AI-powered scraping)...[/cyan]")
    try:
        from applyassist.discovery.smartextract import run_smart_extract
        run_smart_extract(workers=workers)
        stats["smartextract"] = "ok"
    except Exception as e:
        log.error("Smart extract failed: %s", e)
        console.print(f"  [red]Smart extract error:[/red] {e}")
        stats["smartextract"] = f"error: {e}"

    # Guardrails: exclude region-locked roles and keyword-blocklisted ones right
    # after discovery, so the dashboard reflects an eligible-only list immediately
    # (re-applied before scoring once descriptions are enriched).
    try:
        from applyassist.config import load_exclusions
        from applyassist.database import apply_blocklist
        from applyassist.filters import classify_jobs, ELIGIBLE_STATUSES
        bl = load_exclusions()
        n_bl = apply_blocklist(bl["title_contains"], bl["company_contains"])
        counts = classify_jobs()  # uses the descriptions the scrapers already fetched
        eligible = sum(counts.get(s, 0) for s in ELIGIBLE_STATUSES)
        locked = counts.get("region_locked", 0)
        unknown = counts.get("unknown", 0)
        console.print(
            f"  [dim]Classified new jobs: {eligible} eligible, {locked} region-locked "
            f"(excluded), {unknown} unknown (parked). Blocklist excluded {n_bl}.[/dim]"
        )
        console.print(
            f"  [dim]Only the {eligible} eligible will be scored/tailored — "
            f"unknown jobs are visible in the dashboard; use 'Enrich' to process one.[/dim]"
        )
        stats["classified"] = counts
    except Exception as e:
        log.error("Guardrail classification failed: %s", e)

    return stats


def _run_enrich(workers: int = 1) -> dict:
    """Stage: Detail enrichment — scrape full descriptions and apply URLs."""
    try:
        from applyassist.enrichment.detail import run_enrichment
        run_enrichment(workers=workers)
        return {"status": "ok"}
    except Exception as e:
        log.error("Enrichment failed: %s", e)
        return {"status": f"error: {e}"}


def _run_score(score_model: str | None = None, score_provider: str | None = None) -> dict:
    """Stage: LLM scoring — assign fit scores 1-10."""
    try:
        from applyassist.scoring.scorer import run_scoring
        result = run_scoring(model=score_model, provider=score_provider)
        # run_scoring checkpoints per-job and returns a 'halted' status if the
        # provider cut us off (quota/connection). Pass that through so the
        # pipeline can stop cleanly instead of marching into tailor/cover.
        if isinstance(result, dict) and result.get("halted"):
            return result
        return {"status": "ok"}
    except Exception as e:
        log.error("Scoring failed: %s", e)
        return {"status": f"error: {e}"}


def _run_tailor(min_score: int = 7, validation_mode: str = "normal",
                tailor_provider: str | None = None) -> dict:
    """Stage: Resume tailoring — generate tailored resumes for high-fit jobs."""
    from applyassist.llm import LLMHaltError
    try:
        from applyassist.scoring.tailor import run_tailoring
        run_tailoring(min_score=min_score, validation_mode=validation_mode,
                      provider=tailor_provider)
        return {"status": "ok"}
    except LLMHaltError as e:
        return {"status": "halted: rate_limit", "halted": "rate_limit",
                "resume_hint": e.resume_hint,
                "message": f"Tailoring stopped: {e}. Progress saved — {e.resume_hint}."}
    except Exception as e:
        log.error("Tailoring failed: %s", e)
        return {"status": f"error: {e}"}


def _run_cover(min_score: int = 7, validation_mode: str = "normal",
               cover_provider: str | None = None) -> dict:
    """Stage: Cover letter generation."""
    from applyassist.llm import LLMHaltError
    try:
        from applyassist.scoring.cover_letter import run_cover_letters
        run_cover_letters(min_score=min_score, validation_mode=validation_mode,
                          provider=cover_provider)
        return {"status": "ok"}
    except LLMHaltError as e:
        return {"status": "halted: rate_limit", "halted": "rate_limit",
                "resume_hint": e.resume_hint,
                "message": f"Cover letters stopped: {e}. Progress saved — {e.resume_hint}."}
    except Exception as e:
        log.error("Cover letter generation failed: %s", e)
        return {"status": f"error: {e}"}


def _run_pdf() -> dict:
    """Stage: PDF conversion — convert tailored resumes and cover letters to PDF."""
    try:
        from applyassist.scoring.pdf import batch_convert
        batch_convert()
        return {"status": "ok"}
    except Exception as e:
        log.error("PDF conversion failed: %s", e)
        return {"status": f"error: {e}"}


# Map stage names to their runner functions
_STAGE_RUNNERS: dict[str, callable] = {
    "discover": _run_discover,
    "enrich":   _run_enrich,
    "score":    _run_score,
    "tailor":   _run_tailor,
    "cover":    _run_cover,
    "pdf":      _run_pdf,
}


# ---------------------------------------------------------------------------
# Stage resolution
# ---------------------------------------------------------------------------

def _resolve_stages(stage_names: list[str]) -> list[str]:
    """Resolve 'all' and validate/order stage names."""
    if "all" in stage_names:
        return list(STAGE_ORDER)

    resolved = []
    for name in stage_names:
        if name not in STAGE_META:
            console.print(
                f"[red]Unknown stage:[/red] '{name}'. "
                f"Available: {', '.join(STAGE_ORDER)}, all"
            )
            raise SystemExit(1)
        if name not in resolved:
            resolved.append(name)

    # Maintain canonical order
    return [s for s in STAGE_ORDER if s in resolved]


# ---------------------------------------------------------------------------
# Streaming pipeline helpers
# ---------------------------------------------------------------------------

class _StageTracker:
    """Thread-safe tracker for which stages have finished producing work."""

    def __init__(self):
        self._events: dict[str, threading.Event] = {
            stage: threading.Event() for stage in STAGE_ORDER
        }
        self._results: dict[str, dict] = {}
        self._lock = threading.Lock()

    def mark_done(self, stage: str, result: dict | None = None) -> None:
        with self._lock:
            self._results[stage] = result or {"status": "ok"}
        self._events[stage].set()

    def is_done(self, stage: str) -> bool:
        return self._events[stage].is_set()

    def wait(self, stage: str, timeout: float | None = None) -> bool:
        return self._events[stage].wait(timeout=timeout)

    def get_results(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._results)


# SQL to count pending work for each stage
_PENDING_SQL: dict[str, str] = {
    "enrich": "SELECT COUNT(*) FROM jobs WHERE detail_scraped_at IS NULL",
    "score":  "SELECT COUNT(*) FROM jobs WHERE full_description IS NOT NULL AND fit_score IS NULL",
    "tailor": (
        "SELECT COUNT(*) FROM jobs WHERE fit_score >= ? "
        "AND full_description IS NOT NULL "
        "AND tailored_resume_path IS NULL "
        "AND COALESCE(tailor_attempts, 0) < 5"
    ),
    "cover": (
        "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL "
        "AND (cover_letter_path IS NULL OR cover_letter_path = '') "
        "AND COALESCE(cover_attempts, 0) < 5"
    ),
    "pdf": (
        "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL "
        "AND tailored_resume_path LIKE '%.txt'"
    ),
}

# How long to sleep between polling loops in streaming mode (seconds)
_STREAM_POLL_INTERVAL = 10


def _count_pending(stage: str, min_score: int = 7) -> int:
    """Count pending work items for a stage."""
    sql = _PENDING_SQL.get(stage)
    if sql is None:
        return 0
    conn = get_connection()
    if "?" in sql:
        return conn.execute(sql, (min_score,)).fetchone()[0]
    return conn.execute(sql).fetchone()[0]


def _run_stage_streaming(
    stage: str,
    tracker: _StageTracker,
    stop_event: threading.Event,
    min_score: int = 7,
    workers: int = 1,
    validation_mode: str = "normal",
) -> None:
    """Run a single stage in streaming mode: loop until upstream done + no work.

    For discover: runs once, then marks done.
    For all others: polls DB for pending work, runs the batch processor,
    and repeats until upstream is done and no pending work remains.
    """
    runner = _STAGE_RUNNERS[stage]
    kwargs: dict = {}
    if stage in ("tailor", "cover"):
        kwargs["min_score"] = min_score
        kwargs["validation_mode"] = validation_mode
    if stage in ("discover", "enrich"):
        kwargs["workers"] = workers

    upstream = _UPSTREAM[stage]

    if stage == "discover":
        # Discover runs once (its sub-scrapers already do their full crawl)
        try:
            result = runner(**kwargs)
            tracker.mark_done(stage, result)
        except Exception as e:
            log.exception("Stage '%s' crashed", stage)
            tracker.mark_done(stage, {"status": f"error: {e}"})
        return

    # For downstream stages: loop until upstream done + no pending work
    passes = 0
    while not stop_event.is_set():
        # Wait for upstream to start producing work (first pass only)
        if passes == 0 and upstream and not tracker.is_done(upstream):
            # Wait a bit for upstream to produce some work before first run
            tracker.wait(upstream, timeout=_STREAM_POLL_INTERVAL)

        pending = _count_pending(stage, min_score)

        if pending > 0:
            try:
                runner(**kwargs)
                passes += 1
            except Exception as e:
                log.error("Stage '%s' error (pass %d): %s", stage, passes, e)
                passes += 1
        else:
            # No work right now
            upstream_done = upstream is None or tracker.is_done(upstream)
            if upstream_done:
                # No work and upstream is done — this stage is finished
                break
            # Upstream still running, wait and retry
            if stop_event.wait(timeout=_STREAM_POLL_INTERVAL):
                break  # Stop requested

    tracker.mark_done(stage, {"status": "ok", "passes": passes})


# ---------------------------------------------------------------------------
# Pipeline orchestrators
# ---------------------------------------------------------------------------

def _run_sequential(ordered: list[str], min_score: int, workers: int = 1,
                    validation_mode: str = "normal",
                    auto_resume: bool = False, resume_wait: int = 120,
                    score_model: str | None = None,
                    score_provider: str | None = None,
                    tailor_provider: str | None = None,
                    cover_provider: str | None = None) -> dict:
    """Execute stages one at a time (original behavior).

    With auto_resume, a stage that halts on a rate limit is slept on and re-run
    (it resumes from its per-job checkpoint) until it finishes or stops making
    progress — so a large batch can run unattended on a free-tier API.
    """
    results: list[dict] = []
    errors: dict[str, str] = {}
    pipeline_start = time.time()
    _MAX_STALL = 5  # consecutive no-progress resumes before giving up

    for name in ordered:
        meta = STAGE_META[name]
        console.print(f"\n{'=' * 70}")
        console.print(f"  [bold]STAGE: {name}[/bold] — {meta['desc']}")
        console.print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
        console.print(f"{'=' * 70}")

        runner = _STAGE_RUNNERS[name]
        kwargs: dict = {}
        if name in ("tailor", "cover"):
            kwargs["min_score"] = min_score
            kwargs["validation_mode"] = validation_mode
        if name in ("discover", "enrich"):
            kwargs["workers"] = workers
        if name == "score":
            kwargs["score_model"] = score_model
            kwargs["score_provider"] = score_provider
        if name == "tailor":
            kwargs["tailor_provider"] = tailor_provider
        if name == "cover":
            kwargs["cover_provider"] = cover_provider

        stall = 0
        while True:
            pending_before = _count_pending(name, min_score) if name in _PENDING_SQL else None
            t0 = time.time()
            halt_reason = None
            try:
                result = runner(**kwargs)
                elapsed = time.time() - t0
                status = "ok"
                halt_message = None
                if isinstance(result, dict):
                    status = result.get("status", "ok")
                    halt_reason = result.get("halted")
                    if halt_reason:
                        halt_message = result.get("message", "Stopped: LLM provider unavailable.")
                    if name == "discover":
                        sub_errors = [
                            f"{k}: {v}" for k, v in result.items()
                            if isinstance(v, str) and v.startswith("error")
                        ]
                        if sub_errors:
                            status = "partial"
            except Exception as e:
                elapsed = time.time() - t0
                status = f"error: {e}"
                halt_message = None
                log.exception("Stage '%s' crashed", name)
                console.print(f"\n  [red]STAGE FAILED:[/red] {e}")

            # Auto-resume only on a rate-limit halt; sleep and retry the same
            # (checkpointed) stage. Bail if it stops making progress.
            if not (halt_message and auto_resume and halt_reason == "rate_limit"):
                break
            progressed = True
            pend_now = None
            if pending_before is not None:
                pend_now = _count_pending(name, min_score)
                progressed = pend_now < pending_before
            stall = 0 if progressed else stall + 1
            if stall >= _MAX_STALL:
                console.print(
                    f"  [red]Auto-resume made no progress after {_MAX_STALL} tries "
                    f"(quota likely exhausted for now). Stopping.[/red]"
                )
                break
            remaining = f" (~{pend_now} left)" if pend_now is not None else ""
            console.print(
                f"  [yellow]⏳ Rate limited — auto-resuming '{name}' in {resume_wait}s{remaining}. "
                f"Progress is saved.[/yellow]"
            )
            try:
                time.sleep(resume_wait)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted during wait — stopping.[/yellow]")
                break

        results.append({"stage": name, "status": status, "elapsed": elapsed})
        if status not in ("ok", "partial"):
            errors[name] = status

        console.print(f"\n  Stage '{name}' completed in {elapsed:.1f}s — {status}")

        # If a stage halted due to LLM quota/connection limits, stop the whole
        # pipeline here — downstream LLM stages would just hit the same wall.
        # Progress is already checkpointed, so the next run resumes cleanly.
        if halt_message:
            # The correct resume command continues from the halted stage onward
            # (NOT bare `run`, which would re-scrape from discover). Drop discover
            # /enrich so resume never re-scrapes — the data is already in the DB.
            resume_stages = [s for s in ordered[ordered.index(name):]
                             if s not in ("discover", "enrich")] or [name]
            resume_cmd = "./applyassist.sh run " + " ".join(resume_stages)
            console.print(Panel(
                f"[bold yellow]Paused — LLM quota/limit reached.[/bold yellow]\n\n"
                f"{halt_message}\n\n"
                f"[bold]To resume[/bold] (continues where it stopped, no re-scraping):\n"
                f"   [bold cyan]{resume_cmd}[/bold cyan]\n\n"
                f"[dim]Do NOT use bare `./applyassist.sh run` — that restarts discovery "
                f"from scratch.[/dim]",
                title="⏸  Checkpoint saved",
                border_style="yellow",
            ))
            break

    total_elapsed = time.time() - pipeline_start
    return {"stages": results, "errors": errors, "elapsed": total_elapsed}


def _eligible_unscored() -> int:
    """Count jobs that can still be scored (enriched, eligible, not yet scored)."""
    return get_connection().execute(
        "SELECT COUNT(*) FROM jobs WHERE fit_score IS NULL "
        "AND full_description IS NOT NULL AND COALESCE(excluded,0)=0 "
        "AND location_status IN ('local','worldwide','relocation','override')"
    ).fetchone()[0]


def _run_batched(ordered: list[str], min_score: int, batch_size: int,
                 validation_mode: str, auto_resume: bool, resume_wait: int,
                 score_model: str | None, score_provider: str | None,
                 tailor_provider: str | None, cover_provider: str | None) -> dict:
    """Pipeline in small batches: score until `batch_size` eligible jobs are
    ready, push that batch through tailor → cover → pdf, then repeat. Gets
    ready-to-apply jobs out fast and keeps quota/memory bounded.
    """
    from applyassist.llm import LLMHaltError
    from applyassist.scoring.scorer import run_scoring
    from applyassist.scoring.tailor import run_tailoring
    from applyassist.scoring.cover_letter import run_cover_letters
    from applyassist.scoring.pdf import batch_convert

    do = {s: (s in ordered) for s in STAGE_ORDER}
    pipeline_start = time.time()
    results: list[dict] = []

    def stage(fn, label):
        """Run a stage fn; on a rate-limit halt, sleep+retry if auto_resume.
        Returns (ok, result_dict): ok=False means stop the whole run."""
        stalls = 0
        while not _stop_requested():
            halted = hint = None
            res = None
            try:
                res = fn()
                if isinstance(res, dict) and res.get("halted"):
                    halted, hint = True, res.get("resume_hint", "")
            except LLMHaltError as e:
                halted, hint = True, e.resume_hint
            except Exception as e:
                log.error("Batch stage '%s' error: %s", label, e)
                return True, {}  # non-halt error: skip, keep going
            if not halted:
                return True, (res if isinstance(res, dict) else {})
            if not auto_resume:
                console.print(Panel(
                    f"[yellow]Paused on '{label}' — rate limit/quota.[/yellow]\n{hint}\n\n"
                    f"Re-run the same command to continue.",
                    title="⏸  Checkpoint saved", border_style="yellow"))
                return False, {}
            stalls += 1
            if stalls > 5:
                console.print(f"  [red]'{label}' made no progress after 5 retries — stopping.[/red]")
                return False, {}
            console.print(f"  [yellow]⏳ '{label}' rate-limited — waiting {resume_wait}s...[/yellow]")
            time.sleep(resume_wait)
        return False, {}

    batch_no = 0
    while True:
        # 1. If scoring, score until a batch of eligible jobs is ready (or done).
        if do["score"]:
            guard = 0
            while _count_pending("tailor", min_score) < batch_size and _eligible_unscored() > 0:
                ok, _ = stage(lambda: run_scoring(limit=batch_size, model=score_model,
                                                  provider=score_provider), "score")
                if not ok:
                    return {"stages": results, "errors": {}, "elapsed": time.time() - pipeline_start}
                guard += 1
                if guard > 2000:
                    break

        # 2. Remaining work = jobs still needing a résumé OR a cover letter.
        pend_t = _count_pending("tailor", min_score) if do["tailor"] else 0
        pend_c = _count_pending("cover", min_score) if do["cover"] else 0
        if pend_t == 0 and pend_c == 0:
            break

        batch_no += 1
        console.print(f"\n[bold cyan]── Batch {batch_no}: {pend_t} to tailor, {pend_c} to cover "
                      f"(≤{batch_size}/stage) ──[/bold cyan]")

        produced = 0  # résumés + cover letters actually generated this batch
        if do["tailor"] and pend_t > 0:
            ok, r = stage(lambda: run_tailoring(min_score=min_score, limit=batch_size,
                                                validation_mode=validation_mode,
                                                provider=tailor_provider), "tailor")
            if not ok:
                break
            produced += r.get("approved", 0)
        if do["cover"] and pend_c > 0:
            ok, r = stage(lambda: run_cover_letters(min_score=min_score, limit=batch_size,
                                                    validation_mode=validation_mode,
                                                    provider=cover_provider), "cover")
            if not ok:
                break
            produced += r.get("generated", 0)
        if do["pdf"]:
            stage(lambda: batch_convert(), "pdf")

        # Stop only if a full batch produced NOTHING (remaining jobs are stuck on
        # validation/judge failures and will never succeed) and we're not still
        # scoring up new candidates.
        if produced == 0 and not (do["score"] and _eligible_unscored() > 0):
            console.print("  [yellow]No résumés or covers produced this batch — remaining jobs "
                          "are failing validation/judge. Stopping.[/yellow]")
            break

    elapsed = time.time() - pipeline_start
    console.print(f"\n[green]Batched run complete[/green] — {batch_no} batch(es) in {elapsed:.0f}s.")
    return {"stages": results, "errors": {}, "elapsed": elapsed}


def _stop_requested() -> bool:
    return False


def _run_streaming(ordered: list[str], min_score: int, workers: int = 1,
                   validation_mode: str = "normal") -> dict:
    """Execute stages concurrently with DB as conveyor belt."""
    tracker = _StageTracker()
    stop_event = threading.Event()
    pipeline_start = time.time()

    console.print(f"\n  [bold cyan]STREAMING MODE[/bold cyan] — stages run concurrently")
    console.print(f"  Poll interval: {_STREAM_POLL_INTERVAL}s\n")

    # Mark stages NOT in `ordered` as done so downstream doesn't wait for them
    for stage in STAGE_ORDER:
        if stage not in ordered:
            tracker.mark_done(stage, {"status": "skipped"})

    # Launch each stage in its own thread
    threads: dict[str, threading.Thread] = {}
    start_times: dict[str, float] = {}

    for name in ordered:
        start_times[name] = time.time()
        t = threading.Thread(
            target=_run_stage_streaming,
            args=(name, tracker, stop_event, min_score, workers, validation_mode),
            name=f"stage-{name}",
            daemon=True,
        )
        threads[name] = t
        t.start()
        console.print(f"  [dim]Started thread:[/dim] {name}")

    # Wait for all threads to finish
    try:
        for name in ordered:
            threads[name].join()
            elapsed = time.time() - start_times[name]
            console.print(
                f"  [green]Completed:[/green] {name} ({elapsed:.1f}s)"
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — stopping stages...[/yellow]")
        stop_event.set()
        for t in threads.values():
            t.join(timeout=10)

    total_elapsed = time.time() - pipeline_start

    # Build results from tracker
    all_results = tracker.get_results()
    results: list[dict] = []
    errors: dict[str, str] = {}

    for name in ordered:
        r = all_results.get(name, {"status": "unknown"})
        elapsed = time.time() - start_times.get(name, pipeline_start)
        status = r.get("status", "ok")

        results.append({"stage": name, "status": status, "elapsed": elapsed})
        if status not in ("ok", "partial", "skipped"):
            errors[name] = status

    return {"stages": results, "errors": errors, "elapsed": total_elapsed}


def run_pipeline(
    stages: list[str] | None = None,
    min_score: int = 7,
    dry_run: bool = False,
    stream: bool = False,
    workers: int = 1,
    validation_mode: str = "normal",
    auto_resume: bool = False,
    resume_wait: int = 120,
    score_model: str | None = None,
    score_provider: str | None = None,
    tailor_provider: str | None = None,
    cover_provider: str | None = None,
    batch: int = 0,
) -> dict:
    """Run pipeline stages.

    Args:
        stages: List of stage names, or None / ["all"] for full pipeline.
        min_score: Minimum fit score for tailor/cover stages.
        dry_run: If True, preview stages without executing.
        stream: If True, run stages concurrently (streaming mode).
        workers: Number of parallel threads for discovery/enrichment stages.

    Returns:
        Dict with keys: stages (list of result dicts), errors (dict), elapsed (float).
    """
    # Bootstrap
    load_env()
    ensure_dirs()
    init_db()

    # Resolve stages
    if stages is None:
        stages = ["all"]
    ordered = _resolve_stages(stages)

    # Banner
    mode = "streaming" if stream else "sequential"
    console.print()
    console.print(Panel.fit(
        f"[bold]ApplyAssist Pipeline[/bold] ({mode})",
        border_style="blue",
    ))
    console.print(f"  Min score:  {min_score}")
    console.print(f"  Workers:    {workers}")
    console.print(f"  Validation: {validation_mode}")
    console.print(f"  Stages:     {' -> '.join(ordered)}")

    # Pre-run stats
    pre_stats = get_stats()
    console.print(f"  DB:        {pre_stats['total']} jobs, {pre_stats['pending_detail']} pending enrichment")

    if dry_run:
        console.print(f"\n  [yellow]DRY RUN[/yellow] — would execute ({mode}):")
        for name in ordered:
            meta = STAGE_META[name]
            console.print(f"    {name:<12s}  {meta['desc']}")
        console.print(f"\n  No changes made.")
        return {"stages": [], "errors": {}, "elapsed": 0.0}

    # Smart resume: a full run starts at discover and re-scrapes the boards. If
    # the DB already holds discovered jobs that still need processing, don't
    # silently re-scrape — ask whether to continue the existing batch instead.
    if "discover" in ordered and sys.stdin.isatty():
        total = pre_stats.get("total", 0)
        # Count what will ACTUALLY be processed: eligible (not excluded, not
        # parked-unknown) jobs still needing scoring — not the raw unscored count,
        # which would misleadingly include region-locked/excluded rows.
        conn = get_connection()
        eligible_pending = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE fit_score IS NULL "
            "AND COALESCE(excluded,0)=0 "
            "AND (location_status IN ('local','worldwide','relocation','override') OR location_status IS NULL)"
        ).fetchone()[0]
        excluded = conn.execute("SELECT COUNT(*) FROM jobs WHERE excluded=1").fetchone()[0]
        parked = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE COALESCE(excluded,0)=0 AND location_status='unknown'"
        ).fetchone()[0]
        if total > 0:
            console.print(Panel(
                f"You already have [bold]{total}[/bold] discovered jobs:\n"
                f"  • [bold cyan]{eligible_pending}[/bold cyan] eligible & still to process "
                f"(enrich/score/tailor)\n"
                f"  • [dim]{excluded} excluded (region-locked/blocklisted) — skipped[/dim]\n"
                f"  • [dim]{parked} unknown-location — parked (use the dashboard 'Enrich' button)[/dim]\n\n"
                f"  [bold cyan]c[/bold cyan] = process the {eligible_pending} eligible existing jobs "
                f"(skip re-scraping) [dim]— recommended[/dim]\n"
                f"  [bold cyan]d[/bold cyan] = discover fresh AND process everything "
                f"(adds new postings)\n"
                f"  [bold cyan]q[/bold cyan] = cancel",
                title="Existing jobs found",
                border_style="cyan",
            ))
            choice = input("> ").strip().lower() if sys.stdin.isatty() else "d"
            if choice == "q":
                console.print("Cancelled.")
                return {"stages": [], "errors": {}, "elapsed": 0.0}
            if choice == "c":
                ordered = [s for s in ordered if s != "discover"]
                console.print(f"[dim]Skipping discovery. Stages: {' -> '.join(ordered)}[/dim]")

    # Execute
    if batch and batch > 0 and not stream:
        console.print(f"  [cyan]Batched mode:[/cyan] {batch} job(s) per batch through "
                      f"{' -> '.join(ordered)}")
        result = _run_batched(ordered, min_score, batch, validation_mode,
                              auto_resume, resume_wait, score_model, score_provider,
                              tailor_provider, cover_provider)
    elif stream:
        result = _run_streaming(ordered, min_score, workers=workers,
                                validation_mode=validation_mode)
    else:
        result = _run_sequential(ordered, min_score, workers=workers,
                                 validation_mode=validation_mode,
                                 auto_resume=auto_resume, resume_wait=resume_wait,
                                 score_model=score_model, score_provider=score_provider,
                                 tailor_provider=tailor_provider, cover_provider=cover_provider)

    # Summary table
    console.print(f"\n{'=' * 70}")
    summary = Table(title="Pipeline Summary", show_header=True, header_style="bold")
    summary.add_column("Stage", style="bold")
    summary.add_column("Status")
    summary.add_column("Time", justify="right")

    for r in result["stages"]:
        elapsed_str = f"{r['elapsed']:.1f}s"
        status_display = r["status"][:30]
        if r["status"] == "ok":
            style = "green"
        elif r["status"] in ("partial", "skipped"):
            style = "yellow"
        else:
            style = "red"
        summary.add_row(r["stage"], f"[{style}]{status_display}[/{style}]", elapsed_str)

    summary.add_row("", "", "")
    summary.add_row("[bold]Total[/bold]", "", f"[bold]{result['elapsed']:.1f}s[/bold]")
    console.print(summary)

    # Final DB stats
    final = get_stats()
    console.print(f"\n  [bold]DB Final State:[/bold]")
    console.print(f"    Total jobs:     {final['total']}")
    console.print(f"    With desc:      {final['with_description']}")
    console.print(f"    Scored:         {final['scored']}")
    console.print(f"    Tailored:       {final['tailored']}")
    console.print(f"    Cover letters:  {final['with_cover_letter']}")
    console.print(f"    Ready to apply: {final['ready_to_apply']}")
    console.print(f"    Applied:        {final['applied']}")
    console.print(f"{'=' * 70}\n")

    return result
