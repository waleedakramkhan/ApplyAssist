"""ApplyAssist CLI — the main entry point."""

from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from applyassist import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

app = typer.Typer(
    name="applyassist",
    help="AI job-search copilot: discover, score, and tailor — then you review and submit.",
    no_args_is_help=True,
)
console = Console()
log = logging.getLogger(__name__)

# Valid pipeline stages (in execution order)
VALID_STAGES = ("discover", "enrich", "score", "tailor", "cover", "pdf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    """Common setup: load env, create dirs, init DB."""
    from applyassist.config import load_env, ensure_dirs
    from applyassist.database import init_db

    load_env()
    ensure_dirs()
    init_db()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]applyassist[/bold] {__version__}")
        raise typer.Exit()


def _print_responsible_use_notice() -> None:
    """Print the once-per-run responsible-use notice before applying.

    Surfaces the philosophy in front of the user, not just in the README:
    volume is not the bottleneck, review everything, respect rate limits.
    """
    from rich.panel import Panel
    console.print(Panel(
        "[bold]ApplyAssist is a copilot, not a spray bot.[/bold]\n"
        "- More applications is not the goal — more [italic]interviews per application[/italic] is.\n"
        "- You review and submit every application. The agent does the tedious filling.\n"
        "- No CAPTCHA solving, no bot-detection evasion. If a site asks for a human, you step in.\n"
        "- A daily cap keeps your pace human. Referrals beat cold applications — spend the time you save there.",
        title="Responsible use",
        border_style="cyan",
    ))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """ApplyAssist — AI job-search copilot. It discovers, scores, and tailors; you review and submit."""


@app.command()
def init() -> None:
    """Run the first-time setup wizard (profile, resume, search config)."""
    from applyassist.wizard.init import run_wizard

    run_wizard()


@app.command()
def manifest() -> None:
    """Write a CSV mapping every tailored job to its résumé + cover-letter files."""
    _bootstrap()
    import csv
    from pathlib import Path
    from applyassist.config import APP_DIR
    from applyassist.database import get_connection

    conn = get_connection()
    rows = conn.execute(
        "SELECT fit_score, company, title, location, url, tailored_resume_path, "
        "cover_letter_path, applied_at FROM jobs WHERE tailored_resume_path IS NOT NULL "
        "ORDER BY applied_at IS NOT NULL, fit_score DESC, company"
    ).fetchall()
    out = APP_DIR / "applications.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score", "company", "title", "location", "applied",
                    "job_url", "resume_pdf", "cover_pdf"])
        for sc, co, ti, loc, url, rp, cp, ap in rows:
            rpdf = str(Path(rp).with_suffix(".pdf")) if rp else ""
            cpdf = str(Path(cp).with_suffix(".pdf")) if cp else ""
            w.writerow([sc, co or "(see posting)", ti, loc or "",
                        "YES" if ap else "", url, rpdf, cpdf])
    console.print(f"[green]Wrote {len(rows)} rows[/green] -> {out}")
    console.print(f"[dim]Open it:[/dim] open {out}")


@app.command()
def run(
    stages: Optional[list[str]] = typer.Argument(
        None,
        help=(
            "Pipeline stages to run. "
            f"Valid: {', '.join(VALID_STAGES)}, all. "
            "Defaults to 'all' if omitted."
        ),
    ),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for tailor/cover stages."),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel threads for discovery/enrichment stages."),
    stream: bool = typer.Option(False, "--stream", help="Run stages concurrently (streaming mode)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview stages without executing."),
    batch: int = typer.Option(0, "--batch", help="Process in batches of N: score until N eligible jobs are ready, push that batch through tailor→cover→pdf, then repeat. Gets ready-to-apply jobs out fast (e.g. --batch 5)."),
    auto_resume: bool = typer.Option(False, "--auto-resume", help="On a rate-limit halt, sleep and automatically resume the stage instead of stopping. Lets a big batch run unattended on free-tier APIs."),
    resume_wait: int = typer.Option(120, "--resume-wait", help="Seconds to sleep before each auto-resume (only with --auto-resume)."),
    provider: Optional[str] = typer.Option(None, "--provider", help="Default provider for ALL LLM stages this run (reads LLM_URL_<NAME>/LLM_API_KEY_<NAME>/LLM_MODEL_<NAME> from .env). Pass a COMMA-SEPARATED chain (e.g. --provider cerebras,gemini,groq) to auto-rotate to the next free provider when one hits its daily limit."),
    score_model: Optional[str] = typer.Option(None, "--score-model", help="Use a different model for the SCORE stage only (e.g. 'gemini-2.0-flash'), inferring provider from the model name. Tailor/cover keep the default writing model."),
    score_provider: Optional[str] = typer.Option(None, "--score-provider", help="Run the SCORE stage on a separate named provider (e.g. --score-provider cerebras) while tailor/cover use the default --provider. Wins over --score-model."),
    tailor_provider: Optional[str] = typer.Option(None, "--tailor-provider", help="Run the TAILOR (résumé) stage on a separate named provider (e.g. --tailor-provider cerebras) while the rest use the default --provider."),
    cover_provider: Optional[str] = typer.Option(None, "--cover-provider", help="Run the COVER LETTER stage on a separate named provider (e.g. --cover-provider mistral) while the rest use the default --provider."),
    validation: str = typer.Option(
        "normal",
        "--validation",
        help=(
            "Validation strictness for tailor/cover stages. "
            "strict: banned words = errors, judge must pass. "
            "normal: banned words = warnings only (default, recommended for Gemini free tier). "
            "lenient: banned words ignored, LLM judge skipped (fastest, fewest API calls)."
        ),
    ),
) -> None:
    """Run pipeline stages: discover, enrich, score, tailor, cover, pdf."""
    _bootstrap()

    if provider:
        from applyassist.llm import set_default_provider
        set_default_provider(provider)

    from applyassist.pipeline import run_pipeline

    stage_list = stages if stages else ["all"]

    # Validate stage names
    for s in stage_list:
        if s != "all" and s not in VALID_STAGES:
            console.print(
                f"[red]Unknown stage:[/red] '{s}'. "
                f"Valid stages: {', '.join(VALID_STAGES)}, all"
            )
            raise typer.Exit(code=1)

    # Gate AI stages behind Tier 2
    llm_stages = {"score", "tailor", "cover"}
    if any(s in stage_list for s in llm_stages) or "all" in stage_list:
        from applyassist.config import check_tier
        check_tier(2, "AI scoring/tailoring")

    # Validate the --validation flag value
    valid_modes = ("strict", "normal", "lenient")
    if validation not in valid_modes:
        console.print(
            f"[red]Invalid --validation value:[/red] '{validation}'. "
            f"Choose from: {', '.join(valid_modes)}"
        )
        raise typer.Exit(code=1)

    result = run_pipeline(
        stages=stage_list,
        min_score=min_score,
        dry_run=dry_run,
        stream=stream,
        workers=workers,
        validation_mode=validation,
        auto_resume=auto_resume,
        resume_wait=resume_wait,
        score_model=score_model,
        score_provider=score_provider,
        tailor_provider=tailor_provider,
        cover_provider=cover_provider,
        batch=batch,
    )

    if result.get("errors"):
        raise typer.Exit(code=1)


@app.command(name="import-alerts")
def import_alerts(
    from_gmail: bool = typer.Option(False, "--from-gmail", help="Pull alert emails directly from Gmail over IMAP (needs GMAIL_ADDRESS + GMAIL_APP_PASSWORD). No manual export."),
    gmail_days: Optional[int] = typer.Option(None, "--gmail-days", help="With --from-gmail, only look at alerts newer than N days."),
    gmail_query: Optional[str] = typer.Option(None, "--gmail-query", help="With --from-gmail, override the Gmail search query (X-GM-RAW syntax)."),
    from_files: Optional[str] = typer.Option(None, "--from-files", help="A single exported file (.eml/.mbox/.html) OR a folder of them. Defaults to ~/.applyassist/inbox."),
    include_region_locked: bool = typer.Option(False, "--include-region-locked", help="Bypass the region-lock filter: keep region-locked roles (status 'override') and score/tailor them too."),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Max new jobs to import."),
    run: bool = typer.Option(True, "--run/--no-run", help="Auto-continue into score → tailor → cover → pdf for eligible jobs (default on)."),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for tailoring."),
) -> None:
    """Import jobs from LinkedIn job-alert emails (bypasses discovery).

    Two ways to get the emails in:
      --from-gmail   Connect to Gmail over IMAP and pull every matching alert
                     automatically (recommended for 100s of emails). Set
                     GMAIL_ADDRESS and GMAIL_APP_PASSWORD in ~/.applyassist/.env.
      (default)      Read exported .eml/.mbox/.html from ~/.applyassist/inbox.

    Either way it extracts the job links, pulls each posting from LinkedIn's
    guest API (title/company/location/description, no login), inserts them
    already-enriched, excludes region-locked ones, and — by default — scores and
    tailors the eligible jobs.
    """
    _bootstrap()

    from applyassist.config import ALERTS_INBOX
    from applyassist.discovery.linkedin_alerts import import_alerts as do_import

    folder = from_files or str(ALERTS_INBOX)
    if include_region_locked:
        console.print("[yellow]--include-region-locked:[/yellow] region-locked roles will be kept and processed.")
    try:
        if from_gmail:
            console.print("[bold blue]Importing LinkedIn alerts[/bold blue] from Gmail (IMAP)")
            stats = do_import(gmail=True, gmail_query=gmail_query, gmail_days=gmail_days,
                              run=run, limit=limit, min_score=min_score,
                              bypass_region_lock=include_region_locked)
        else:
            console.print(f"[bold blue]Importing LinkedIn alerts[/bold blue] from {folder}")
            stats = do_import(folder=folder, run=run, limit=limit, min_score=min_score,
                              bypass_region_lock=include_region_locked)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    console.print(
        f"\n[green]Done.[/green] Found {stats['found']} job link(s): "
        f"{stats['new']} new imported, {stats['already']} already in DB, "
        f"{stats['failed']} could not be fetched."
    )
    cls = stats.get("classified", {})
    if cls:
        from applyassist.filters import ELIGIBLE_STATUSES
        eligible = sum(cls.get(s, 0) for s in ELIGIBLE_STATUSES)
        console.print(
            f"[dim]Classified: {eligible} eligible, "
            f"{cls.get('region_locked', 0)} region-locked (excluded), "
            f"{cls.get('unknown', 0)} unknown.[/dim]"
        )
    if stats["found"] == 0:
        if from_gmail:
            console.print(
                "[yellow]No LinkedIn job links found in Gmail.[/yellow] Check that "
                "GMAIL_ADDRESS is right, you actually have alert emails, and try a "
                "wider window (e.g. drop --gmail-days or raise it)."
            )
        else:
            console.print(
                f"[yellow]No LinkedIn job links found.[/yellow] Export your alert emails "
                f"(.eml or .mbox) into [bold]{folder}[/bold] and try again, or use "
                f"[bold]--from-gmail[/bold] to pull them automatically."
            )


@app.command(name="include-region-locked")
def include_region_locked(
    today: bool = typer.Option(False, "--today", help="Only un-exclude jobs discovered today."),
    strategy: Optional[str] = typer.Option(None, "--strategy", help="Restrict to one discovery strategy, e.g. linkedin_alert."),
    run: bool = typer.Option(False, "--run", help="After un-excluding, run score → tailor → cover → pdf on them."),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for tailoring when --run is set."),
) -> None:
    """Un-exclude region-locked jobs already in the DB so they get processed.

    Flips region-locked rows to the eligible 'override' status (excluded=0).
    Only location exclusions are touched — manual/blocklist exclusions stay.
    """
    _bootstrap()
    from applyassist.database import bypass_region_locked

    n = bypass_region_locked(today_only=today, strategy=strategy)
    scope = []
    if today:
        scope.append("discovered today")
    if strategy:
        scope.append(f"strategy={strategy}")
    scope_str = (" (" + ", ".join(scope) + ")") if scope else ""
    console.print(f"[green]Un-excluded {n} region-locked job(s){scope_str}.[/green] "
                  f"They're now eligible (status 'override').")

    if run and n:
        from applyassist.pipeline import run_pipeline
        run_pipeline(stages=["score", "tailor", "cover", "pdf"], min_score=min_score)
    elif n:
        console.print("[dim]Run them with:[/dim] ./applyassist.sh run score tailor cover pdf")


@app.command()
def apply(
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Max applications to prepare this run (default 1)."),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel browser workers."),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for job selection."),
    model: str = typer.Option("haiku", "--model", "-m", help="Claude model name."),
    continuous: bool = typer.Option(False, "--continuous", "-c", help="Run forever, polling for new jobs."),
    autopilot: bool = typer.Option(
        False, "--autopilot",
        help="OPT-IN: also click Submit (autonomous). Default is review mode, where the agent fills "
             "the form and STOPS so you review and submit. Autopilot raises ban/ToS risk -- use sparingly.",
    ),
    headless: bool = typer.Option(False, "--headless", help="Run browsers headless. Ignored in review mode (you need to see the form to submit it)."),
    url: Optional[str] = typer.Option(None, "--url", help="Apply to a specific job URL."),
    gen: bool = typer.Option(False, "--gen", help="Generate prompt file for manual debugging instead of running."),
    mark_applied: Optional[str] = typer.Option(None, "--mark-applied", help="Manually mark a job URL as applied (after you click Submit)."),
    mark_failed: Optional[str] = typer.Option(None, "--mark-failed", help="Manually mark a job URL as failed (provide URL)."),
    fail_reason: Optional[str] = typer.Option(None, "--fail-reason", help="Reason for --mark-failed."),
    reset_failed: bool = typer.Option(False, "--reset-failed", help="Reset all failed jobs for retry."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the autopilot confirmation prompt."),
) -> None:
    """Prepare applications for you to review and submit.

    Default (review mode): the agent finds a high-fit job, opens it in a visible browser,
    fills every field, uploads your tailored resume + cover letter, answers screening
    questions, then STOPS. You eyeball the form and click Submit. ApplyAssist never solves
    CAPTCHAs and (without --autopilot) never submits for you.
    """
    _bootstrap()

    from applyassist.config import check_tier, PROFILE_PATH as _profile_path
    from applyassist.database import get_connection

    # --- Utility modes (no Chrome/Claude needed) ---

    if mark_applied:
        from applyassist.apply.launcher import mark_job
        mark_job(mark_applied, "applied")
        console.print(f"[green]Marked as applied:[/green] {mark_applied}")
        return

    if mark_failed:
        from applyassist.apply.launcher import mark_job
        mark_job(mark_failed, "failed", reason=fail_reason)
        console.print(f"[yellow]Marked as failed:[/yellow] {mark_failed} ({fail_reason or 'manual'})")
        return

    if reset_failed:
        from applyassist.apply.launcher import reset_failed as do_reset
        count = do_reset()
        console.print(f"[green]Reset {count} failed job(s) for retry.[/green]")
        return

    # --- Full apply mode ---

    # Check 1: Tier 3 required (Claude Code CLI + Chrome)
    check_tier(3, "assisted-apply")

    # Check 2: Profile exists
    if not _profile_path.exists():
        console.print(
            "[red]Profile not found.[/red]\n"
            "Run [bold]applyassist init[/bold] to create your profile first."
        )
        raise typer.Exit(code=1)

    # Check 3: Tailored resumes exist (skip for --gen with --url)
    if not (gen and url):
        conn = get_connection()
        ready = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL AND applied_at IS NULL"
        ).fetchone()[0]
        if ready == 0:
            console.print(
                "[red]No tailored resumes ready.[/red]\n"
                "Run [bold]applyassist run score tailor[/bold] first to prepare applications."
            )
            raise typer.Exit(code=1)

    if gen:
        from applyassist.apply.launcher import gen_prompt, BASE_CDP_PORT
        target = url or ""
        if not target:
            console.print("[red]--gen requires --url to specify which job.[/red]")
            raise typer.Exit(code=1)
        prompt_file = gen_prompt(target, min_score=min_score, model=model)
        if not prompt_file:
            console.print("[red]No matching job found for that URL.[/red]")
            raise typer.Exit(code=1)
        mcp_path = _profile_path.parent / ".mcp-apply-0.json"
        console.print(f"[green]Wrote prompt to:[/green] {prompt_file}")
        console.print(f"\n[bold]Run manually:[/bold]")
        console.print(
            f"  claude --model {model} -p "
            f"--mcp-config {mcp_path} "
            f"--permission-mode bypassPermissions < {prompt_file}"
        )
        return

    from applyassist.apply.launcher import main as apply_main

    submit_mode = "autopilot" if autopilot else "review"

    # Review mode is human-in-the-loop: the browser MUST be visible so you can submit.
    if submit_mode == "review" and headless:
        console.print("[yellow]Ignoring --headless in review mode (you need to see the form to submit it).[/yellow]")
        headless = False

    _print_responsible_use_notice()

    if autopilot:
        console.print(
            "\n[bold red]Autopilot mode: the agent will click Submit itself.[/bold red]\n"
            "This applies without your per-application review and carries higher ban/ToS risk.\n"
            "Review mode (the default, no flag) is strongly recommended.\n"
        )
        if not yes and not typer.confirm("Proceed with autopilot?"):
            console.print("Aborted. Re-run without --autopilot for review mode.")
            raise typer.Exit(code=0)

    effective_limit = limit if limit is not None else (0 if continuous else 1)

    console.print(f"\n[bold blue]Launching ApplyAssist[/bold blue] ([bold]{submit_mode}[/bold] mode)")
    console.print(f"  Limit:    {'unlimited' if continuous else effective_limit}")
    console.print(f"  Workers:  {workers}")
    console.print(f"  Model:    {model}")
    console.print(f"  Headless: {headless}")
    if url:
        console.print(f"  Target:   {url}")
    console.print()

    apply_main(
        limit=effective_limit,
        target_url=url,
        min_score=min_score,
        headless=headless,
        model=model,
        submit_mode=submit_mode,
        continuous=continuous,
        workers=workers,
    )


@app.command()
def status() -> None:
    """Show pipeline statistics from the database."""
    _bootstrap()

    from applyassist.database import get_stats

    stats = get_stats()

    console.print("\n[bold]ApplyAssist Pipeline Status[/bold]\n")

    # Summary table
    summary = Table(title="Pipeline Overview", show_header=True, header_style="bold cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")

    summary.add_row("Total jobs discovered", str(stats["total"]))
    summary.add_row("With full description", str(stats["with_description"]))
    summary.add_row("Pending enrichment", str(stats["pending_detail"]))
    summary.add_row("Enrichment errors", str(stats["detail_errors"]))
    summary.add_row("Scored by LLM", str(stats["scored"]))
    summary.add_row("Pending scoring", str(stats["unscored"]))
    summary.add_row("Tailored resumes", str(stats["tailored"]))
    summary.add_row("Pending tailoring (7+)", str(stats["untailored_eligible"]))
    summary.add_row("Cover letters", str(stats["with_cover_letter"]))
    summary.add_row("Ready to apply", str(stats["ready_to_apply"]))
    summary.add_row("Applied", str(stats["applied"]))
    summary.add_row("Apply errors", str(stats["apply_errors"]))

    console.print(summary)

    # Score distribution
    if stats["score_distribution"]:
        dist_table = Table(title="\nScore Distribution", show_header=True, header_style="bold yellow")
        dist_table.add_column("Score", justify="center")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Bar")

        max_count = max(count for _, count in stats["score_distribution"]) or 1
        for score, count in stats["score_distribution"]:
            bar_len = int(count / max_count * 30)
            if score >= 7:
                color = "green"
            elif score >= 5:
                color = "yellow"
            else:
                color = "red"
            bar = f"[{color}]{'=' * bar_len}[/{color}]"
            dist_table.add_row(str(score), str(count), bar)

        console.print(dist_table)

    # By site
    if stats["by_site"]:
        site_table = Table(title="\nJobs by Source", show_header=True, header_style="bold magenta")
        site_table.add_column("Site")
        site_table.add_column("Count", justify="right")

        for site, count in stats["by_site"]:
            site_table.add_row(site or "Unknown", str(count))

        console.print(site_table)

    console.print()


@app.command()
def dashboard(
    static: bool = typer.Option(False, "--static", help="Write a static HTML file instead of serving the interactive dashboard."),
    port: Optional[int] = typer.Option(None, "--port", help="Port for the interactive dashboard server."),
    no_open: bool = typer.Option(False, "--no-open", help="Don't auto-open the browser."),
) -> None:
    """Open the interactive dashboard: filter roles and exclude ones you don't want.

    Exclusions persist — excluded jobs and keyword-blocklisted ones are skipped by
    scoring, tailoring, and apply. Use --static for a plain read-only HTML file.
    """
    _bootstrap()

    if static:
        from applyassist.view import open_dashboard
        open_dashboard()
        return

    from applyassist.dashboard_server import serve
    serve(port=port, open_browser=not no_open)


@app.command()
def doctor() -> None:
    """Check your setup and diagnose missing requirements."""
    import shutil
    from applyassist.config import (
        load_env, PROFILE_PATH, RESUME_PATH, RESUME_PDF_PATH,
        SEARCH_CONFIG_PATH, ENV_PATH, get_chrome_path,
    )

    load_env()

    ok_mark = "[green]OK[/green]"
    fail_mark = "[red]MISSING[/red]"
    warn_mark = "[yellow]WARN[/yellow]"

    results: list[tuple[str, str, str]] = []  # (check, status, note)

    # --- Tier 1 checks ---
    # Profile
    if PROFILE_PATH.exists():
        results.append(("profile.json", ok_mark, str(PROFILE_PATH)))
    else:
        results.append(("profile.json", fail_mark, "Run 'applyassist init' to create"))

    # Resume
    if RESUME_PATH.exists():
        results.append(("resume.txt", ok_mark, str(RESUME_PATH)))
    elif RESUME_PDF_PATH.exists():
        results.append(("resume.txt", warn_mark, "Only PDF found — plain-text needed for AI stages"))
    else:
        results.append(("resume.txt", fail_mark, "Run 'applyassist init' to add your resume"))

    # Search config
    if SEARCH_CONFIG_PATH.exists():
        results.append(("searches.yaml", ok_mark, str(SEARCH_CONFIG_PATH)))
    else:
        results.append(("searches.yaml", warn_mark, "Will use example config — run 'applyassist init'"))

    # jobspy (discovery dep installed separately)
    try:
        import jobspy  # noqa: F401
        results.append(("python-jobspy", ok_mark, "Job board scraping available"))
    except ImportError:
        results.append(("python-jobspy", warn_mark,
                        "pip install --no-deps python-jobspy && pip install pydantic tls-client requests markdownify regex"))

    # --- Tier 2 checks ---
    import os
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_local = bool(os.environ.get("LLM_URL"))
    if has_gemini:
        model = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
        results.append(("LLM API key", ok_mark, f"Gemini ({model})"))
    elif has_openai:
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        results.append(("LLM API key", ok_mark, f"OpenAI ({model})"))
    elif has_local:
        results.append(("LLM API key", ok_mark, f"Local: {os.environ.get('LLM_URL')}"))
    else:
        results.append(("LLM API key", fail_mark,
                        "Set GEMINI_API_KEY in ~/.applyassist/.env (run 'applyassist init')"))

    # --- Tier 3 checks ---
    # Claude Code CLI
    claude_bin = shutil.which("claude")
    if claude_bin:
        results.append(("Claude Code CLI", ok_mark, claude_bin))
    else:
        results.append(("Claude Code CLI", fail_mark,
                        "Install from https://claude.ai/code (needed for assisted apply)"))

    # Chrome
    try:
        chrome_path = get_chrome_path()
        results.append(("Chrome/Chromium", ok_mark, chrome_path))
    except FileNotFoundError:
        results.append(("Chrome/Chromium", fail_mark,
                        "Install Chrome or set CHROME_PATH env var (needed for assisted apply)"))

    # Node.js / npx (for Playwright MCP)
    npx_bin = shutil.which("npx")
    if npx_bin:
        results.append(("Node.js (npx)", ok_mark, npx_bin))
    else:
        results.append(("Node.js (npx)", fail_mark,
                        "Install Node.js 18+ from nodejs.org (needed for assisted apply)"))

    # --- Render results ---
    console.print()
    console.print("[bold]ApplyAssist Doctor[/bold]\n")

    col_w = max(len(r[0]) for r in results) + 2
    for check, status, note in results:
        pad = " " * (col_w - len(check))
        console.print(f"  {check}{pad}{status}  [dim]{note}[/dim]")

    console.print()

    # Tier summary
    from applyassist.config import get_tier, TIER_LABELS
    tier = get_tier()
    console.print(f"[bold]Current tier: Tier {tier} — {TIER_LABELS[tier]}[/bold]")

    if tier == 1:
        console.print("[dim]  → Tier 2 unlocks: scoring, tailoring, cover letters (needs LLM API key)[/dim]")
        console.print("[dim]  → Tier 3 unlocks: assisted apply (needs Claude Code CLI + Chrome + Node.js)[/dim]")
    elif tier == 2:
        console.print("[dim]  → Tier 3 unlocks: assisted apply (needs Claude Code CLI + Chrome + Node.js)[/dim]")

    console.print()


if __name__ == "__main__":
    app()
