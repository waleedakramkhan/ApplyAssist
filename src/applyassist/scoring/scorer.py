"""Job fit scoring: LLM-powered evaluation of candidate-job match quality.

Scores jobs on a 1-10 scale by comparing the user's resume against each
job description. All personal data is loaded at runtime from the user's
profile and resume file.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone

from applyassist.config import RESUME_PATH, load_profile
from applyassist.database import get_connection, get_jobs_by_stage
from applyassist.llm import get_client, LLMHaltError

log = logging.getLogger(__name__)


# ── Scoring Prompt ────────────────────────────────────────────────────────

SCORE_PROMPT = """You are a job fit evaluator. Given a candidate's resume and a job description, score how well the candidate fits the role.

SCORING CRITERIA:
- 9-10: Perfect match. Candidate has direct experience in nearly all required skills and qualifications.
- 7-8: Strong match. Candidate has most required skills, minor gaps easily bridged.
- 5-6: Moderate match. Candidate has some relevant skills but missing key requirements.
- 3-4: Weak match. Significant skill gaps, would need substantial ramp-up.
- 1-2: Poor match. Completely different field or experience level.

IMPORTANT FACTORS:
- Weight technical skills heavily (programming languages, frameworks, tools)
- Consider transferable experience (automation, scripting, API work)
- Factor in the candidate's project experience
- Be realistic about experience level vs. job requirements (years of experience, seniority)

RESPOND IN EXACTLY THIS FORMAT (no other text):
SCORE: [1-10]
KEYWORDS: [comma-separated ATS keywords from the job description that match or could match the candidate]
REASONING: [2-3 sentences explaining the score]"""


def _parse_score_response(response: str) -> dict:
    """Parse the LLM's score response into structured data.

    Args:
        response: Raw LLM response text.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    score = 0
    keywords = ""
    reasoning = response

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                score = int(re.search(r"\d+", line).group())
                score = max(1, min(10, score))
            except (AttributeError, ValueError):
                score = 0
        elif line.startswith("KEYWORDS:"):
            keywords = line.replace("KEYWORDS:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()

    return {"score": score, "keywords": keywords, "reasoning": reasoning}


def score_job(resume_text: str, job: dict, client=None) -> dict:
    """Score a single job against the resume.

    Args:
        resume_text: The candidate's full resume text.
        job: Job dict with keys: title, site, location, full_description.

    Returns:
        {"score": int, "keywords": str, "reasoning": str}
    """
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    messages = [
        {"role": "system", "content": SCORE_PROMPT},
        {"role": "user", "content": f"RESUME:\n{resume_text}\n\n---\n\nJOB POSTING:\n{job_text}"},
    ]

    try:
        client = client or get_client()
        response = client.chat(messages, max_tokens=512, temperature=0.2)
        return _parse_score_response(response)
    except LLMHaltError:
        # Quota exhausted / provider unreachable — do NOT record a score.
        # Propagate so the batch halts and checkpoints; this job stays unscored
        # and is retried on the next run.
        raise
    except Exception as e:
        log.error("LLM error scoring job '%s': %s", job.get("title", "?"), e)
        return {"score": 0, "keywords": "", "reasoning": f"LLM error: {e}"}


def score_one(url: str) -> dict:
    """Score a single job by URL, on demand (the dashboard 'Enrich' button).

    Bypasses the location-eligibility gate — the user explicitly chose to process
    this parked job. Writes the score to the DB. Returns a small result dict.
    """
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()
    row = conn.execute("SELECT * FROM jobs WHERE url = ?", (url,)).fetchone()
    if not row:
        return {"ok": False, "error": "job not found"}
    job = dict(zip(row.keys(), row))
    if not (job.get("full_description") or "").strip():
        return {"ok": False, "error": "no description available to score yet"}
    try:
        result = score_job(resume_text, job)
    except LLMHaltError as e:
        return {"ok": False, "error": f"LLM unavailable ({e})"}
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ? WHERE url = ?",
        (result["score"], f"{result['keywords']}\n{result['reasoning']}", now, url),
    )
    conn.commit()
    return {"ok": True, "score": result["score"], "reasoning": result["reasoning"]}


def run_scoring(limit: int = 0, rescore: bool = False, model: str | None = None,
                provider: str | None = None) -> dict:
    """Score unscored jobs that have full descriptions.

    Args:
        limit: Maximum number of jobs to score in this run.
        rescore: If True, re-score all jobs (not just unscored ones).

    Returns:
        {"scored": int, "errors": int, "elapsed": float, "distribution": list}
    """
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    # Apply the keyword blocklist first so we never spend quota scoring jobs the
    # user has chosen to exclude (e.g. "sales", "manager", a company).
    from applyassist.config import load_exclusions
    from applyassist.database import apply_blocklist
    from applyassist.filters import classify_jobs
    bl = load_exclusions()
    n_blocked = apply_blocklist(bl["title_contains"], bl["company_contains"], conn=conn)
    if n_blocked:
        log.info("Blocklist excluded %d job(s) before scoring.", n_blocked)
    # Classify any not-yet-classified jobs (sets location_status; excludes
    # region-locked). Ensures resume-scoring (`run score` without discover) still
    # gates correctly so only eligible jobs are scored.
    counts = classify_jobs(conn=conn)
    if counts:
        log.info("Location classification: %s", counts)

    if rescore:
        query = "SELECT * FROM jobs WHERE full_description IS NOT NULL AND COALESCE(excluded,0) = 0"
        if limit > 0:
            query += f" LIMIT {limit}"
        jobs = conn.execute(query).fetchall()
    else:
        jobs = get_jobs_by_stage(conn=conn, stage="pending_score", limit=limit)

    if not jobs:
        log.info("No unscored jobs with descriptions found.")
        return {"scored": 0, "errors": 0, "elapsed": 0.0, "distribution": []}

    # Convert sqlite3.Row to dicts if needed
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    total = len(jobs)
    # Optional per-stage override: score on a different provider/model than the
    # default writing model (e.g. bulk scoring on Cerebras, tailor/cover on the
    # default). provider wins over model if both are given.
    from applyassist.llm import build_client_for_model, build_client_for_provider
    score_client = None
    if provider:
        score_client = build_client_for_provider(provider)
        log.info("Scoring on provider override: %s", provider)
    elif model:
        score_client = build_client_for_model(model)
        log.info("Scoring with override model: %s", model)
    log.info("Scoring %d jobs sequentially...", total)
    t0 = time.time()
    completed = 0
    errors = 0
    halted: str | None = None
    resume_hint = ""

    for job in jobs:
        try:
            result = score_job(resume_text, job, client=score_client)
        except LLMHaltError as e:
            # Quota/connection exhausted. Stop here — everything scored so far is
            # already committed below, and unscored jobs stay pending for resume.
            halted = "rate_limit" if e.__class__.__name__ == "LLMRateLimitError" else "connection"
            resume_hint = e.resume_hint
            log.warning("Scoring halted after %d/%d jobs: %s", completed, total, e)
            break

        # Checkpoint immediately so progress survives an interruption.
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE jobs SET fit_score = ?, score_reasoning = ?, scored_at = ? WHERE url = ?",
            (result["score"], f"{result['keywords']}\n{result['reasoning']}", now, job["url"]),
        )
        conn.commit()
        completed += 1
        if result["score"] == 0:
            errors += 1

        log.info(
            "[%d/%d] score=%d  %s",
            completed, total, result["score"], job.get("title", "?")[:60],
        )

    elapsed = time.time() - t0
    log.info("Done: %d scored in %.1fs", completed, elapsed)

    # Score distribution
    dist = conn.execute("""
        SELECT fit_score, COUNT(*) FROM jobs
        WHERE fit_score IS NOT NULL
        GROUP BY fit_score ORDER BY fit_score DESC
    """).fetchall()
    distribution = [(row[0], row[1]) for row in dist]

    remaining = total - completed
    result = {
        "scored": completed,
        "errors": errors,
        "elapsed": elapsed,
        "distribution": distribution,
        "remaining": remaining,
    }
    if halted:
        result["status"] = f"halted: {halted}"
        result["halted"] = halted
        result["resume_hint"] = resume_hint
        result["message"] = (
            f"Scored {completed}/{total} before the LLM provider cut us off "
            f"({halted}). {remaining} job(s) left, progress saved — {resume_hint}."
        )
    return result
