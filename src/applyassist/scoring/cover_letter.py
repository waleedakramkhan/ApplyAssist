"""Cover letter generation: LLM-powered, profile-driven, with validation.

Generates concise, engineering-voice cover letters tailored to specific job
postings. All personal data (name, skills, achievements) comes from the user's
profile at runtime. No hardcoded personal information.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone

from applyassist.config import COVER_LETTER_DIR, RESUME_PATH, load_profile
from applyassist.database import get_connection, get_jobs_by_stage
from applyassist.llm import get_client, LLMHaltError
from applyassist.scoring.validator import (
    BANNED_WORDS,
    LLM_LEAK_PHRASES,
    extract_resume_metrics,
    fabricated_numbers,
    sanitize_text,
    validate_cover_letter,
)

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5  # max cross-run retries before giving up


# ── Prompt Builder (profile-driven) ──────────────────────────────────────

def _build_cover_letter_prompt(profile: dict, resume_text: str = "") -> str:
    """Build the cover letter system prompt from the user's profile.

    All personal data, skills, and sign-off name come from the profile.
    """
    personal = profile.get("personal", {})
    boundary = profile.get("skills_boundary", {})
    resume_facts = profile.get("resume_facts", {})

    # Preferred name for the sign-off (falls back to full name)
    sign_off_name = personal.get("preferred_name") or personal.get("full_name", "")

    # Flatten all allowed skills
    all_skills: list[str] = []
    for items in boundary.values():
        if isinstance(items, list):
            all_skills.extend(items)
    skills_str = ", ".join(all_skills) if all_skills else "the tools listed in the resume"

    # Real metrics: prefer numbers actually present in the resume over the
    # profile's (often placeholder) real_metrics list.
    real_metrics = resume_facts.get("real_metrics", [])
    preserved_projects = resume_facts.get("preserved_projects", [])
    extracted = extract_resume_metrics(resume_text)
    allowed_metrics = extracted or [m for m in real_metrics if any(c.isdigit() for c in m)]

    # Build achievement examples for the prompt
    projects_hint = ""
    if preserved_projects:
        projects_hint = f"\nKnown projects to reference: {', '.join(preserved_projects)}"

    metrics_hint = ""
    if allowed_metrics:
        metrics_hint = (
            f"\nThe ONLY numbers you may use (exactly as written, from the resume): "
            f"{', '.join(allowed_metrics)}. Never invent, inflate, or reword a number; "
            f"a claim without a real number stays qualitative."
        )

    # Build the full banned list from the validator so the prompt stays in sync
    # with what will actually be rejected — the validator checks all of these.
    all_banned = ", ".join(f'"{w}"' for w in BANNED_WORDS)
    leak_banned = ", ".join(f'"{p}"' for p in LLM_LEAK_PHRASES)

    return f"""You are an expert cover letter writer and job application strategist writing for {sign_off_name}. The single goal is to make a recruiter want to interview {sign_off_name}.

## PROCESS (do this silently, then write):
1. Read the job description and pull out the 3-5 most important qualifications, themes, or problems the team is actually trying to solve.
2. From the resume, pick the 1-2 experiences that most directly match those priorities. Ignore the rest.
3. Write the letter around that overlap. Depth on the strongest evidence beats breadth.

## STRUCTURE (3 short paragraphs, ~250-350 words):
1. HOOK — Open with something specific about THIS company or role: its product, scale, growth stage, mission, or the hard technical problem it's solving, tied to why {sign_off_name}'s background fits it. Make the role feel concrete and real from the first sentence.
2. EVIDENCE — The strongest 1-2 matching experiences from the resume, with concrete detail and real numbers. Frame each as solving the team's problem, not as a list of accomplishments.
3. CLOSE — Why this exact background fits this exact role, then a warm, confident line inviting a conversation.

## OPENING — WHAT NOT TO DO (these read as templates and kill the letter):
- Do NOT start with "I am writing to apply for..." or "I'm excited about..." or "I'm interested in...".
- Do NOT open by summarizing the job description back to them.
- Do NOT start the first paragraph, or any sentence, with "What" (no "What stands out about...", "What excites me about...").
- Do NOT start with "As a...".
- The opening must sound like a sharp, specific person with judgment wrote it, not a generator.

## VOICE:
- Natural, human, personally written. Direct, not stiff, not overly formal, not casual.
- Persuasive without exaggeration. Confident without hype.
- Vary sentence length and structure so it reads like a person.
- Never narrate yourself (BAD: "This demonstrates my commitment to X." just state the fact).
- Never hedge (BAD: "might help with some of your challenges." GOOD: "solves the same problem your team is facing.").
- No bullet points, no headings, no subject line. Prose only.

## NUMBERS:
- Use numbers, not words, for impact ("cut deployment time by 50%", not "in half") so achievements are visible at a glance.{metrics_hint}

## HARD RULES — fabrication is rejected:
- Use ONLY information supported by the resume and the job description. Invent nothing: no experience, no metrics, no qualifications, no tools.
- The candidate's real tools are ONLY: {skills_str}. Do not name any tool outside this list. If the job wants a tool not listed, write about the work, not the tool.
- Do not repeat the resume verbatim; reframe it for this role.{projects_hint}

## BANNED WORDS / PHRASES (an automated validator rejects ANY of these, do not use even once):
{all_banned}

## ALSO BANNED (meta-commentary the validator catches):
{leak_banned}

## PUNCTUATION: No em dashes or en dashes. Use commas or periods.

## SALUTATION: If a specific recruiter or hiring manager name appears in the job description, address them by name ("Dear [Name],"). Otherwise start with exactly "Dear Hiring Manager,".

## SELF-CHECK before output (silently): opening is specific and not formulaic; sounds human; no em dashes; no unsupported claims or invented numbers; clearly tailored to this role; uses real numbers; a recruiter would be more likely to interview after reading it.

Sign off with just "{sign_off_name}" on the last line.

Output ONLY the final letter. No preamble, no "Here is the cover letter:", no notes, no analysis. Start directly with the salutation."""


# ── Helpers ──────────────────────────────────────────────────────────────

def _strip_preamble(text: str) -> str:
    """Remove LLM preamble before 'Dear Hiring Manager,' if present.

    Gemini and other models sometimes output "Here is the cover letter:" or
    similar meta-commentary before the actual letter text. Strip everything
    before the first occurrence of "Dear" so the validator's start-check passes.
    """
    dear_idx = text.lower().find("dear")
    if dear_idx > 0:
        return text[dear_idx:]
    return text


# ── Core Generation ──────────────────────────────────────────────────────

def generate_cover_letter(
    resume_text: str, job: dict, profile: dict,
    max_retries: int = 3, validation_mode: str = "normal", client=None,
) -> str:
    """Generate a cover letter with fresh context on each retry + auto-sanitize.

    Same design as tailor_resume: fresh conversation per attempt, issues noted
    in the prompt, no conversation history stacking.

    Args:
        resume_text:      The candidate's resume text (base or tailored).
        job:              Job dict with title, site, location, full_description.
        profile:          User profile dict.
        max_retries:      Maximum retry attempts.
        validation_mode:  "strict", "normal", or "lenient".

    Returns:
        The cover letter text (best attempt even if validation failed).
    """
    # Use the REAL employer (company), never the job board (site='LinkedIn').
    company = (job.get("company") or "").strip()
    company_line = (
        f"COMPANY: {company}\n" if company
        else "COMPANY: (not provided — do NOT name a company; write naturally, "
             "open on the role/work, not on the employer's name)\n"
    )
    job_text = (
        f"TITLE: {job['title']}\n"
        f"{company_line}"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    avoid_notes: list[str] = []
    letter = ""
    client = client or get_client()
    cl_prompt_base = _build_cover_letter_prompt(profile, resume_text=resume_text)

    for attempt in range(max_retries + 1):
        # Fresh conversation every attempt
        prompt = cl_prompt_base
        if avoid_notes:
            prompt += "\n\n## AVOID THESE ISSUES:\n" + "\n".join(
                f"- {n}" for n in avoid_notes[-5:]
            )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                f"RESUME:\n{resume_text}\n\n---\n\n"
                f"TARGET JOB:\n{job_text}\n\n"
                "Write the cover letter:"
            )},
        ]

        letter = client.chat(messages, max_tokens=1024, temperature=0.7)
        letter = sanitize_text(letter)  # auto-fix em dashes, smart quotes
        letter = _strip_preamble(letter)  # remove any "Here is the letter:" prefix

        validation = validate_cover_letter(letter, mode=validation_mode)
        problems = list(validation["errors"])

        # Cover-letter quality hinges on killing templated/weak phrasing, so
        # promote banned-word warnings to retry triggers (in normal mode they'd
        # otherwise be silently accepted). The model gets up to max_retries
        # attempts to rewrite without them; the last attempt is kept regardless.
        for w in validation.get("warnings", []):
            if w.lower().startswith("banned"):
                problems.append(f"Rewrite without these templated/weak phrases: {w}")

        # Fabricated numbers are a hard error regardless of mode — a fake metric
        # in a cover letter is as damaging as one in the resume.
        fake_nums = fabricated_numbers(letter.lower(), resume_text.lower())
        if fake_nums:
            problems.append(f"Fabricated number(s) not in resume: {', '.join(fake_nums[:5])}")

        if not problems:
            return letter

        avoid_notes.extend(problems)
        log.debug(
            "Cover letter attempt %d/%d needs rework: %s",
            attempt + 1, max_retries + 1, problems,
        )

    return letter  # last attempt even if it still has issues


# ── Batch Entry Point ────────────────────────────────────────────────────

def run_cover_letters(min_score: int = 7, limit: int = 20,
                      validation_mode: str = "normal",
                      provider: str | None = None) -> dict:
    """Generate cover letters for high-scoring jobs that have tailored resumes.

    Args:
        min_score:       Minimum fit_score threshold.
        limit:           Maximum jobs to process.
        validation_mode: "strict", "normal", or "lenient".

    Returns:
        {"generated": int, "errors": int, "elapsed": float}
    """
    profile = load_profile()
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    # Optional per-stage provider override (e.g. cover letters on NVIDIA/Mistral
    # while the rest of the pipeline runs on the default provider).
    cover_client = None
    if provider:
        from applyassist.llm import build_client_for_provider
        cover_client = build_client_for_provider(provider)
        log.info("Cover letters on provider override: %s", provider)

    # Fetch jobs that have tailored resumes but no cover letter yet
    jobs = conn.execute(
        "SELECT * FROM jobs "
        "WHERE fit_score >= ? AND tailored_resume_path IS NOT NULL "
        "AND full_description IS NOT NULL "
        "AND (cover_letter_path IS NULL OR cover_letter_path = '') "
        "AND COALESCE(cover_attempts, 0) < ? "
        "ORDER BY fit_score DESC LIMIT ?",
        (min_score, MAX_ATTEMPTS, limit),
    ).fetchall()

    if not jobs:
        log.info("No jobs needing cover letters (score >= %d).", min_score)
        return {"generated": 0, "errors": 0, "elapsed": 0.0}

    # Convert rows to dicts
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    COVER_LETTER_DIR.mkdir(parents=True, exist_ok=True)
    log.info(
        "Generating cover letters for %d jobs (score >= %d)...",
        len(jobs), min_score,
    )
    t0 = time.time()
    completed = 0
    results: list[dict] = []
    error_count = 0

    for job in jobs:
        completed += 1
        try:
            letter = generate_cover_letter(resume_text, job, profile,
                                          validation_mode=validation_mode,
                                          client=cover_client)

            # Unique per-job prefix (job id) so same-title jobs never collide.
            from applyassist.config import asset_prefix
            prefix = asset_prefix(job)

            cl_path = COVER_LETTER_DIR / f"{prefix}_CL.txt"
            cl_path.write_text(letter, encoding="utf-8")

            # Generate PDF (best-effort)
            pdf_path = None
            try:
                from applyassist.scoring.pdf import convert_to_pdf
                pdf_path = str(convert_to_pdf(cl_path))
            except Exception:
                log.debug("PDF generation failed for %s", cl_path, exc_info=True)

            result = {
                "url": job["url"],
                "path": str(cl_path),
                "pdf_path": pdf_path,
                "title": job["title"],
                "site": job["site"],
            }
            results.append(result)

            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            log.info(
                "%d/%d [OK] | %.1f jobs/min | %s",
                completed, len(jobs), rate * 60, result["title"][:40],
            )
        except LLMHaltError:
            # Rate-limit/quota halt — stop cleanly so the batch checkpoints.
            raise
        except Exception as e:
            result = {
                "url": job["url"], "title": job["title"], "site": job["site"],
                "path": None, "pdf_path": None, "error": str(e),
            }
            error_count += 1
            results.append(result)
            log.error("%d/%d [ERROR] %s -- %s", completed, len(jobs), job["title"][:40], e)

        # Checkpoint THIS job immediately so the dashboard updates live and an
        # interrupt never re-generates a finished cover letter.
        now = datetime.now(timezone.utc).isoformat()
        if result.get("path"):
            conn.execute(
                "UPDATE jobs SET cover_letter_path=?, cover_letter_at=?, "
                "cover_attempts=COALESCE(cover_attempts,0)+1 WHERE url=?",
                (result["path"], now, result["url"]),
            )
        else:
            conn.execute(
                "UPDATE jobs SET cover_attempts=COALESCE(cover_attempts,0)+1 WHERE url=?",
                (result["url"],),
            )
        conn.commit()

    saved = sum(1 for r in results if r.get("path"))
    elapsed = time.time() - t0
    log.info("Cover letters done in %.1fs: %d generated, %d errors", elapsed, saved, error_count)

    return {
        "generated": saved,
        "errors": error_count,
        "elapsed": elapsed,
    }
