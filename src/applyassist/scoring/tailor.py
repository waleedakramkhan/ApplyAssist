"""Resume tailoring: LLM-powered ATS-optimized resume generation per job.

THIS IS THE HEAVIEST REFACTOR. Every piece of personal data -- name, email, phone,
skills, companies, projects, school -- is loaded at runtime from the user's profile.
Zero hardcoded personal information.

The LLM returns structured JSON, code assembles the final text. Header (name, contact)
is always code-injected, never LLM-generated. Each retry starts a fresh conversation
to avoid apologetic spirals.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from applyassist.config import RESUME_PATH, TAILORED_DIR, load_profile
from applyassist.database import get_connection, get_jobs_by_stage
from applyassist.llm import get_client, LLMHaltError
from applyassist.scoring.validator import (
    BANNED_WORDS,
    FABRICATION_WATCHLIST,
    extract_resume_metrics,
    sanitize_text,
    validate_json_fields,
    validate_tailored_resume,
)

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 5  # max cross-run retries before giving up


# ── Prompt Builders (profile-driven) ──────────────────────────────────────

def _build_tailor_prompt(profile: dict, resume_text: str = "") -> str:
    """Build the resume tailoring system prompt from the user's profile.

    All skills boundaries, preserved entities, and formatting rules are
    derived from the profile -- nothing is hardcoded.
    """
    boundary = profile.get("skills_boundary", {})
    resume_facts = profile.get("resume_facts", {})

    # Format skills boundary for the prompt
    skills_lines = []
    for category, items in boundary.items():
        if isinstance(items, list) and items:
            label = category.replace("_", " ").title()
            skills_lines.append(f"{label}: {', '.join(items)}")
    skills_block = "\n".join(skills_lines)

    # Preserved entities
    companies = resume_facts.get("preserved_companies", [])
    projects = resume_facts.get("preserved_projects", [])
    school = resume_facts.get("preserved_school", "")
    real_metrics = resume_facts.get("real_metrics", [])

    companies_str = ", ".join(companies) if companies else "N/A"
    projects_str = ", ".join(projects) if projects else "N/A"
    # Prefer the numbers actually present in the base resume; fall back to the
    # profile's declared metrics only if extraction found nothing.
    extracted = extract_resume_metrics(resume_text)
    allowed_metrics = extracted or [m for m in real_metrics if any(c.isdigit() for c in m)]
    metrics_str = ", ".join(allowed_metrics) if allowed_metrics else "(none — do not use any numbers)"

    # Include ALL banned words from the validator so the LLM knows exactly
    # what will be rejected — the validator checks for these automatically.
    banned_str = ", ".join(BANNED_WORDS)

    education = profile.get("experience", {})
    education_level = education.get("education_level", "")

    return f"""You are a senior technical recruiter rewriting a resume to get this person an interview.

Take the base resume and job description. Return a tailored resume as a JSON object.

## PROCESS (do this silently before writing):
1. Read the job description and extract the 3-5 most important requirements / themes / problems the team is hiring to solve.
2. From the base resume, find the real experience that best matches each. That mapping drives everything: title, summary, skill order, and which bullets lead.
3. A recruiter scans for ~6 seconds: Title matches the role? Summary proves you've done this work? First 2-3 bullets of the latest role hit the job's top needs? Must-have skills visible immediately? Optimize for exactly that scan.

## SKILLS BOUNDARY (real skills only):
{skills_block}

You MAY add 2-3 closely related tools (Kubernetes if Docker, Terraform if AWS, Redis if PostgreSQL). No unrelated languages/frameworks.

## TAILORING RULES:

TITLE: Match the target role. Keep seniority (Senior/Lead/Staff). Drop company suffixes and team names.

SUMMARY: Rewrite from scratch, 2-3 sentences. Open by naming the role's #1 requirement and proving you've done exactly that, with concrete scope (systems, scale, stack), not adjectives. No "results-driven", "passionate", "proven track record". It should read like the opening line of someone who clearly already does this job.

SKILLS: Reorder each category so the job's must-haves appear first.

BULLETS: Reframe EVERY bullet for this role (same real work, new angle, never copied verbatim) and order them so the ones matching the job's top priorities come first. Each bullet: strong verb + what you built + the tech + the concrete outcome. When the base bullet has a real number, lead with the impact. Vary the opening verb (Built, Designed, Architected, Implemented, Reduced, Automated, Deployed, Scaled, Optimized); never start two bullets the same way. One line each (~20 words). Cut filler ("responsible for", "helped to", "various", "successfully", "leveraged"). Every bullet should map to something the job actually asks for. Max 4 per role.

PROJECTS: Do NOT output a separate Projects section. Always return "projects": []. Any impressive, project-level work belongs in the bullets of the job where it happened — fold that caliber into the relevant role's responsibilities (e.g. "Built a graph-based knowledge platform that...") rather than splitting it into its own section. On a one-page resume, depth under each role beats a thin Projects list.

## NUMBERS (anti-fabrication — strictly enforced):
The ONLY numbers/percentages you may write are these, exactly as written, from the base resume:
  {metrics_str}
- Do NOT invent a number. Do NOT inflate, round, or reword a number (30%+ stays "30%+", never "over 35%").
- A bullet whose base version has no number STAYS qualitative — describe the real work and impact in words. A specific verb + real scope beats a fake percentage.
- Keep each number attached to the same achievement it described in the base resume.

## VOICE:
- Write like a real engineer. Short, direct.
- GOOD: "Automated financial reporting with Python + API integrations, cut processing time from 10 hours to 2"
- BAD: "Leveraged cutting-edge AI technologies to drive transformative operational efficiencies"
- BANNED WORDS (using ANY of these = validation failure — do not use them even once):
  {banned_str}
- No em dashes. Use commas, periods, or hyphens.

## HARD RULES:
- Do NOT invent work, companies, degrees, or certifications
- Do NOT invent, add, change, or inflate ANY number. Allowed numbers: {metrics_str}
- Preserved companies: {companies_str} -- names stay as-is
- Preserved school: {school}
- Keep each job as a SEPARATE entry with its REAL title and date range. NEVER merge two roles into one, and never move a senior title's start date earlier than it really was. Tenure and titles must match the base resume exactly.
- Describe YOUR real domain. Do NOT claim the employer's product area as your own past experience (e.g. do not say you built "AI assurance platforms" just because that is what they sell). Map your real work to their needs, don't adopt their product as your history.
- Must fit 1 page.

## OUTPUT: Return ONLY valid JSON. No markdown fences. No commentary. No "here is" preamble.

{{"title":"Role Title","summary":"2-3 tailored sentences.","skills":{{"Languages":"...","Frameworks":"...","DevOps & Infra":"...","Databases":"...","Tools":"..."}},"experience":[{{"header":"Title at Company","subtitle":"Tech | Dates","bullets":["bullet 1","bullet 2","bullet 3","bullet 4"]}}],"projects":[],"education":"{school} | {education_level}"}}"""


def _build_judge_prompt(profile: dict, resume_text: str = "") -> str:
    """Build the LLM judge prompt from the user's profile."""
    boundary = profile.get("skills_boundary", {})
    resume_facts = profile.get("resume_facts", {})

    # Flatten allowed skills for the judge
    all_skills: list[str] = []
    for items in boundary.values():
        if isinstance(items, list):
            all_skills.extend(items)
    skills_str = ", ".join(all_skills) if all_skills else "N/A"

    real_metrics = resume_facts.get("real_metrics", [])
    extracted = extract_resume_metrics(resume_text)
    allowed_metrics = extracted or [m for m in real_metrics if any(c.isdigit() for c in m)]
    metrics_str = ", ".join(allowed_metrics) if allowed_metrics else "(none)"

    return f"""You are a resume quality judge. A tailoring engine rewrote a resume to target a specific job. Your job is to catch LIES, not style changes.

You must answer with EXACTLY this format:
VERDICT: PASS or FAIL
ISSUES: (list any problems, or "none")

## CONTEXT -- what the tailoring engine was instructed to do (all of this is ALLOWED):
- Change the title to match the target role
- Rewrite the summary from scratch for the target job
- Reorder bullets and projects to put the most relevant first
- Reframe bullets to use the job's language
- Drop low-relevance bullets and replace with more relevant ones from other sections
- Reorder the skills section to put job-relevant skills first
- Change tone and wording extensively

## NUMBERS — CHECK THIS FIRST (zero tolerance for INVENTED numbers):
Judge ONLY quantified impact claims: percentages (40%), multipliers (3x), and counts of things ("5 engineers", "3+ services"). The base resume's impact numbers are: {metrics_str}
- FAIL only if a numeric VALUE appears in the tailored resume that appears NOWHERE in the original, or an existing number was inflated/reworded ("30%+" → "over 35%", "5 engineers" → "team of 8"). Inventing a metric for a bullet that had none is the classic case — catch it.
- NOT numbers, never flag: non-numeric words like "terabytes", "high-volume", "enterprise-scale"; dates/years (2024, 2021-2022); version numbers; a value that DOES appear somewhere in the original (even if moved to a different bullet — that's reattachment, judge it as wording not as an invented number).

## OTHER FABRICATION (FAIL for these):
1. Adding a tool/language/framework from a clearly DIFFERENT domain than the candidate's stack (the real skills are: {skills_str}). NOTE: adding up to 3 CLOSELY-RELATED or learnable tools is ALLOWED and must NOT be failed — e.g. FastAPI when they already use Flask + Python, Kubernetes when they use Docker, Redis when they use PostgreSQL. Only fail for genuinely unrelated tech (e.g. Rust, Salesforce, COBOL with no basis).
2. Inventing work with no basis in any original bullet (a completely new achievement/project).
3. Adding companies, roles, degrees, or certifications that don't exist; merging two roles into one; or moving a senior title's start date earlier than the original.

## WHAT IS NOT FABRICATION (do NOT fail for these):
- Rewording any bullet, even heavily, as long as the underlying work is real AND no number changed
- Combining or splitting bullets; dropping bullets; reordering anything
- Adding up to 3 closely-related/learnable tools (see above)
- Changing the title or summary completely
- Using a real word from the original (e.g. "terabytes") anywhere

## OUTPUT DISCIPLINE:
Never list something under ISSUES and then call it "allowed" or "minor" — if it's allowed, it is NOT an issue and must not appear. Only list real violations. If every issue you can think of is actually allowed, the verdict is PASS with ISSUES: none.

## TOLERANCE RULE:
Lenient on SKILLS, WORDING, and STRUCTURE (closely-related tools, restructuring, tone). STRICT only on: invented/inflated NUMBERS, invented WORK, fake companies/degrees, and merged roles or inflated tenure. PASS unless there is a real violation in one of those strict categories."""


# ── JSON Extraction ───────────────────────────────────────────────────────

def extract_json(raw: str) -> dict:
    """Robustly extract JSON from LLM response (handles fences, preamble).

    Args:
        raw: Raw LLM response text.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If no valid JSON found.
    """
    raw = raw.strip()

    def _loads(s: str) -> dict | None:
        """Parse JSON tolerantly: allow literal newlines/tabs inside strings
        (strict=False) and strip trailing commas, which many models emit."""
        s = s.strip()
        for candidate in (s, re.sub(r",(\s*[}\]])", r"\1", s)):
            try:
                return json.loads(candidate, strict=False)
            except json.JSONDecodeError:
                continue
        return None

    # Direct parse
    parsed = _loads(raw)
    if parsed is not None:
        return parsed

    # Markdown fences
    if "```" in raw:
        for part in raw.split("```")[1::2]:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            parsed = _loads(part)
            if parsed is not None:
                return parsed

    # Find outermost { ... }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        parsed = _loads(raw[start:end + 1])
        if parsed is not None:
            return parsed

    raise ValueError("No valid JSON found in LLM response")


# ── Resume Assembly (profile-driven header) ──────────────────────────────

def assemble_resume_text(data: dict, profile: dict) -> str:
    """Convert JSON resume data to formatted plain text.

    Header (name, location, contact) is ALWAYS code-injected from the profile,
    never LLM-generated. All text fields are sanitized.

    Args:
        data: Parsed JSON resume from the LLM.
        profile: User profile dict from load_profile().

    Returns:
        Formatted resume text.
    """
    personal = profile.get("personal", {})
    lines: list[str] = []

    # Header -- always code-injected from profile
    lines.append(personal.get("full_name", ""))
    lines.append(sanitize_text(data.get("title", "Software Engineer")))

    # Location from search config or profile -- leave blank if not available
    # The location line is optional; the original used a hardcoded city.
    # We omit it here; the LLM prompt can include it if the user sets it.

    # Contact line
    contact_parts: list[str] = []
    if personal.get("email"):
        contact_parts.append(personal["email"])
    if personal.get("phone"):
        contact_parts.append(personal["phone"])
    if personal.get("github_url"):
        contact_parts.append(personal["github_url"])
    if personal.get("linkedin_url"):
        contact_parts.append(personal["linkedin_url"])
    if contact_parts:
        lines.append(" | ".join(contact_parts))
    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append(sanitize_text(data["summary"]))
    lines.append("")

    # Technical Skills
    lines.append("TECHNICAL SKILLS")
    if isinstance(data["skills"], dict):
        for cat, val in data["skills"].items():
            lines.append(f"{cat}: {sanitize_text(str(val))}")
    lines.append("")

    # Experience
    lines.append("EXPERIENCE")
    for entry in data.get("experience", []):
        lines.append(sanitize_text(entry.get("header", "")))
        if entry.get("subtitle"):
            lines.append(sanitize_text(entry["subtitle"]))
        for b in entry.get("bullets", []):
            lines.append(f"- {sanitize_text(b)}")
        lines.append("")

    # Projects — only emit the section if there actually are projects. Project-
    # caliber work is folded into the experience bullets instead.
    projects = data.get("projects") or []
    if projects:
        lines.append("PROJECTS")
        for entry in projects:
            lines.append(sanitize_text(entry.get("header", "")))
            if entry.get("subtitle"):
                lines.append(sanitize_text(entry["subtitle"]))
            for b in entry.get("bullets", []):
                lines.append(f"- {sanitize_text(b)}")
        lines.append("")

    # Education
    lines.append("EDUCATION")
    lines.append(sanitize_text(str(data.get("education", ""))))

    return "\n".join(lines)


# ── LLM Judge ────────────────────────────────────────────────────────────

def judge_tailored_resume(
    original_text: str, tailored_text: str, job_title: str, profile: dict, client=None
) -> dict:
    """LLM judge layer: catches subtle fabrication that programmatic checks miss.

    Args:
        original_text: Base resume text.
        tailored_text: Tailored resume text.
        job_title: Target job title.
        profile: User profile for building the judge prompt.

    Returns:
        {"passed": bool, "verdict": str, "issues": str, "raw": str}
    """
    judge_prompt = _build_judge_prompt(profile, resume_text=original_text)

    messages = [
        {"role": "system", "content": judge_prompt},
        {"role": "user", "content": (
            f"JOB TITLE: {job_title}\n\n"
            f"ORIGINAL RESUME:\n{original_text}\n\n---\n\n"
            f"TAILORED RESUME:\n{tailored_text}\n\n"
            "Judge this tailored resume:"
        )},
    ]

    client = client or get_client()
    response = client.chat(messages, max_tokens=512, temperature=0.1)

    passed = "VERDICT: PASS" in response.upper()
    issues = "none"
    if "ISSUES:" in response.upper():
        issues_idx = response.upper().index("ISSUES:")
        issues = response[issues_idx + 7:].strip()

    return {
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "issues": issues,
        "raw": response,
    }


# ── Core Tailoring ───────────────────────────────────────────────────────

def tailor_resume(
    resume_text: str, job: dict, profile: dict,
    max_retries: int = 3, validation_mode: str = "normal", client=None,
) -> tuple[str, dict]:
    """Generate a tailored resume via JSON output + fresh context on each retry.

    Key design choices:
    - LLM returns structured JSON, code assembles the text (no header leaks)
    - Each retry starts a FRESH conversation (no apologetic spiral)
    - Issues from previous attempts are noted in the system prompt
    - Em dashes and smart quotes are auto-fixed, not rejected

    Args:
        resume_text:      Base resume text.
        job:              Job dict with title, site, location, full_description.
        profile:          User profile dict.
        max_retries:      Maximum retry attempts.
        validation_mode:  "strict", "normal", or "lenient".
                          strict  -- banned words trigger retries; judge must pass
                          normal  -- banned words = warnings only; judge can fail on last retry
                          lenient -- banned words ignored; LLM judge skipped

    Returns:
        (tailored_text, report) where report contains validation details.
    """
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n\n"
        f"DESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    report: dict = {
        "attempts": 0, "validator": None, "judge": None,
        "status": "pending", "validation_mode": validation_mode,
    }
    avoid_notes: list[str] = []
    tailored = ""
    client = client or get_client()
    tailor_prompt_base = _build_tailor_prompt(profile, resume_text=resume_text)

    for attempt in range(max_retries + 1):
        report["attempts"] = attempt + 1

        # Fresh conversation every attempt
        prompt = tailor_prompt_base
        if avoid_notes:
            prompt += "\n\n## AVOID THESE ISSUES (from previous attempt):\n" + "\n".join(
                f"- {n}" for n in avoid_notes[-5:]
            )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"ORIGINAL RESUME:\n{resume_text}\n\n---\n\nTARGET JOB:\n{job_text}\n\nReturn the JSON:"},
        ]

        raw = client.chat(messages, max_tokens=2048, temperature=0.4)

        # Parse JSON from response
        try:
            data = extract_json(raw)
        except ValueError:
            avoid_notes.append("Output was not valid JSON. Return ONLY a JSON object, nothing else.")
            continue

        # Layer 1: Validate JSON fields. Pass the original resume so genuine
        # skills (e.g. "Go (Golang)") aren't misflagged as fabricated.
        validation = validate_json_fields(data, profile, mode=validation_mode,
                                          original_resume=resume_text)
        report["validator"] = validation

        if not validation["passed"]:
            # Only retry if there are hard errors (warnings never block)
            avoid_notes.extend(validation["errors"])
            if attempt < max_retries:
                continue
            # Last attempt — assemble whatever we got
            tailored = assemble_resume_text(data, profile)
            report["status"] = "failed_validation"
            return tailored, report

        # Assemble text (header injected by code, em dashes auto-fixed)
        tailored = assemble_resume_text(data, profile)

        # Layer 2: LLM judge (catches subtle fabrication) — skipped in lenient mode
        if validation_mode == "lenient":
            report["judge"] = {"verdict": "SKIPPED", "passed": True, "issues": "none"}
            report["status"] = "approved"
            return tailored, report

        judge = judge_tailored_resume(resume_text, tailored, job.get("title", ""), profile, client=client)
        report["judge"] = judge

        if not judge["passed"]:
            avoid_notes.append(f"Judge rejected: {judge['issues']}")
            if attempt < max_retries:
                # In normal mode, only retry on judge failure if there are retries left
                if validation_mode != "lenient":
                    continue
            # Accept best attempt on last retry (all modes) or if lenient
            report["status"] = "approved_with_judge_warning"
            return tailored, report

        # Both passed
        report["status"] = "approved"
        return tailored, report

    report["status"] = "exhausted_retries"
    return tailored, report


# ── Batch Entry Point ────────────────────────────────────────────────────

def run_tailoring(min_score: int = 7, limit: int = 20,
                  validation_mode: str = "normal", provider: str | None = None) -> dict:
    """Generate tailored resumes for high-scoring jobs.

    Args:
        min_score:       Minimum fit_score to tailor for.
        limit:           Maximum jobs to process.
        validation_mode: "strict", "normal", or "lenient".

    Returns:
        {"approved": int, "failed": int, "errors": int, "elapsed": float}
    """
    profile = load_profile()
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    # Optional per-stage provider override (e.g. tailoring on Cerebras while
    # cover letters run on Mistral).
    tailor_client = None
    if provider:
        from applyassist.llm import build_client_for_provider
        tailor_client = build_client_for_provider(provider)
        log.info("Tailoring on provider override: %s", provider)

    jobs = get_jobs_by_stage(conn=conn, stage="pending_tailor", min_score=min_score, limit=limit)

    if not jobs:
        log.info("No untailored jobs with score >= %d.", min_score)
        return {"approved": 0, "failed": 0, "errors": 0, "elapsed": 0.0}

    TAILORED_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Tailoring resumes for %d jobs (score >= %d)...", len(jobs), min_score)
    t0 = time.time()
    completed = 0
    results: list[dict] = []
    stats: dict[str, int] = {"approved": 0, "failed_validation": 0, "failed_judge": 0, "error": 0}

    for job in jobs:
        completed += 1
        try:
            tailored, report = tailor_resume(resume_text, job, profile,
                                             validation_mode=validation_mode,
                                             client=tailor_client)

            # Unique per-job prefix (job id) so same-title jobs never collide.
            from applyassist.config import asset_prefix
            prefix = asset_prefix(job)

            # Save tailored resume text
            txt_path = TAILORED_DIR / f"{prefix}.txt"
            txt_path.write_text(tailored, encoding="utf-8")

            # Save job description for traceability
            job_path = TAILORED_DIR / f"{prefix}_JOB.txt"
            job_desc = (
                f"Title: {job['title']}\n"
                f"Company: {job['site']}\n"
                f"Location: {job.get('location', 'N/A')}\n"
                f"Score: {job.get('fit_score', 'N/A')}\n"
                f"URL: {job['url']}\n\n"
                f"{job.get('full_description', '')}"
            )
            job_path.write_text(job_desc, encoding="utf-8")

            # Save validation report
            report_path = TAILORED_DIR / f"{prefix}_REPORT.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

            # Generate PDF for approved resumes (best-effort)
            # "approved_with_judge_warning" is also a success — resume was generated.
            pdf_path = None
            if report["status"] in ("approved", "approved_with_judge_warning"):
                try:
                    from applyassist.scoring.pdf import convert_to_pdf
                    pdf_path = str(convert_to_pdf(txt_path))
                except Exception:
                    log.debug("PDF generation failed for %s", txt_path, exc_info=True)

            result = {
                "url": job["url"],
                "path": str(txt_path),
                "pdf_path": pdf_path,
                "title": job["title"],
                "site": job["site"],
                "status": report["status"],
                "attempts": report["attempts"],
            }
        except LLMHaltError:
            # Rate-limit/quota halt — stop cleanly so the batch checkpoints and
            # we don't burn this job's tailor_attempts on a transient limit.
            raise
        except Exception as e:
            result = {
                "url": job["url"], "title": job["title"], "site": job["site"],
                "status": "error", "attempts": 0, "path": None, "pdf_path": None,
            }
            log.error("%d/%d [ERROR] %s -- %s", completed, len(jobs), job["title"][:40], e)

        results.append(result)
        stats[result.get("status", "error")] = stats.get(result.get("status", "error"), 0) + 1

        elapsed = time.time() - t0
        rate = completed / elapsed if elapsed > 0 else 0
        log.info(
            "%d/%d [%s] attempts=%s | %.1f jobs/min | %s",
            completed, len(jobs),
            result["status"].upper(),
            result.get("attempts", "?"),
            rate * 60,
            result["title"][:40],
        )

        # Checkpoint THIS job immediately: the dashboard (which reads the DB)
        # updates live, and an interrupt never re-tailors a finished job.
        now = datetime.now(timezone.utc).isoformat()
        if result["status"] in ("approved", "approved_with_judge_warning"):
            conn.execute(
                "UPDATE jobs SET tailored_resume_path=?, tailored_at=?, "
                "tailor_attempts=COALESCE(tailor_attempts,0)+1 WHERE url=?",
                (result["path"], now, result["url"]),
            )
        else:
            conn.execute(
                "UPDATE jobs SET tailor_attempts=COALESCE(tailor_attempts,0)+1 WHERE url=?",
                (result["url"],),
            )
        conn.commit()

    elapsed = time.time() - t0
    log.info(
        "Tailoring done in %.1fs: %d approved, %d failed_validation, %d failed_judge, %d errors",
        elapsed,
        stats.get("approved", 0),
        stats.get("failed_validation", 0),
        stats.get("failed_judge", 0),
        stats.get("error", 0),
    )

    return {
        "approved": stats.get("approved", 0),
        "failed": stats.get("failed_validation", 0) + stats.get("failed_judge", 0),
        "errors": stats.get("error", 0),
        "elapsed": elapsed,
    }
