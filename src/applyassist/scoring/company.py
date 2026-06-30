"""Recover the real hiring company from a job description via the LLM.

LinkedIn-sourced jobs store site='LinkedIn' (the board) and never captured the
employer, so cover letters were addressed to "LinkedIn". The company name is
almost always present in the description — this extracts it.
"""

import logging

from applyassist.database import get_connection
from applyassist.llm import get_client, build_client_for_provider, LLMHaltError

log = logging.getLogger(__name__)

_PROMPT = (
    "Extract the HIRING COMPANY name from this job description. "
    "Return ONLY the company name — no quotes, no role, no location, no extra words. "
    "It is the employer posting the job (often named repeatedly, in an 'About us' "
    "section, or in phrases like 'at <Company>', 'join <Company>', 'bei <Company>'). "
    "Never return the job board ('LinkedIn'). If the company truly is not named, "
    "reply exactly: UNKNOWN."
)


def extract_company(description: str, title: str = "", client=None) -> str:
    """Return the hiring company name from a description, or '' if unknown."""
    text = (description or "").strip()
    if not text:
        return ""
    client = client or get_client()
    resp = client.chat(
        [{"role": "system", "content": _PROMPT},
         {"role": "user", "content": f"JOB TITLE: {title}\n\nDESCRIPTION:\n{text[:4000]}\n\nCompany name:"}],
        max_tokens=30, temperature=0,
    )
    name = (resp or "").strip().splitlines()[0].strip().strip('"').strip("*").strip() if resp else ""
    if not name or name.upper() == "UNKNOWN" or name.lower() == "linkedin" or len(name) > 80:
        return ""
    return name


def backfill_companies(provider: str | None = None, limit: int = 0,
                       only_ready: bool = False) -> dict:
    """Fill the `company` column for jobs missing it, from their description.

    Stops cleanly on a rate-limit halt (resumable). Returns {updated, total}.
    """
    conn = get_connection()
    client = build_client_for_provider(provider) if provider else get_client()
    cond = "(company IS NULL OR company = '' OR company = 'LinkedIn')"
    if only_ready:
        cond += " AND tailored_resume_path IS NOT NULL"
    q = ("SELECT url, title, full_description FROM jobs "
         f"WHERE full_description IS NOT NULL AND full_description != '' AND {cond} "
         "ORDER BY tailored_resume_path IS NULL, fit_score DESC")
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = conn.execute(q).fetchall()
    total = len(rows)
    log.info("Company backfill: %d job(s) to resolve...", total)
    done = 0
    for url, title, desc in rows:
        try:
            company = extract_company(desc, title or "", client=client)
        except LLMHaltError as e:
            log.warning("Company backfill halted after %d/%d: %s", done, total, e)
            break
        except Exception as e:
            log.warning("Company extract failed for %s: %s", url, e)
            continue
        conn.execute("UPDATE jobs SET company = ? WHERE url = ?", (company or "", url))
        conn.commit()
        done += 1
        if done % 25 == 0:
            log.info("Company backfill: %d/%d", done, total)
    log.info("Company backfill done: %d/%d resolved.", done, total)
    return {"updated": done, "total": total}
