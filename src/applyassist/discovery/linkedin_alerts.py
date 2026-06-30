"""Import jobs from exported LinkedIn job-alert emails.

The board scrapers (JobSpy) return mostly irrelevant, region-locked jobs. The
user's own LinkedIn job *alerts* are curated, high-signal. This module bypasses
discovery: read exported alert emails (.eml/.mbox/.html), extract the job IDs,
and pull each posting from LinkedIn's GUEST endpoint
(https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/<id>) — which returns
title/company/location/description with no auth (the same trick JobSpy uses).

Jobs are inserted pre-enriched (full_description + detail_scraped_at set) so the
normal Playwright enrichment — which LinkedIn blocks with a login wall — is
skipped. They then flow through classify → score → tailor like any other job.
"""

from __future__ import annotations

import imaplib
import logging
import mailbox
import os
import re
import time
from datetime import datetime, timezone
from email import message_from_bytes
from email.message import Message
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from applyassist.database import get_connection
from applyassist.filters import classify_jobs

log = logging.getLogger(__name__)

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_GUEST = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{}"
_CANONICAL = "https://www.linkedin.com/jobs/view/{}"

# LinkedIn job IDs appear in several link shapes in alert emails:
#   /comm/jobs/view/<id>, /jobs/view/<id>, ?currentJobId=<id>, jobPostingId=<id>
_ID_RE = re.compile(r"(?:jobs/view/|currentJobId=|jobPostingId%3D|jobPostingId=)(\d{6,})")


# ---------------------------------------------------------------------------
# Email parsing
# ---------------------------------------------------------------------------

def _html_from_message(msg: Message) -> str:
    """Extract the HTML (or text) body from an email message."""
    parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype in ("text/html", "text/plain"):
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        parts.append(payload.decode(charset, errors="replace"))
                except Exception:
                    continue
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            parts.append(payload.decode(charset, errors="replace"))
    return "\n".join(parts)


def _ids_from_text(text: str) -> set[str]:
    return set(_ID_RE.findall(text or ""))


def _ids_from_file(path: Path) -> set[str]:
    """Extract LinkedIn job IDs from one exported email file.

    Supports .eml (single message), .mbox (mailbox of many messages), and raw
    .html/.htm/.txt bodies.
    """
    ids: set[str] = set()
    suffix = path.suffix.lower()
    try:
        if suffix == ".eml":
            msg = message_from_bytes(path.read_bytes())
            ids |= _ids_from_text(_html_from_message(msg))
        elif suffix == ".mbox":
            for msg in mailbox.mbox(str(path)):
                ids |= _ids_from_text(_html_from_message(msg))
        elif suffix in (".html", ".htm", ".txt"):
            ids |= _ids_from_text(path.read_text(encoding="utf-8", errors="replace"))
    except PermissionError as e:
        # macOS TCC blocks the terminal from reading Downloads/Desktop/Documents
        # without Full Disk Access (errno 1, "Operation not permitted").
        log.error(
            "Permission denied reading %s: %s. On macOS, move the file out of "
            "Downloads/Desktop/Documents (e.g. into ~/jobs-import/), or grant your "
            "terminal Full Disk Access in System Settings, or use --from-gmail.",
            path, e,
        )
    except Exception as e:
        log.warning("Could not parse %s: %s", path.name, e)
    return ids


def parse_alert_files(path: str | Path) -> set[str]:
    """Return the set of LinkedIn job IDs from exported alert emails.

    `path` may be a single file (e.g. a Google Takeout `.mbox`) or a folder of
    files. Folders are scanned non-recursively. Supports .eml, .mbox, and raw
    .html/.htm/.txt — mix and match.
    """
    path = Path(path).expanduser()
    ids: set[str] = set()
    if not path.exists():
        return ids

    if path.is_file():
        return _ids_from_file(path)

    for child in sorted(path.iterdir()):
        if child.is_file():
            ids |= _ids_from_file(child)
    return ids


# ---------------------------------------------------------------------------
# Gmail IMAP fetch (no manual export)
# ---------------------------------------------------------------------------

_GMAIL_HOST = "imap.gmail.com"

# Gmail search syntax (X-GM-RAW). LinkedIn alerts come from a few senders; this
# matches the common ones. Override with GMAIL_ALERT_QUERY.
_DEFAULT_GMAIL_QUERY = (
    "from:jobalerts-noreply@linkedin.com OR from:jobs-noreply@linkedin.com "
    "OR (from:linkedin.com subject:alert)"
)


def gmail_credentials() -> tuple[str | None, str | None]:
    """Read Gmail IMAP credentials from the environment.

    GMAIL_ADDRESS       — the Gmail address to read from.
    GMAIL_APP_PASSWORD  — a Google *app password* (not your normal password).
                          Create one at https://myaccount.google.com/apppasswords
                          (requires 2-Step Verification enabled).
    """
    return os.environ.get("GMAIL_ADDRESS"), os.environ.get("GMAIL_APP_PASSWORD")


def fetch_gmail_ids(
    address: str,
    app_password: str,
    query: str | None = None,
    days: int | None = None,
    mailbox_name: str = '"[Gmail]/All Mail"',
    max_messages: int | None = None,
) -> set[str]:
    """Pull LinkedIn job IDs straight from Gmail over IMAP — no manual export.

    Connects read-only, searches with Gmail's native query syntax (X-GM-RAW),
    downloads each matching alert, and extracts the job IDs. Returns the set of
    LinkedIn job IDs found.
    """
    query = query or _DEFAULT_GMAIL_QUERY
    if days:
        query = f"({query}) newer_than:{days}d"

    ids: set[str] = set()
    imap = imaplib.IMAP4_SSL(_GMAIL_HOST)
    try:
        try:
            imap.login(address, app_password)
        except imaplib.IMAP4.error as e:
            raise RuntimeError(
                "Gmail login failed. Use a Google *app password* (not your normal "
                "password), and make sure IMAP is enabled in Gmail settings. "
                f"Underlying error: {e}"
            ) from e

        imap.select(mailbox_name, readonly=True)
        typ, data = imap.search(None, "X-GM-RAW", f'"{query}"')
        if typ != "OK":
            log.warning("Gmail search failed (%s) for query: %s", typ, query)
            return ids

        msg_nums = data[0].split() if data and data[0] else []
        if max_messages:
            msg_nums = msg_nums[-max_messages:]
        log.info("Gmail: %d matching alert email(s)", len(msg_nums))

        for num in msg_nums:
            typ, msg_data = imap.fetch(num, "(RFC822)")
            if typ != "OK" or not msg_data or not msg_data[0]:
                continue
            raw = msg_data[0][1]
            if not isinstance(raw, (bytes, bytearray)):
                continue
            try:
                msg = message_from_bytes(raw)
                ids |= _ids_from_text(_html_from_message(msg))
            except Exception as e:
                log.warning("Could not parse Gmail message %s: %s", num, e)
        return ids
    finally:
        try:
            imap.logout()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LinkedIn guest posting fetch
# ---------------------------------------------------------------------------

def _text(soup: BeautifulSoup, *selectors: str) -> str | None:
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            t = el.get_text(" ", strip=True)
            if t:
                return t
    return None


def fetch_guest_posting(job_id: str, client: httpx.Client | None = None,
                        max_retries: int = 3) -> dict | None:
    """Fetch one LinkedIn job from the guest endpoint. Returns a job dict or None."""
    own_client = client is None
    client = client or httpx.Client(timeout=20, follow_redirects=True,
                                    headers={"User-Agent": _UA})
    try:
        url = _GUEST.format(job_id)
        for attempt in range(max_retries):
            try:
                r = client.get(url)
            except httpx.HTTPError as e:
                log.warning("Guest fetch error for %s: %s", job_id, e)
                return None
            if r.status_code == 429:
                wait = 5 * (attempt + 1)
                log.info("LinkedIn rate-limited on %s, waiting %ds", job_id, wait)
                time.sleep(wait)
                continue
            if r.status_code != 200 or not r.text.strip():
                log.warning("Guest fetch %s -> HTTP %s", job_id, r.status_code)
                return None
            soup = BeautifulSoup(r.text, "html.parser")
            desc_el = (soup.select_one(".show-more-less-html__markup")
                       or soup.select_one(".description__text"))
            description = desc_el.get_text("\n", strip=True) if desc_el else ""
            title = _text(soup, ".top-card-layout__title", "h2", "h3")
            company = _text(soup, ".topcard__org-name-link",
                            ".top-card-layout__second-subline a")
            location = _text(soup, ".topcard__flavor--bullet",
                             ".top-card-layout__second-subline .topcard__flavor")
            if not (title or description):
                return None
            return {
                "id": job_id,
                "url": _CANONICAL.format(job_id),
                "title": title or "LinkedIn job",
                # No "LinkedIn" fallback — that's the job board, not the employer.
                # Left blank when the page scrape misses it; recovered from the
                # description by company extraction (see scoring.company).
                "company": company or "",
                "location": location or "",
                "description": description,
            }
        return None
    finally:
        if own_client:
            client.close()


# ---------------------------------------------------------------------------
# Import orchestration
# ---------------------------------------------------------------------------

def _existing_ids(conn) -> set[str]:
    rows = conn.execute(
        "SELECT url FROM jobs WHERE url LIKE '%linkedin.com/jobs/view/%'"
    ).fetchall()
    out: set[str] = set()
    for (u,) in rows:
        m = re.search(r"/jobs/view/(\d{6,})", u or "")
        if m:
            out.add(m.group(1))
    return out


def import_alerts(folder: str | Path | None = None, run: bool = True,
                  limit: int | None = None, min_score: int = 7,
                  gmail: bool = False, gmail_query: str | None = None,
                  gmail_days: int | None = None,
                  gmail_address: str | None = None,
                  gmail_password: str | None = None,
                  bypass_region_lock: bool = False) -> dict:
    """Import LinkedIn alert jobs, enrich via guest API, classify, and
    (optionally) score/tailor the eligible ones.

    Source of job IDs is either Gmail (gmail=True, via IMAP) or a folder of
    exported emails. Returns a stats dict.
    """
    conn = get_connection()
    if gmail:
        address = gmail_address or os.environ.get("GMAIL_ADDRESS")
        password = gmail_password or os.environ.get("GMAIL_APP_PASSWORD")
        if not (address and password):
            raise RuntimeError(
                "Gmail import needs GMAIL_ADDRESS and GMAIL_APP_PASSWORD set "
                "(in ~/.applyassist/.env). Create an app password at "
                "https://myaccount.google.com/apppasswords"
            )
        all_ids = fetch_gmail_ids(address, password, query=gmail_query,
                                  days=gmail_days)
    else:
        all_ids = parse_alert_files(folder)
    already = _existing_ids(conn)
    todo = sorted(all_ids - already)
    if limit:
        todo = todo[:limit]

    stats = {"found": len(all_ids), "already": len(all_ids & already),
             "new": 0, "fetched": 0, "failed": 0}

    now = datetime.now(timezone.utc).isoformat()
    with httpx.Client(timeout=20, follow_redirects=True,
                      headers={"User-Agent": _UA}) as client:
        for jid in todo:
            posting = fetch_guest_posting(jid, client=client)
            if not posting:
                stats["failed"] += 1
                continue
            stats["fetched"] += 1
            try:
                conn.execute(
                    "INSERT INTO jobs (url, title, description, location, site, company, strategy, "
                    "discovered_at, full_description, application_url, detail_scraped_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (posting["url"], posting["title"], posting["description"][:500],
                     posting["location"], "LinkedIn", posting.get("company", ""),
                     "linkedin_alert", now,
                     posting["description"], posting["url"], now),
                )
                stats["new"] += 1
            except Exception:
                pass  # duplicate (race) — ignore
            time.sleep(1.0)  # be polite to LinkedIn
        conn.commit()

    # Classify the freshly imported jobs (sets location_status, excludes region-locked
    # unless bypass_region_lock keeps them as eligible "override").
    counts = classify_jobs(conn=conn, bypass_region_lock=bypass_region_lock)
    stats["classified"] = counts

    if run and stats["new"]:
        from applyassist.pipeline import run_pipeline
        run_pipeline(stages=["score", "tailor", "cover", "pdf"], min_score=min_score)

    return stats
