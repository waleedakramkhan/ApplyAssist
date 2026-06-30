"""Location eligibility filtering.

The boards' notion of "remote" means "remote *within this country*", not
"remote worldwide". So a search returns piles of "Remote, US" roles a candidate
in another country can't be hired for. This module classifies a job's
location/description and lets the pipeline auto-exclude region-locked roles.

Eligibility policy (relative to the candidate's home location, read from the
profile — see `default_local_terms`):
  - "local"        -> the candidate's own city / country
  - "worldwide"    -> explicitly work-from-anywhere / global remote
  - "relocation"   -> a configured relocation-friendly country AND the posting
                      offers relocation or visa sponsorship
  - "region_locked"-> anything else tied to a specific country (US, Brazil, …),
                      OR a relocation-friendly country WITHOUT relocation support
  - "unknown"      -> no clear signal (bare "Remote", blank) -> kept, scorer decides

Only "region_locked" gets auto-excluded. The others are kept.

Location preferences are NOT hardcoded — they come from the user's profile:
  personal.city / personal.country (the home location), and optionally a
  `location_preferences` block: {local_terms: [...], relocation_countries: [...]}.
"""

from __future__ import annotations

import re


def default_local_terms() -> list[str]:
    """Home-location terms from the profile (personal.city + personal.country,
    plus any explicit location_preferences.local_terms). Lower-cased, de-duped."""
    try:
        from applyassist.config import load_profile
        p = load_profile()
    except Exception:
        return []
    pers = p.get("personal", {})
    terms = [pers.get("city", ""), pers.get("country", "")]
    terms += (p.get("location_preferences", {}) or {}).get("local_terms", []) or []
    return list(dict.fromkeys(t.strip().lower() for t in terms if t and t.strip()))


def default_relocation_countries() -> list[str]:
    """Relocation-friendly countries from the profile's optional
    location_preferences.relocation_countries (empty if unset)."""
    try:
        from applyassist.config import load_profile
        prefs = (load_profile().get("location_preferences", {}) or {})
    except Exception:
        return []
    return [c.strip().lower() for c in prefs.get("relocation_countries", []) if c and c.strip()]

# Signals that a posting supports relocation / visa sponsorship.
_RELOCATION = re.compile(
    r"relocat|visa sponsor|sponsor\w*\s+(your\s+)?visa|will sponsor|visa support|"
    r"sponsorship (is )?(available|provided|offered)|we sponsor|work permit support|"
    r"help(s)? (you )?relocate", re.I,
)

# Worldwide signal in a LOCATION string (locations rarely say these unless they mean it).
_WORLDWIDE = re.compile(
    r"\b(worldwide|anywhere|work\s*from\s*anywhere|remote[\s\-]*global|"
    r"any\s*(country|location|timezone)|fully\s*distributed)\b", re.I,
)
# STRICT worldwide signal for DESCRIPTIONS — must be an explicit "anywhere in the
# world" type phrase. Bare "global"/"anywhere" is too noisy in body text
# ("global team", "anywhere in the US") and must NOT count.
_WORLDWIDE_DESC = re.compile(
    r"work\s*from\s*anywhere|remote\s*from\s*anywhere|anywhere\s*in\s*the\s*world|"
    r"fully\s*remote\s*,?\s*worldwide|hire\s*(from\s*)?anywhere|"
    r"work\s*remotely\s*from\s*any\s*country|remote\s*\(?\s*global", re.I,
)

# Explicit "you must be authorized in country X" phrasing in the description.
_AUTH_LOCK = re.compile(
    r"authorized to work in the (united states|us|u\.s|uk|united kingdom|eu|canada)|"
    r"must (be|reside|are).{0,30}(united states|the us|u\.s)|"
    r"(us|u\.s\.?|united states) work authorization|"
    r"must be (located|based) in the (us|united states|uk|eu)|"
    r"eligible to work in (the )?(us|united states|uk|canada|eu)", re.I,
)

# Country / region tokens (matched against the LOCATION string). (regex, label)
_FOREIGN = [
    (r"united states", "US"), (r"\bu\.?s\.?a\b", "US"), (r"\busa\b", "US"),
    (r"\bus\b", "US"), (r"[,\-/(]\s*us\b", "US"), (r"\bu\.s\.?\b", "US"),
    (r"\bcanada\b", "Canada"), (r"\bbrazil\b", "Brazil"), (r"\bvietnam\b", "Vietnam"),
    (r"\bphilippines\b", "Philippines"), (r"\bmexico\b", "Mexico"), (r"\bindia\b", "India"),
    (r"united kingdom", "UK"), (r"\buk\b", "UK"), (r"\bgermany\b", "Germany"),
    (r"\bfrance\b", "France"), (r"\bitaly\b", "Italy"), (r"\brome\b", "Italy"),
    (r"\bspain\b", "Spain"), (r"netherlands", "Netherlands"), (r"\bpoland\b", "Poland"),
    (r"portugal", "Portugal"), (r"australia", "Australia"), (r"singapore", "Singapore"),
    (r"\bjapan\b", "Japan"), (r"\bireland\b", "Ireland"), (r"\beurope\b", "Europe"),
    (r"\bemea\b", "EMEA"), (r"\blatam\b", "LATAM"), (r"\bapac\b", "APAC"),
    (r"\bkorea\b", "Korea"), (r"\bchina\b", "China"),
    # "Nationwide" / "US-based" = remote-but-US-only.
    (r"\bnationwide\b", "US"), (r"\bus[\s\-]based\b", "US"), (r"\bu\.s\.?[\s\-]based\b", "US"),
]

# Locational cue words that, when near a country/place token in the DESCRIPTION,
# mean the ROLE is in that place (vs. the company merely mentioning a country).
_DESC_LOC_CUE = (
    r"(?:based|located|reside|residing|sits?|sitting|hiring|position is|role is|"
    r"must be (?:based|located)|candidates? (?:must be )?(?:based|located)|"
    r"work(?:ing)? from|onsite|on-site)"
)

# Full US state names (catches "Texas Remote Work", "Florida - Remote", etc.)
_US_STATES = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey",
    "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont", "virginia",
    "washington", "west virginia", "wisconsin", "wyoming", "d.c", "dc metro",
]


def _url_location_text(url: str | None) -> str:
    """Extract a location-ish blob from a job URL path.

    Workday/Greenhouse/Lever/etc. embed the location in the path, e.g.
    .../job/Mexico-Mexico-City/Lead-Engineer... . The board host is dropped to
    avoid false matches; separators become spaces so word-boundary regex works.
    """
    if not url:
        return ""
    try:
        from urllib.parse import urlparse
        path = urlparse(url).path
    except Exception:
        path = url
    return re.sub(r"[-_/+.]", " ", path).lower()


def _desc_region_lock(desc: str, relocation_countries: list[str]) -> tuple[str, str] | None:
    """Detect a region lock stated in the description body.

    Returns (status, detail) or None. Only matches a country/US-state when it's
    next to a locational cue ("based in Brazil", "candidates must be located in
    Texas") so a company merely *mentioning* a country doesn't trigger it.
    """
    # Standalone strong signals (no cue needed).
    if re.search(r"\bnationwide\b", desc) or re.search(r"\bu\.?s\.?[\s\-]based\b", desc):
        return ("region_locked", "US")
    cue = _DESC_LOC_CUE
    # Country/US-state within ~25 chars after a locational cue.
    for pat, label in _FOREIGN:
        if re.search(cue + r"[^.\n]{0,25}" + pat, desc):
            if label in (c.title() for c in relocation_countries):
                if _RELOCATION.search(desc):
                    return ("relocation", label)
                return ("region_locked", f"{label} (no relocation)")
            return ("region_locked", label)
    for state in _US_STATES:
        if re.search(cue + r"[^.\n]{0,25}\b" + re.escape(state) + r"\b", desc):
            return ("region_locked", "US")
    return None


def classify_location(location: str | None, description: str = "",
                      local_terms: list[str] | None = None,
                      allow_worldwide: bool = True,
                      relocation_countries: list[str] | None = None,
                      url: str | None = None, title: str | None = None) -> tuple[str, str]:
    """Return (status, detail). status in: local, worldwide, relocation, region_locked, unknown."""
    local_terms = local_terms if local_terms is not None else default_local_terms()
    relocation_countries = (relocation_countries if relocation_countries is not None
                            else default_relocation_countries())
    loc = (location or "").lower()
    desc = (description or "").lower()[:3000]
    urlloc = _url_location_text(url)
    ttl = (title or "").lower()
    # Location signal = location field + URL path + title (boards often leave the
    # location column blank but put it in the URL or title, e.g. "(Remote Nationwide)").
    locsig = (loc + " " + urlloc + " " + ttl).strip()

    # 1. Home location (candidate's own city/country, from profile) — always eligible
    if any(t in locsig for t in local_terms) or any(t in desc[:500] for t in local_terms):
        return ("local", "")

    # 2. Worldwide signal in the LOCATION string — eligible
    if allow_worldwide and _WORLDWIDE.search(loc):
        return ("worldwide", "")

    # 3. Relocation-friendly EU country in the LOCATION/URL — eligible only if the
    #    posting actually offers relocation / visa sponsorship.
    for country in relocation_countries:
        if re.search(r"\b" + re.escape(country) + r"\b", locsig):
            if _RELOCATION.search(desc):
                return ("relocation", country.title())
            return ("region_locked", f"{country.title()} (no relocation)")

    # 4. Any other foreign country / US state in the LOCATION/URL — region locked.
    #    (Checked BEFORE description signals — location is authoritative.)
    for pat, label in _FOREIGN:
        if re.search(pat, locsig):
            return ("region_locked", label)
    for state in _US_STATES:
        if re.search(r"\b" + re.escape(state) + r"\b", locsig):
            return ("region_locked", "US")

    # 5. Explicit work-authorization lock in the description — region locked
    if _AUTH_LOCK.search(desc):
        return ("region_locked", "work-auth required")

    # 6. Region lock stated in the description body ("based in Brazil",
    #    "Remote Nationwide", "candidates must be located in Texas", …)
    desc_lock = _desc_region_lock(desc, relocation_countries)
    if desc_lock:
        return desc_lock

    # 7. Strict worldwide phrase in the description — eligible
    if allow_worldwide and _WORLDWIDE_DESC.search(desc):
        return ("worldwide", "")

    # 8. No signal (bare "Remote", blank) — keep, let the scorer decide
    return ("unknown", "")


def apply_location_policy(conn=None, local_terms: list[str] | None = None,
                          allow_worldwide: bool = True,
                          relocation_countries: list[str] | None = None) -> int:
    """Auto-exclude region-locked jobs. Returns count newly excluded.

    Only flips not-yet-excluded jobs (won't undo a manual restore). Excluded
    jobs are skipped by scoring/tailoring/apply and hidden by default in the
    dashboard.
    """
    from applyassist.database import get_connection
    if conn is None:
        conn = get_connection()
    # Resolve location prefs once (from profile) rather than per row.
    if local_terms is None:
        local_terms = default_local_terms()
    if relocation_countries is None:
        relocation_countries = default_relocation_countries()
    rows = conn.execute(
        "SELECT url, location, full_description, title FROM jobs WHERE COALESCE(excluded,0) = 0"
    ).fetchall()
    n = 0
    for r in rows:
        status, detail = classify_location(r[1], r[2] or "", local_terms,
                                           allow_worldwide, relocation_countries,
                                           url=r[0], title=r[3])
        if status == "region_locked":
            conn.execute(
                "UPDATE jobs SET excluded = 1, excluded_reason = ? WHERE url = ?",
                (f"location: {detail}", r[0]),
            )
            n += 1
    if n:
        conn.commit()
    return n


# "override" = region lock manually bypassed (via --include-region-locked or the
# include-region-locked command). Treated as eligible everywhere a job's
# location_status is gated, so these flow through score/tailor/apply normally.
ELIGIBLE_STATUSES = ("local", "worldwide", "relocation", "override")


def classify_jobs(conn=None, reclassify: bool = False,
                  local_terms: list[str] | None = None,
                  allow_worldwide: bool = True,
                  relocation_countries: list[str] | None = None,
                  bypass_region_lock: bool = False) -> dict:
    """Classify jobs' location eligibility and store it on each row.

    Runs at discovery time using the location, URL, title AND the description
    (which the scrapers already fetch), so region-locked roles are caught and
    excluded *at the top* — before any enrichment or LLM scoring spend.

    By default only classifies rows missing a status (newly discovered). Sets
    location_status for every job; region_locked rows are also marked excluded.
    Never un-excludes a manually-excluded job. Returns {status: count}.

    If bypass_region_lock is True, roles that would be region_locked are instead
    recorded as "override" (eligible) and NOT excluded — so they flow through
    scoring/tailoring like any other job.
    """
    from applyassist.database import get_connection
    if conn is None:
        conn = get_connection()
    # Resolve location prefs once (from profile) rather than per row.
    if local_terms is None:
        local_terms = default_local_terms()
    if relocation_countries is None:
        relocation_countries = default_relocation_countries()
    where = "" if reclassify else "WHERE location_status IS NULL"
    rows = conn.execute(
        f"SELECT url, location, full_description, title, COALESCE(excluded,0) "
        f"FROM jobs {where}"
    ).fetchall()
    counts: dict = {}
    for url, location, desc, title, already_excluded in rows:
        status, detail = classify_location(location, desc or "", local_terms,
                                            allow_worldwide, relocation_countries,
                                            url=url, title=title)
        if status == "region_locked" and bypass_region_lock:
            status, detail = "override", detail
        counts[status] = counts.get(status, 0) + 1
        if status == "region_locked":
            conn.execute(
                "UPDATE jobs SET location_status = ?, excluded = 1, excluded_reason = ? WHERE url = ?",
                (status, f"location: {detail}", url),
            )
        else:
            # Eligible or unknown: record status, don't touch a manual exclusion.
            conn.execute(
                "UPDATE jobs SET location_status = ? WHERE url = ?",
                (status, url),
            )
    conn.commit()
    return counts
