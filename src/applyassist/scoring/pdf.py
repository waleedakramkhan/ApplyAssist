"""Text-to-PDF conversion for tailored resumes and cover letters.

Parses the structured text resume format, renders via an HTML/CSS template,
and exports to PDF using headless Chromium via Playwright.
"""

import logging
from html import escape
from pathlib import Path

from applyassist.config import TAILORED_DIR

log = logging.getLogger(__name__)


# ── Resume Parser ────────────────────────────────────────────────────────

def parse_resume(text: str) -> dict:
    """Parse a structured text resume into sections.

    Expects a format with header lines (name, title, location, contact)
    followed by ALL-CAPS section headers (SUMMARY, TECHNICAL SKILLS, etc.).

    Args:
        text: Full resume text.

    Returns:
        {"name": str, "title": str, "location": str, "contact": str, "sections": dict}
    """
    lines = [line.rstrip() for line in text.strip().split("\n")]

    # Header: first few lines before SUMMARY
    header_lines: list[str] = []
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip().upper() == "SUMMARY":
            body_start = i
            break
        if line.strip():
            header_lines.append(line.strip())

    name = header_lines[0] if len(header_lines) > 0 else ""
    title = header_lines[1] if len(header_lines) > 1 else ""
    # The header may have 3 or 4 lines depending on whether location is included
    location = ""
    contact = ""
    if len(header_lines) > 3:
        location = header_lines[2]
        contact = header_lines[3]
    elif len(header_lines) > 2:
        # Could be location or contact -- check for email/phone indicators
        if "@" in header_lines[2] or "|" in header_lines[2]:
            contact = header_lines[2]
        else:
            location = header_lines[2]

    # Split body into sections by ALL-CAPS headers
    sections: dict[str, str] = {}
    current_section: str | None = None
    current_lines: list[str] = []

    for line in lines[body_start:]:
        stripped = line.strip()
        # Detect section headers (all caps, no leading dash/bullet, longer than 3 chars)
        if (
            stripped
            and stripped == stripped.upper()
            and not stripped.startswith("-")
            and len(stripped) > 3
            and not stripped.startswith("\u2022")
        ):
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = stripped
            current_lines = []
        else:
            current_lines.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_lines).strip()

    return {
        "name": name,
        "title": title,
        "location": location,
        "contact": contact,
        "sections": sections,
    }


def parse_skills(text: str) -> list[tuple[str, str]]:
    """Parse skills section into (category, value) pairs.

    Args:
        text: The TECHNICAL SKILLS section text.

    Returns:
        List of (category_name, skills_string) tuples.
    """
    skills: list[tuple[str, str]] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if ":" in line:
            cat, val = line.split(":", 1)
            skills.append((cat.strip(), val.strip()))
    return skills


def parse_entries(text: str) -> list[dict]:
    """Parse experience/project entries from section text.

    Args:
        text: The EXPERIENCE or PROJECTS section text.

    Returns:
        List of {"title": str, "subtitle": str, "bullets": list[str]} dicts.
    """
    entries: list[dict] = []
    lines = text.strip().split("\n")
    current: dict | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- ") or stripped.startswith("\u2022 "):
            if current:
                current["bullets"].append(stripped[2:].strip())
        elif current is None or (
            not stripped.startswith("-")
            and not stripped.startswith("\u2022")
            and len(current.get("bullets", [])) > 0
        ):
            # New entry
            if current:
                entries.append(current)
            current = {"title": stripped, "subtitle": "", "bullets": []}
        elif current and not current["subtitle"]:
            current["subtitle"] = stripped
        else:
            if current:
                current["bullets"].append(stripped)

    if current:
        entries.append(current)

    return entries


# ── HTML Template ────────────────────────────────────────────────────────

def _split_subtitle(subtitle: str) -> tuple[str, str]:
    """Split an entry subtitle into (meta, dates).

    Subtitles look like "Tech | 2024 - Present" or "Python | AWS | Kafka". The
    part containing a year is the dates (right-aligned); the rest is meta
    (company/stack). The generic placeholder "Tech" is dropped as noise.
    """
    parts = [p.strip() for p in (subtitle or "").split("|")]
    dates, meta = "", []
    for part in parts:
        if not part:
            continue
        if not dates and any(ch.isdigit() for ch in part):
            dates = part
        elif part.lower() != "tech":
            meta.append(part)
    return " | ".join(meta), dates


def _entry_html(e: dict) -> str:
    """Render one experience/project entry: title row with right-aligned dates."""
    meta, dates = _split_subtitle(e.get("subtitle", ""))
    bullets = "".join(f"<li>{b}</li>" for b in e["bullets"])
    dates_html = f'<span class="entry-dates">{dates}</span>' if dates else ""
    meta_html = f'<div class="entry-subtitle">{meta}</div>' if meta else ""
    return (
        f'<div class="entry">'
        f'<div class="entry-head"><span class="entry-title">{e["title"]}</span>{dates_html}</div>'
        f'{meta_html}<ul>{bullets}</ul></div>'
    )


def build_html(resume: dict) -> str:
    """Build professional resume HTML from parsed data.

    Args:
        resume: Parsed resume dict from parse_resume().

    Returns:
        Complete HTML string ready for PDF rendering.
    """
    sections = resume["sections"]

    # Skills
    skills_html = ""
    if "TECHNICAL SKILLS" in sections:
        skills = parse_skills(sections["TECHNICAL SKILLS"])
        rows = ""
        for cat, val in skills:
            rows += f'<div class="skill-row"><span class="skill-cat">{cat}:</span> {val}</div>\n'
        skills_html = f'<div class="section"><div class="section-title">Technical Skills</div>{rows}</div>'

    # Experience / Projects share the same entry rendering: role on the left,
    # dates right-aligned on the same row (standard résumé layout).
    exp_html = ""
    if "EXPERIENCE" in sections:
        items = "".join(_entry_html(e) for e in parse_entries(sections["EXPERIENCE"]))
        exp_html = f'<div class="section"><div class="section-title">Experience</div>{items}</div>'

    proj_html = ""
    if "PROJECTS" in sections:
        proj_entries = parse_entries(sections["PROJECTS"])
        if proj_entries:  # skip an empty Projects heading when there are none
            items = "".join(_entry_html(e) for e in proj_entries)
            proj_html = f'<div class="section"><div class="section-title">Projects</div>{items}</div>'

    # Education
    edu_html = ""
    if "EDUCATION" in sections:
        edu_text = sections["EDUCATION"].strip()
        edu_html = f'<div class="section"><div class="section-title">Education</div><div class="edu">{edu_text}</div></div>'

    # Summary
    summary_html = ""
    if "SUMMARY" in sections:
        summary_html = f'<div class="section"><div class="section-title">Summary</div><div class="summary">{sections["SUMMARY"].strip()}</div></div>'

    # Contact line parsing
    contact = resume["contact"]
    contact_parts = [p.strip() for p in contact.split("|")] if contact else []
    contact_html = " &nbsp;|&nbsp; ".join(contact_parts)

    # Location line (may be empty)
    location_html = f'<div class="location">{resume["location"]}</div>' if resume["location"] else ""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: letter;
    margin: 0.35in 0.5in;
}}
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}
body {{
    font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.35;
    color: #1a1a1a;
}}
.header {{
    text-align: center;
    margin-bottom: 4px;
    padding-bottom: 4px;
    border-bottom: 1.5px solid #2a7ab5;
}}
.name {{
    font-size: 18pt;
    font-weight: 700;
    color: #1a3a5c;
    letter-spacing: 0.5px;
}}
.title {{
    font-size: 10.5pt;
    color: #3a6b8c;
    margin: 1px 0;
}}
.location {{
    font-size: 9pt;
    color: #555;
}}
.contact {{
    font-size: 9pt;
    color: #444;
    margin-top: 1px;
}}
.contact a {{
    color: #2c3e50;
    text-decoration: none;
}}
.section {{
    margin-top: 5px;
}}
.section-title {{
    font-size: 10pt;
    font-weight: 700;
    color: #1a3a5c;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-bottom: 1.5px solid #2a7ab5;
    padding-bottom: 1px;
    margin-bottom: 3px;
}}
.summary {{
    font-size: 9.5pt;
    color: #333;
    line-height: 1.4;
}}
.skill-row {{
    font-size: 9.5pt;
    margin: 0;
    line-height: 1.35;
}}
.skill-cat {{
    font-weight: 600;
    color: #1a3a5c;
}}
.entry {{
    margin-bottom: 4px;
    break-inside: avoid;
}}
.entry-head {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
}}
.entry-title {{
    font-weight: 600;
    font-size: 10pt;
    color: #1a3a5c;
}}
.entry-dates {{
    font-size: 9pt;
    color: #4a7a9b;
    font-style: italic;
    white-space: nowrap;
}}
.entry-subtitle {{
    font-size: 9pt;
    color: #4a7a9b;
    font-style: italic;
    margin-bottom: 1px;
}}
ul {{
    margin-left: 14px;
    padding: 0;
}}
li {{
    font-size: 9.5pt;
    margin-bottom: 1px;
    line-height: 1.35;
}}
.edu {{
    font-size: 10pt;
}}
</style>
</head>
<body>
<div class="header">
    <div class="name">{resume['name']}</div>
    <div class="title">{resume['title']}</div>
    {location_html}
    <div class="contact">{contact_html}</div>
</div>
{summary_html}
{skills_html}
{exp_html}
{proj_html}
{edu_html}
</body>
</html>"""


def _contact_from_profile() -> tuple[str, str]:
    """Return (full_name, contact_line) from the user's profile for letterheads.

    contact_line is "email | phone | linkedin" with empty parts dropped. Falls
    back to ("", "") if the profile can't be loaded.
    """
    try:
        from applyassist.config import load_profile
        p = (load_profile() or {}).get("personal", {})
    except Exception:
        return "", ""
    name = p.get("full_name") or p.get("preferred_name") or ""
    parts = [p.get("email"), p.get("phone"), p.get("linkedin_url") or p.get("portfolio_url")]
    contact = " &nbsp;|&nbsp; ".join(x for x in parts if x)
    return name, contact


def build_cover_letter_html(text: str) -> str:
    """Render a cover letter (prose) to professional letter HTML.

    Cover letters are free prose, NOT the sectioned resume format, so they get
    their own template: a name/contact letterhead, then the body paragraphs with
    the greeting and signature preserved.
    """
    name, contact = _contact_from_profile()
    # The letter body already contains "Dear ..." and the sign-off name. Drop a
    # trailing signature line that just repeats the letterhead name (avoids
    # printing the name twice) and render the rest as paragraphs split on blanks.
    raw_paras = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    paras: list[str] = []
    for i, para in enumerate(raw_paras):
        is_last = i == len(raw_paras) - 1
        if is_last and name and para.strip().lower() == name.strip().lower():
            continue  # letterhead already shows the name
        # collapse single newlines inside a paragraph into spaces
        clean = " ".join(line.strip() for line in para.split("\n"))
        paras.append(clean)
    body = "".join(f'<p class="cl-para">{escape(p)}</p>' for p in paras)
    sign = f'<div class="cl-sign">{escape(name)}</div>' if name else ""
    name_html = f'<div class="name">{escape(name)}</div>' if name else ""
    contact_html = f'<div class="contact">{contact}</div>' if contact else ""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{ size: letter; margin: 0.9in 1in; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
    font-size: 11pt; line-height: 1.5; color: #1a1a1a;
}}
.header {{ margin-bottom: 22px; padding-bottom: 10px; border-bottom: 1.5px solid #2a7ab5; }}
.name {{ font-size: 20pt; font-weight: 700; color: #1a3a5c; letter-spacing: 0.5px; }}
.contact {{ font-size: 9.5pt; color: #444; margin-top: 3px; }}
.cl-para {{ margin-bottom: 12px; text-align: left; }}
.cl-sign {{ margin-top: 18px; font-weight: 600; color: #1a3a5c; }}
</style>
</head>
<body>
<div class="header">{name_html}{contact_html}</div>
{body}
{sign}
</body>
</html>"""


# ── PDF Renderer ─────────────────────────────────────────────────────────

# Letter page geometry at 96dpi. Usable content box = page minus @page margins.
# Resume template uses 0.35in top/bottom + 0.5in sides. The width matters as much
# as the height: scrollHeight must be measured at the REAL print width, or text
# wraps differently than it prints and the fit estimate is wrong.
_RESUME_MARGIN_TB, _RESUME_MARGIN_LR = 0.35, 0.5
_LETTER_USABLE_PX = (11.0 - 2 * _RESUME_MARGIN_TB) * 96 * 0.985
_LETTER_CONTENT_W_PX = round((8.5 - 2 * _RESUME_MARGIN_LR) * 96)  # 720


def render_pdf(html: str, output_path: str, fit_one_page: bool = True,
               usable_px: float = _LETTER_USABLE_PX,
               content_width_px: int = _LETTER_CONTENT_W_PX,
               min_scale: float = 0.62, max_scale: float = 1.3) -> None:
    """Render HTML to PDF using Playwright's headless Chromium.

    If fit_one_page is set, the content is measured at the real print width and
    scaled to fill ~target_fill of one page: shrunk when it overflows (so it
    never spills to a second page) and modestly grown when it's short (so a thin
    résumé doesn't leave the bottom half of the page blank). Scale is clamped to
    [min_scale, max_scale] to stay legible and avoid comically large text.

    Args:
        html: Complete HTML string.
        output_path: Path to write the PDF file.
        fit_one_page: Scale content to one page (shrink to fit, grow to fill).
        usable_px: Usable page height in px the content must fit within.
        content_width_px: Print content width — viewport is set to this so the
            measured wrap height matches what actually prints.
        min_scale: Lower bound on the shrink factor.
        max_scale: Upper bound on the grow factor (short content).
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        # Match the viewport to the printed content width so wrapping (and thus
        # measured height) reflects the actual PDF, not the default 1280px page.
        page = browser.new_page(viewport={"width": content_width_px, "height": 1400})
        page.set_content(html, wait_until="networkidle")
        if fit_one_page:
            def height_at(scale: float) -> float:
                # CSS zoom reflows (unlike transform), so re-measure at each scale
                # — zoom changes text wrapping and height is NOT linear in scale.
                page.evaluate(f"() => {{ document.body.style.zoom = '{scale:.4f}'; }}")
                return page.evaluate(
                    "() => Math.ceil(document.body.getBoundingClientRect().height)"
                )

            ceiling = usable_px * 0.97  # one-page safety margin
            # Binary-search the largest scale in [min, max] whose rendered height
            # still fits one page. Monotonic in scale, so this converges fast and
            # both shrinks overflowing content and grows short content to fill.
            lo, hi, best = min_scale, max_scale, min_scale
            for _ in range(8):
                mid = (lo + hi) / 2
                if height_at(mid) <= ceiling:
                    best, lo = mid, mid
                else:
                    hi = mid
            height_at(best)  # apply the winning scale
        page.pdf(
            path=output_path,
            format="Letter",
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            print_background=True,
        )
        browser.close()


# ── Public API ───────────────────────────────────────────────────────────

def convert_to_pdf(
    text_path: Path, output_path: Path | None = None, html_only: bool = False
) -> Path:
    """Convert a text resume/cover letter to PDF.

    Args:
        text_path: Path to the .txt file to convert.
        output_path: Optional override for the output path. Defaults to same
            name with .pdf extension.
        html_only: If True, output HTML instead of PDF.

    Returns:
        Path to the generated PDF (or HTML) file.
    """
    text_path = Path(text_path)
    text = text_path.read_text(encoding="utf-8")
    # Cover letters are prose, not the sectioned resume format — render them with
    # the letter template (detected by the _CL filename suffix the cover stage uses).
    is_cover = text_path.stem.endswith("_CL")
    if is_cover:
        html = build_cover_letter_html(text)
    else:
        resume = parse_resume(text)
        html = build_html(resume)

    if html_only:
        out = output_path or text_path.with_suffix(".html")
        out = Path(out)
        out.write_text(html, encoding="utf-8")
        log.info("HTML generated: %s", out)
        return out

    out = output_path or text_path.with_suffix(".pdf")
    out = Path(out)
    if is_cover:
        # Letter template: 0.9in top/bottom + 1in side margins. Only shrink to
        # avoid a 2nd page; never grow a short letter to fill the page.
        usable = (11.0 - 1.8) * 96 * 0.985
        width = round((8.5 - 2.0) * 96)
        render_pdf(html, str(out), usable_px=usable, content_width_px=width, max_scale=1.0)
    else:
        render_pdf(html, str(out))
    log.info("PDF generated: %s", out)
    return out


def batch_convert(limit: int = 50) -> int:
    """Convert .txt files in TAILORED_DIR that don't have corresponding PDFs.

    Scans for .txt files (excluding _JOB.txt and _REPORT.json), checks if a
    .pdf with the same stem already exists, and converts any that are missing.

    Args:
        limit: Maximum number of files to convert.

    Returns:
        Number of PDFs generated.
    """
    if not TAILORED_DIR.exists():
        log.warning("Tailored directory does not exist: %s", TAILORED_DIR)
        return 0

    txt_files = sorted(TAILORED_DIR.glob("*.txt"))
    # Exclude _JOB.txt and _CL.txt files from resume conversion
    # (they get their own conversion calls)
    candidates = [
        f for f in txt_files
        if not f.name.endswith("_JOB.txt")
    ]

    # Filter to those without a corresponding PDF
    to_convert: list[Path] = []
    for f in candidates:
        pdf_path = f.with_suffix(".pdf")
        if not pdf_path.exists():
            to_convert.append(f)
        if len(to_convert) >= limit:
            break

    if not to_convert:
        log.info("All text files already have PDFs.")
        return 0

    log.info("Converting %d files to PDF...", len(to_convert))
    converted = 0
    for f in to_convert:
        try:
            convert_to_pdf(f)
            converted += 1
        except Exception as e:
            log.error("Failed to convert %s: %s", f.name, e)

    log.info("Done: %d/%d PDFs generated in %s", converted, len(to_convert), TAILORED_DIR)
    return converted
