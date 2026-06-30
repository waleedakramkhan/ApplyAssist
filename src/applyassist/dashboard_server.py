"""Interactive ApplyAssist dashboard — served locally.

Unlike the old static HTML file, this serves a small JSON API so the page is
genuinely interactive:

  GET  /                -> the dashboard page (HTML + CSS + JS, self-contained)
  GET  /api/jobs        -> all jobs + stats as JSON
  POST /api/exclude     -> {url, excluded}            exclude/restore one job
  GET  /api/blocklist   -> current keyword blocklist
  POST /api/blocklist   -> {action, field, value}     add/remove a keyword

Exclusions persist to the DB (per-job) and to ~/.applyassist/exclusions.yaml
(keyword blocklist), and the rest of the pipeline (score/tailor/apply) skips
excluded jobs. Pure stdlib — no extra dependencies.
"""

from __future__ import annotations

import json
import mimetypes
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rich.console import Console

from applyassist.config import load_exclusions, save_exclusions
from applyassist.database import (
    apply_blocklist,
    get_all_jobs,
    get_connection,
    set_excluded,
)

console = Console()


# ---------------------------------------------------------------------------
# Data shaping
# ---------------------------------------------------------------------------

def _remote_hint(location: str | None) -> bool:
    loc = (location or "").lower()
    return any(k in loc for k in ("remote", "anywhere", "work from home", "wfh", "distributed"))


def _jobs_payload() -> dict:
    rows = get_all_jobs()
    jobs = []
    sites = set()
    for r in rows:
        reasoning = (r.get("score_reasoning") or "").split("\n")
        sites.add(r.get("site") or "?")
        jobs.append({
            "url": r["url"],
            "title": r.get("title") or "Untitled",
            "salary": r.get("salary") or "",
            "location": r.get("location") or "",
            "site": r.get("site") or "?",
            "company": (r.get("company") or "").strip(),
            "apply_url": (au if (au := (r.get("application_url") or "").strip())
                          and au.lower() != "none" else r["url"]),
            "score": r.get("fit_score"),
            "keywords": reasoning[0][:140] if reasoning else "",
            "reasoning": reasoning[1][:240] if len(reasoning) > 1 else "",
            "desc_len": len(r.get("full_description") or ""),
            "tailored": bool(r.get("tailored_resume_path")),
            "cover": bool(r.get("cover_letter_path")),
            "applied": bool(r.get("applied_at")),
            "excluded": bool(r.get("excluded")),
            "excluded_reason": r.get("excluded_reason") or "",
            "remote": _remote_hint(r.get("location")),
            "loc_status": r.get("location_status") or "unknown",
            "scored": r.get("score") is not None,
        })
    conn = get_connection()
    f = lambda s: conn.execute(s).fetchone()[0]
    stats = {
        "total": len(jobs),
        "scored": f("SELECT COUNT(*) FROM jobs WHERE fit_score IS NOT NULL"),
        "high_fit": f("SELECT COUNT(*) FROM jobs WHERE fit_score >= 7 AND COALESCE(excluded,0)=0"),
        "excluded": f("SELECT COUNT(*) FROM jobs WHERE COALESCE(excluded,0)=1"),
        "ready": f("SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL "
                   "AND applied_at IS NULL AND COALESCE(excluded,0)=0"),
    }
    return {"jobs": jobs, "stats": stats, "sites": sorted(s for s in sites if s)}


def _asset_path(job_url: str, kind: str) -> Path | None:
    """Resolve the on-disk path of a job's generated résumé or cover letter.

    kind is "resume" or "cover". Returns the Path if the row exists and the file
    is present, else None.
    """
    col = "tailored_resume_path" if kind == "resume" else "cover_letter_path"
    conn = get_connection()
    row = conn.execute(
        f"SELECT {col} FROM jobs WHERE url = ?", (job_url,)
    ).fetchone()
    if not row or not row[0]:
        return None
    p = Path(row[0]).expanduser()
    # The DB usually stores the .txt path; prefer the rendered .pdf if it exists
    # so the dashboard shows the polished document, not raw text.
    pdf = p.with_suffix(".pdf")
    if pdf.is_file():
        return pdf
    return p if p.is_file() else None


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def _make_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence default request logging
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj).encode("utf-8"))

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return {}

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/" or path.startswith("/index"):
                self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/jobs":
                self._json(_jobs_payload())
            elif path == "/api/blocklist":
                self._json(load_exclusions())
            elif path == "/api/asset":
                self._serve_asset(parse_qs(parsed.query))
            elif path == "/api/applypanel":
                self._apply_panel((parse_qs(parsed.query).get("url") or [""])[0])
            else:
                self._json({"error": "not found"}, 404)

        def _apply_panel(self, url: str):
            """Return everything needed to fill an application by hand: contact,
            salary (both currencies), work-auth answer, and the cover-letter text."""
            if not url:
                return self._json({"error": "url required"}, 400)
            from applyassist.config import load_profile
            from applyassist.database import get_connection
            conn = get_connection()
            row = conn.execute(
                "SELECT title, company, location, application_url, url, cover_letter_path, "
                "tailored_resume_path FROM jobs WHERE url=?", (url,)).fetchone()
            if not row:
                return self._json({"error": "job not found"}, 404)
            p = load_profile()
            pers = p.get("personal", {})
            comp = p.get("compensation", {})
            wa = p.get("work_authorization", {})
            # Read the cover TEXT (.txt), never the binary .pdf, for paste-in fields.
            cover_text = ""
            if row[5]:
                txt = Path(row[5]).expanduser().with_suffix(".txt")
                if txt.is_file():
                    try:
                        cover_text = txt.read_text(encoding="utf-8")
                    except OSError:
                        pass
            # On-disk PDF paths for file-upload dialogs (copy → paste into the
            # file picker's path field).
            def _pdf(dbpath):
                if not dbpath:
                    return ""
                pdf = Path(dbpath).expanduser().with_suffix(".pdf")
                return str(pdf) if pdf.is_file() else ""
            resume_pdf = _pdf(row[6])
            cover_pdf = _pdf(row[5])
            # Monthly salary = annual / 12 (some forms ask monthly).
            def _monthly(annual):
                try:
                    return str(round(int(annual) / 12))
                except (ValueError, TypeError):
                    return ""
            usd_mo = _monthly(comp.get("salary_expectation"))
            pkr_mo = _monthly(comp.get("salary_secondary_amount"))
            fields = [
                ["Full name", pers.get("full_name", "")],
                ["Email", pers.get("email", "")],
                ["Phone", pers.get("phone", "")],
                ["Location", ", ".join(x for x in [pers.get("city"), pers.get("country")] if x)],
                ["LinkedIn", pers.get("linkedin_url", "")],
                ["GitHub", pers.get("github_url", "")],
                ["Salary (USD/yr)", f"{comp.get('salary_expectation','')}"],
                ["Salary (USD/mo)", usd_mo],
                ["Salary (PKR/yr)", f"{comp.get('salary_secondary_amount','')}"],
                ["Salary (PKR/mo)", pkr_mo],
                ["Work authorization", wa.get("work_permit_type", "")],
                ["Need sponsorship?", "Yes" if wa.get("require_sponsorship") else "No"],
                ["Résumé PDF path", resume_pdf],
                ["Cover PDF path", cover_pdf],
            ]
            self._json({
                "title": row[0], "company": row[1] or "", "location": row[2] or "",
                "apply_url": (au if (au := (row[3] or "").strip()) and au.lower() != "none"
                              else row[4]), "url": row[4],
                "fields": [[k, v] for k, v in fields if v != ""],
                "cover_text": cover_text,
            })

        def _serve_asset(self, qs: dict):
            url = (qs.get("url") or [""])[0]
            kind = (qs.get("kind") or [""])[0]
            if not url or kind not in ("resume", "cover"):
                return self._json({"error": "url and kind=resume|cover required"}, 400)
            fpath = _asset_path(url, kind)
            if fpath is None:
                return self._json({"error": "asset not found"}, 404)
            try:
                body = fpath.read_bytes()
            except OSError as e:
                return self._json({"error": f"could not read asset: {e}"}, 500)
            ctype = mimetypes.guess_type(fpath.name)[0] or "application/octet-stream"
            if ctype.startswith("text/"):
                ctype += "; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            # inline so PDFs/text open in the browser tab; filename for "save as".
            self.send_header("Content-Disposition", f'inline; filename="{fpath.name}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            data = self._read_json()
            if self.path == "/api/exclude":
                url = data.get("url")
                if not url:
                    return self._json({"error": "url required"}, 400)
                set_excluded(url, bool(data.get("excluded", True)),
                             reason=data.get("reason", "dashboard"))
                self._json({"ok": True})
            elif self.path == "/api/enrich":
                url = data.get("url")
                if not url:
                    return self._json({"error": "url required"}, 400)
                from applyassist.scoring.scorer import score_one
                self._json(score_one(url))
            elif self.path == "/api/mark":
                url = data.get("url")
                if not url:
                    return self._json({"error": "url required"}, 400)
                from applyassist.database import get_connection
                from datetime import datetime, timezone
                conn = get_connection()
                if bool(data.get("applied", True)):
                    conn.execute("UPDATE jobs SET applied_at=?, apply_status='applied' WHERE url=?",
                                 (datetime.now(timezone.utc).isoformat(), url))
                else:
                    conn.execute("UPDATE jobs SET applied_at=NULL, "
                                 "apply_status=NULL WHERE url=?", (url,))
                conn.commit()
                self._json({"ok": True})
            elif self.path == "/api/blocklist":
                bl = load_exclusions()
                field = "title_contains" if data.get("field") == "title" else "company_contains"
                value = (data.get("value") or "").strip()
                action = data.get("action")
                if action == "add" and value:
                    if value not in bl[field]:
                        bl[field].append(value)
                elif action == "remove":
                    bl[field] = [t for t in bl[field] if t != value]
                save_exclusions(bl)
                # Re-apply so existing matches are excluded immediately.
                newly = apply_blocklist(bl["title_contains"], bl["company_contains"])
                self._json({"ok": True, "blocklist": bl, "newly_excluded": newly})
            else:
                self._json({"error": "not found"}, 404)

    return Handler


def _free_port(preferred: int = 8848) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def serve(port: int | None = None, open_browser: bool = True) -> None:
    """Start the dashboard server and (optionally) open it in the browser."""
    port = port or _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _make_handler())
    url = f"http://127.0.0.1:{port}/"
    console.print(f"[green]ApplyAssist dashboard:[/green] [bold]{url}[/bold]")
    console.print("[dim]Filter, exclude roles, and manage the keyword blocklist live. "
                  "Ctrl+C to stop.[/dim]")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/dim]")
    finally:
        httpd.server_close()


# ---------------------------------------------------------------------------
# The page (self-contained; JS uses relative API URLs)
# ---------------------------------------------------------------------------

_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ApplyAssist Dashboard</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif; background:#0f172a; color:#e2e8f0; padding:1.5rem; }
  h1 { font-size:1.6rem; font-weight:700; }
  .subtitle { color:#94a3b8; margin:.25rem 0 1.25rem; font-size:.9rem; }
  .summary { display:flex; gap:.75rem; flex-wrap:wrap; margin-bottom:1.25rem; }
  .stat { background:#1e293b; border-radius:10px; padding:.75rem 1rem; min-width:120px; cursor:pointer; border:2px solid transparent; transition:border-color .12s, background .12s; }
  .stat:hover { background:#243449; }
  .stat.active { border-color:#e2e8f0; background:#243449; }
  .stat .n { font-size:1.5rem; font-weight:700; }
  .stat .l { color:#94a3b8; font-size:.75rem; }
  .stat.high .n{color:#f59e0b} .stat.ready .n{color:#10b981} .stat.exc .n{color:#ef4444} .stat.scored .n{color:#60a5fa}

  .controls { background:#1e293b; border-radius:12px; padding:1rem; margin-bottom:1.25rem; display:flex; gap:.6rem; flex-wrap:wrap; align-items:center; }
  .controls label { color:#94a3b8; font-size:.78rem; font-weight:600; }
  input,select { background:#334155; border:1px solid #475569; color:#e2e8f0; padding:.4rem .6rem; border-radius:6px; font-size:.8rem; }
  input[type=text]{ width:200px; } input::placeholder{color:#64748b;}
  .btn { background:#334155; border:none; color:#cbd5e1; padding:.4rem .7rem; border-radius:6px; cursor:pointer; font-size:.78rem; }
  .btn:hover{ background:#475569; } .btn.active{ background:#60a5fa; color:#0f172a; font-weight:600; }
  .toggle { display:flex; align-items:center; gap:.35rem; font-size:.78rem; color:#cbd5e1; }
  .hint { font-size:.72rem; color:#64748b; }

  .blocklist { background:#1e293b; border-radius:12px; padding:1rem; margin-bottom:1.25rem; }
  .blocklist h3 { font-size:.9rem; color:#94a3b8; margin-bottom:.6rem; }
  .chips { display:flex; gap:.4rem; flex-wrap:wrap; margin-bottom:.6rem; }
  .chip { background:#7f1d1d; color:#fecaca; padding:.2rem .5rem; border-radius:999px; font-size:.72rem; display:flex; gap:.4rem; align-items:center; }
  .chip b { cursor:pointer; }
  .bl-add { display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; }

  .count { color:#94a3b8; font-size:.82rem; margin-bottom:.75rem; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:.9rem; }
  .card { background:#1e293b; border-radius:10px; padding:.9rem; border-left:3px solid #334155; }
  .card.exc { opacity:.5; border-left-color:#ef4444; }
  .card .top { display:flex; align-items:center; gap:.5rem; margin-bottom:.4rem; }
  .pill { min-width:1.6rem; height:1.6rem; border-radius:6px; color:#0f172a; font-weight:700; font-size:.8rem; display:inline-flex; align-items:center; justify-content:center; }
  .ti { color:#e2e8f0; text-decoration:none; font-weight:600; font-size:.92rem; }
  .ti:hover{ color:#60a5fa; }
  .meta { display:flex; gap:.35rem; flex-wrap:wrap; margin:.35rem 0; }
  .tag { font-size:.7rem; padding:.12rem .45rem; border-radius:4px; background:#334155; color:#94a3b8; }
  .tag.rem{ background:#064e3b; color:#6ee7b7; } .tag.loc{ background:#1e3a5f; color:#93c5fd; }
  .kw { font-size:.72rem; color:#10b981; margin:.2rem 0; }
  .rs { font-size:.72rem; color:#94a3b8; font-style:italic; margin-bottom:.4rem; }
  .badges { font-size:.68rem; color:#64748b; margin:.2rem 0; display:flex; gap:.4rem; flex-wrap:wrap; align-items:center; }
  .badge { padding:.05rem .4rem; border-radius:4px; font-weight:600; }
  .badge.resume { color:#6ee7b7; background:#10b98122; }
  .badge.cover  { color:#93c5fd; background:#3b82f622; }
  .badge.applied{ color:#fcd34d; background:#f59e0b22; }
  .badge.desc   { color:#94a3b8; background:#33415522; font-weight:400; }
  .foot { display:flex; gap:.4rem; justify-content:flex-start; flex-wrap:wrap; margin-top:.5rem; }
  .link { font-size:.76rem; text-decoration:none; padding:.28rem .7rem; border-radius:6px; }
  .apply { color:#60a5fa; border:1px solid #60a5fa55; } .apply:hover{ background:#60a5fa22; }
  .asset { color:#6ee7b7; border:1px solid #10b98155; } .asset:hover{ background:#10b98122; }
  .exbtn { color:#fca5a5; border:1px solid #ef444455; cursor:pointer; background:none; font-size:.76rem; padding:.28rem .7rem; border-radius:6px; }
  .exbtn:hover{ background:#ef444422; } .exbtn.restore{ color:#6ee7b7; border-color:#10b98155; }
  .exbtn.enrich{ color:#c4b5fd; border-color:#8b5cf655; } .exbtn.enrich:hover{ background:#8b5cf622; }
  .exbtn.markapplied{ color:#6ee7b7; border-color:#10b98155; } .exbtn.markapplied:hover{ background:#10b98122; }
  .exbtn.applied-undo{ color:#fcd34d; border-color:#f59e0b55; } .exbtn.applied-undo:hover{ background:#f59e0b22; }
  .exbtn.focus{ color:#fde047; border-color:#eab30855; font-weight:600; } .exbtn.focus:hover{ background:#eab30822; }
  /* Apply Focus panel */
  #fpwrap { position:fixed; inset:0; background:rgba(0,0,0,.6); display:none; z-index:50; }
  #fpwrap.open { display:flex; justify-content:flex-end; }
  #fp { width:min(560px,95vw); height:100%; background:#0f172a; border-left:1px solid #334155; overflow-y:auto; padding:1.2rem 1.4rem; box-shadow:-8px 0 30px rgba(0,0,0,.5); }
  #fp h2 { margin:.2rem 0 .1rem; font-size:1.1rem; color:#e2e8f0; }
  #fp .co { color:#93c5fd; font-weight:600; } #fp .loc { color:#94a3b8; font-size:.8rem; }
  #fp .fprow { display:flex; align-items:center; gap:.5rem; margin:.3rem 0; }
  #fp .fplabel { width:135px; color:#94a3b8; font-size:.76rem; flex-shrink:0; }
  #fp .fpval { flex:1; color:#e2e8f0; font-size:.82rem; word-break:break-word; }
  #fp .cpbtn { background:#1e293b; color:#93c5fd; border:1px solid #334155; border-radius:5px; padding:.18rem .55rem; font-size:.7rem; cursor:pointer; flex-shrink:0; }
  #fp .cpbtn:hover{ background:#334155; } #fp .cpbtn.done{ color:#6ee7b7; border-color:#10b98155; }
  #fp .cover { width:100%; min-height:230px; background:#1e293b; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:.6rem; font-size:.78rem; font-family:inherit; margin-top:.3rem; }
  #fp .fpactions { display:flex; gap:.5rem; flex-wrap:wrap; margin:1rem 0; position:sticky; top:0; background:#0f172a; padding:.5rem 0; }
  #fp .big { padding:.5rem .9rem; border-radius:7px; font-size:.85rem; cursor:pointer; border:1px solid; font-weight:600; }
  #fp .big.open{ color:#60a5fa; border-color:#60a5fa; background:none; }
  #fp .big.done{ color:#0f172a; background:#10b981; border-color:#10b981; }
  #fp .big.skip{ color:#fca5a5; background:none; border-color:#ef444455; }
  #fp .big.close{ color:#94a3b8; background:none; border-color:#334155; }
  #fp .sec { color:#64748b; font-size:.72rem; text-transform:uppercase; letter-spacing:.05em; margin:1rem 0 .3rem; }
  .exbtn:disabled{ opacity:.6; cursor:wait; }
  .pager { display:flex; gap:.6rem; align-items:center; justify-content:center; margin:1.25rem 0 2rem; }
  .pager button { background:#1e293b; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:.4rem .9rem; font-size:.8rem; cursor:pointer; }
  .pager button:hover:not(:disabled){ background:#334155; }
  .pager button:disabled{ opacity:.4; cursor:default; }
  .pager .pg { color:#94a3b8; font-size:.8rem; }
</style>
</head>
<body>
<h1>ApplyAssist Dashboard</h1>
<p class="subtitle" id="sub">Loading…</p>

<div class="summary" id="summary"></div>

<div class="blocklist">
  <h3>Keyword blocklist — auto-excludes matching jobs from scoring/tailoring/apply</h3>
  <div class="chips" id="chips"></div>
  <div class="bl-add">
    <input type="text" id="blval" placeholder="e.g. sales, manager, recruiter…">
    <select id="blfield"><option value="title">in title</option><option value="company">in company</option></select>
    <button class="btn" onclick="addBlock()">Add to blocklist</button>
  </div>
</div>

<div class="controls">
  <label>Search</label>
  <input type="text" id="q" placeholder="title, company, location…" oninput="render()">
  <label>Min score</label>
  <select id="minscore" onchange="render()">
    <option value="0">any</option><option value="5">5+</option>
    <option value="7" selected>7+</option><option value="8">8+</option><option value="9">9+</option>
  </select>
  <label>Site</label>
  <select id="site" onchange="render()"><option value="">all</option></select>
  <label>Sort</label>
  <select id="sort" onchange="render()">
    <option value="score">score</option><option value="title">title</option><option value="site">site</option>
  </select>
  <span class="toggle"><input type="checkbox" id="remoteonly" onchange="render()"> remote only</span>
  <span class="hint">Tip: click a stat tile above to filter (Total / Scored / Strong / Ready / Excluded).</span>
</div>

<div class="count" id="count"></div>
<div class="grid" id="grid"></div>
<div class="pager" id="pager"></div>

<div id="fpwrap" onclick="if(event.target===this)closeFocus()">
  <div id="fp"></div>
</div>

<script>
let DATA = {jobs:[], stats:{}, sites:[]};

async function load() {
  DATA = await (await fetch('/api/jobs')).json();
  const s = DATA.stats;
  document.getElementById('sub').textContent =
    `${s.total} jobs · ${s.scored} scored · ${s.high_fit} strong (7+) · ${s.ready} ready · ${s.excluded} excluded`;
  document.getElementById('summary').innerHTML =
    stat('total', s.total, 'Total', 'all') + stat('scored', s.scored, 'Scored', 'scored') +
    stat('high', s.high_fit, 'Strong 7+', 'strong') + stat('ready', s.ready, 'Ready to apply', 'ready') +
    stat('exc', s.excluded, 'Excluded', 'excluded');
  const sel = document.getElementById('site');
  sel.innerHTML = '<option value="">all</option>' + DATA.sites.map(x=>`<option>${esc(x)}</option>`).join('');
  await loadBlock();
  highlightTiles();
  render();
}
function stat(c,n,l,view){ return `<div class="stat ${c}" id="tile-${view}" onclick="setView('${view}')" title="Filter: ${l}"><div class="n">${n}</div><div class="l">${l}</div></div>`; }

// Stat tiles act as scope filters. activeView selects the base set; the
// dropdowns/search refine within it.
let activeView = 'strong';  // matches the default Min score 7+
function setView(v){
  activeView = v;
  // Keep the Min-score dropdown consistent with the chosen tile.
  document.getElementById('minscore').value = (v === 'strong') ? '7' : '0';
  curPage = 1; lastSig = '';
  highlightTiles();
  render();
}
function highlightTiles(){
  ['all','scored','strong','ready','excluded'].forEach(v=>{
    const el = document.getElementById('tile-'+v);
    if(el) el.classList.toggle('active', v === activeView);
  });
}
function esc(s){ return (s||'').replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

async function loadBlock(){
  const bl = await (await fetch('/api/blocklist')).json();
  const chips = [];
  for (const t of bl.title_contains) chips.push(chip('title', t, 'title'));
  for (const c of bl.company_contains) chips.push(chip('company', c, 'company'));
  document.getElementById('chips').innerHTML = chips.join('') || '<span style="color:#64748b;font-size:.75rem">none yet</span>';
}
function chip(field, val, label){ return `<span class="chip">${esc(val)} <span style="opacity:.6">(${label})</span> <b onclick="rmBlock('${field}','${esc(val)}')">✕</b></span>`; }

async function addBlock(){
  const value = document.getElementById('blval').value.trim();
  if(!value) return;
  const field = document.getElementById('blfield').value;
  const r = await (await fetch('/api/blocklist',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'add', field, value})})).json();
  document.getElementById('blval').value='';
  await load();
}
async function rmBlock(field, value){
  await fetch('/api/blocklist',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'remove', field, value})});
  await load();
}

async function toggleExclude(url, excluded){
  await fetch('/api/exclude',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({url, excluded})});
  const j = DATA.jobs.find(x=>x.url===url); if(j) j.excluded = excluded;
  // refresh stats
  const s = DATA.stats; s.excluded += excluded?1:-1; if(j && j.score>=7){ s.high_fit += excluded?-1:1; }
  render();
}

async function markApplied(url, applied){
  await fetch('/api/mark',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({url, applied})});
  const j = DATA.jobs.find(x=>x.url===url); if(j) j.applied = applied;
  const s = DATA.stats; if(s.applied!=null) s.applied += applied?1:-1;
  render();
}

// ── Apply Focus panel ─────────────────────────────────────────────────
let FOCUS_URL = null;
function esc2(s){ return (s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

async function openFocus(url){
  FOCUS_URL = url;
  const d = await (await fetch('/api/applypanel?url='+encodeURIComponent(url))).json();
  if(d.error){ alert(d.error); return; }
  const rows = d.fields.map((f,i)=>`
    <div class="fprow">
      <span class="fplabel">${esc2(f[0])}</span>
      <span class="fpval" id="fv${i}">${esc2(f[1])}</span>
      <button class="cpbtn" onclick="cp(this,'fv${i}')">Copy</button>
    </div>`).join('');
  document.getElementById('fp').innerHTML = `
    <div class="fpactions">
      <a class="big open" href="${esc2(d.apply_url)}" target="_blank" onclick="markOpened()">↗ Open posting</a>
      <a class="big open" href="/api/asset?kind=resume&url=${encodeURIComponent(url)}" target="_blank">📄 Résumé</a>
      <a class="big open" href="/api/asset?kind=cover&url=${encodeURIComponent(url)}" target="_blank">✉️ Cover PDF</a>
      <button class="big done" onclick="focusApplied()">✓ Applied</button>
      <button class="big skip" onclick="closeFocus()">Skip</button>
      <button class="big close" onclick="closeFocus()">✕</button>
    </div>
    <h2>${esc2(d.title)}</h2>
    <div><span class="co">${esc2(d.company||'(company on posting)')}</span> ${d.location?'· <span class="loc">'+esc2(d.location)+'</span>':''}</div>
    <div class="sec">Copy-paste fields</div>
    ${rows}
    <div class="sec">Cover letter <button class="cpbtn" onclick="cpText(this,'covertext')">Copy all</button></div>
    <textarea class="cover" id="covertext">${esc2(d.cover_text)}</textarea>
  `;
  document.getElementById('fpwrap').classList.add('open');
}
function closeFocus(){ document.getElementById('fpwrap').classList.remove('open'); FOCUS_URL=null; }
function markOpened(){ /* opening posting is the apply start; no-op hook for future tracking */ }
async function cp(btn, id){
  const t = document.getElementById(id).textContent;
  try{ await navigator.clipboard.writeText(t); btn.textContent='✓'; btn.classList.add('done');
       setTimeout(()=>{btn.textContent='Copy'; btn.classList.remove('done');},1200);}catch(e){ alert('Copy failed'); }
}
async function cpText(btn, id){
  const t = document.getElementById(id).value;
  try{ await navigator.clipboard.writeText(t); btn.textContent='✓ Copied'; setTimeout(()=>btn.textContent='Copy all',1200);}catch(e){ alert('Copy failed'); }
}
async function focusApplied(){
  if(FOCUS_URL){ await markApplied(FOCUS_URL, true); }
  closeFocus();
}

async function enrichOne(url, btn){
  if(btn){ btn.disabled=true; btn.textContent='Scoring…'; }
  const r = await (await fetch('/api/enrich',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({url})})).json();
  if(r.ok){
    const j = DATA.jobs.find(x=>x.url===url);
    if(j){ j.score = r.score; j.scored = true; }
    render();
  } else {
    if(btn){ btn.disabled=false; btn.textContent='Enrich'; }
    alert('Could not score: ' + (r.error||'unknown'));
  }
}

const PAGE_SIZE = 60;
let curPage = 1, lastSig = '';

// Scope predicate for each stat-tile view. Counts mirror the summary tiles.
const SCOPE = {
  all:      j => true,
  scored:   j => j.score != null,
  strong:   j => j.score != null && j.score >= 7 && !j.excluded,
  ready:    j => j.tailored && !j.applied && !j.excluded,
  excluded: j => j.excluded,
};

function render(){
  const q = document.getElementById('q').value.toLowerCase();
  const minScore = parseInt(document.getElementById('minscore').value);
  const site = document.getElementById('site').value;
  const sort = document.getElementById('sort').value;
  const remoteOnly = document.getElementById('remoteonly').checked;
  const inScope = SCOPE[activeView] || SCOPE.all;

  let rows = DATA.jobs.filter(j=>{
    if(!inScope(j)) return false;            // stat-tile scope
    // Applied jobs drop out of every working view (only visible under "Total").
    if(j.applied && activeView!=='all') return false;
    const sc = j.score||0;
    if(sc < minScore) return false;
    if(site && j.site!==site) return false;
    if(remoteOnly && !j.remote) return false;
    if(q){ const blob=(j.title+' '+j.site+' '+(j.company||'')+' '+j.location+' '+j.keywords).toLowerCase(); if(!blob.includes(q)) return false; }
    return true;
  });
  rows.sort((a,b)=> sort==='score' ? (b.score||0)-(a.score||0)
    : sort==='title' ? a.title.localeCompare(b.title) : (a.site||'').localeCompare(b.site||''));

  // Reset to page 1 whenever the filters/sort change.
  const sig = [activeView,q,minScore,site,sort,remoteOnly].join('|');
  if(sig!==lastSig){ curPage = 1; lastSig = sig; }

  const total = rows.length;
  const pages = Math.max(1, Math.ceil(total/PAGE_SIZE));
  if(curPage > pages) curPage = pages;
  const start = (curPage-1)*PAGE_SIZE;
  const pageRows = rows.slice(start, start+PAGE_SIZE);

  const viewLabel = {all:'All', scored:'Scored', strong:'Strong 7+', ready:'Ready to apply', excluded:'Excluded'}[activeView] || 'All';
  document.getElementById('count').textContent = total
    ? `${viewLabel} — showing ${start+1}-${start+pageRows.length} of ${total} (page ${curPage}/${pages}) · ${DATA.jobs.length} total`
    : `${viewLabel} — 0 of ${DATA.jobs.length} jobs match these filters`;
  document.getElementById('grid').innerHTML = pageRows.map(card).join('') ||
    '<p style="color:#64748b">No jobs match these filters.</p>';
  renderPager(pages);
}

function renderPager(pages){
  const el = document.getElementById('pager');
  if(pages <= 1){ el.innerHTML = ''; return; }
  el.innerHTML =
    `<button onclick="gotoPage(1)" ${curPage<=1?'disabled':''}>« First</button>`+
    `<button onclick="gotoPage(${curPage-1})" ${curPage<=1?'disabled':''}>‹ Prev</button>`+
    `<span class="pg">Page ${curPage} of ${pages}</span>`+
    `<button onclick="gotoPage(${curPage+1})" ${curPage>=pages?'disabled':''}>Next ›</button>`+
    `<button onclick="gotoPage(${pages})" ${curPage>=pages?'disabled':''}>Last »</button>`;
}

function gotoPage(p){ curPage = p; render(); window.scrollTo({top:0,behavior:'smooth'}); }

function card(j){
  const sc = j.score==null ? '–' : j.score;
  const col = (j.score||0)>=7 ? '#10b981' : ((j.score||0)>=5 ? '#f59e0b' : '#64748b');
  const badges = [
    j.tailored ? '<span class="badge resume">✓ résumé</span>' : '',
    j.cover ? '<span class="badge cover">✓ cover</span>' : '',
    j.applied ? '<span class="badge applied">✓ applied</span>' : '',
    `<span class="badge desc">${j.desc_len.toLocaleString()} char desc</span>`,
  ].filter(Boolean).join('');
  const exTag = j.excluded ? ` <span class="tag" style="background:#7f1d1d;color:#fecaca">excluded${j.excluded_reason?': '+esc(j.excluded_reason):''}</span>`:'';
  const statusColors = {worldwide:'#10b981', local:'#34d399', relocation:'#60a5fa', unknown:'#64748b', region_locked:'#ef4444'};
  const stTag = `<span class="tag" style="background:${statusColors[j.loc_status]||'#64748b'}33;color:${statusColors[j.loc_status]||'#94a3b8'}">${esc(j.loc_status)}</span>`;
  return `<div class="card ${j.excluded?'exc':''}">
    <div class="top"><span class="pill" style="background:${col}">${sc}</span>
      <a class="ti" href="${esc(j.url)}" target="_blank">${esc(j.title)}</a></div>
    <div class="meta">
      ${j.company ? `<span class="tag" style="background:#1e3a5f;color:#93c5fd;font-weight:600">${esc(j.company)}</span><span class="tag" style="font-size:.62rem">via ${esc(j.site)}</span>` : `<span class="tag">${esc(j.site)}</span>`}
      ${stTag}
      ${j.remote?'<span class="tag rem">remote</span>':''}
      ${j.location?`<span class="tag loc">${esc(j.location.slice(0,40))}</span>`:''}
      ${j.salary?`<span class="tag">${esc(j.salary)}</span>`:''}${exTag}
    </div>
    ${j.keywords?`<div class="kw">${esc(j.keywords)}</div>`:''}
    ${j.reasoning?`<div class="rs">${esc(j.reasoning)}</div>`:''}
    <div class="badges">${badges}</div>
    <div class="foot">
      ${j.tailored ? `<button class="exbtn focus" onclick="openFocus('${esc(j.url)}')">⚡ Apply</button>` : ''}
      <a class="link apply" href="${esc(j.apply_url)}" target="_blank">Open posting</a>
      ${j.tailored ? `<a class="link asset" href="/api/asset?kind=resume&url=${encodeURIComponent(j.url)}" target="_blank">📄 Résumé</a>` : ''}
      ${j.cover ? `<a class="link asset" href="/api/asset?kind=cover&url=${encodeURIComponent(j.url)}" target="_blank">✉️ Cover letter</a>` : ''}
      ${j.applied
        ? `<button class="exbtn applied-undo" onclick="markApplied('${esc(j.url)}',false)">✓ Applied — undo</button>`
        : `<button class="exbtn markapplied" onclick="markApplied('${esc(j.url)}',true)">✓ Mark applied</button>`}
      ${!j.scored ? `<button class="exbtn enrich" onclick="enrichOne('${esc(j.url)}',this)">Enrich</button>` : ''}
      ${j.excluded
        ? `<button class="exbtn restore" onclick="toggleExclude('${esc(j.url)}',false)">Restore</button>`
        : `<button class="exbtn" onclick="toggleExclude('${esc(j.url)}',true)">Exclude</button>`}
    </div>
  </div>`;
}

load();
</script>
</body>
</html>"""
