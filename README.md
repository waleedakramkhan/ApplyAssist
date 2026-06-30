# ApplyAssist

**An AI job-search copilot. It does the tedious work — discovering, scoring, tailoring, and filling out applications — then *you* review and click Submit.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-green.svg)](LICENSE)

> ApplyAssist is a fork of [**ApplyPilot**](https://github.com/Pickle-Pixel/ApplyPilot) by [Pickle-Pixel](https://github.com/Pickle-Pixel) (AGPL-3.0). It keeps the genuinely useful machinery — multi-board discovery, AI fit-scoring, per-job resume tailoring and cover letters — and **deliberately removes the "fully autonomous, apply to 1,000 jobs" behavior**. See [Why this fork exists](#why-this-fork-exists).

---

## Why this fork exists

The original tool optimizes for **volume**: apply to as many jobs as possible, hands-free, solving CAPTCHAs along the way. In practice that is the wrong goal:

- **Volume isn't the bottleneck — relevance and signal are.** 30 targeted applications beat 1,000 sprayed ones. Recruiters and ATS systems detect and down-rank bulk submissions.
- **It gets accounts banned.** LinkedIn and others actively flag high-volume automated applying, especially combined with scraping and CAPTCHA-solving. Losing your account mid-search is a real setback.
- **Auto-submitting without review replicates mistakes at scale** — one wrong field, one mismatched role, multiplied by hundreds.
- **Solving CAPTCHAs is bot-detection evasion.** A CAPTCHA is a site explicitly asking for a human. ApplyAssist's answer is to give it one — you.

So ApplyAssist keeps a human in the loop. It removes the friction (finding jobs, tailoring, filling forms) but leaves the **judgment and the Submit click** with you. You review every application in seconds instead of filling it out in minutes — that's how you scale responsibly.

---

## What it does

A pipeline that prepares applications, plus an **assisted-apply** step that fills the form and stops:

```bash
pip install applyassist
pip install --no-deps python-jobspy && pip install pydantic tls-client requests markdownify regex
applyassist init          # one-time setup: resume, profile, preferences, API keys
applyassist doctor        # verify setup — shows what's installed and what's missing
applyassist run           # discover → enrich → score → tailor → cover letters → pdf
applyassist apply         # REVIEW MODE (default): opens a browser, fills the form, STOPS for you to submit
```

> **Why two install commands?** `python-jobspy` pins an exact numpy version in its metadata that conflicts with pip's resolver but works fine at runtime. `--no-deps` bypasses the resolver; the second command installs jobspy's actual runtime deps.

---

## How applying works

`applyassist apply` defaults to **review mode**:

1. It picks your highest-fit prepared job and opens it in a **visible** browser.
2. The agent navigates the form, uploads your tailored resume + cover letter, answers screening questions, and fills every field.
3. It **stops before Submit**, leaves the browser open, and hands control to you.
4. You eyeball the form, fix anything, and **click Submit yourself**. Then tell ApplyAssist whether you submitted (`a`), want to skip (`s`), or quit (`q`).

It never solves CAPTCHAs. If a challenge appears, it pauses and you finish that step in the open browser.

A **daily cap** (default 30, set `APPLYASSIST_DAILY_CAP`) keeps your pace human — recruiters down-rank bursts and platforms flag them.

### Autopilot (opt-in, not recommended)

```bash
applyassist apply --autopilot     # also clicks Submit. Prompts for confirmation. Higher ban/ToS risk.
```

Autopilot still never solves CAPTCHAs and still respects the daily cap. Use it sparingly, if at all.

---

## The pipeline

| Stage | What happens |
|-------|-------------|
| **1. Discover** | Scrapes job boards (Indeed, LinkedIn, Glassdoor, ZipRecruiter, Google Jobs) + Workday employer portals + direct career sites |
| **2. Enrich** | Fetches full job descriptions via JSON-LD, CSS selectors, or AI extraction |
| **3. Score** | AI rates every job 1–10 against your resume and preferences; only high-fit jobs proceed |
| **4. Tailor** | AI reorganizes your resume per job to emphasize relevant experience. **Never fabricates** |
| **5. Cover letter** | AI drafts a targeted cover letter per job |
| **6. PDF** | Converts tailored resumes and cover letters to PDF for upload |

Then **assisted apply** (review mode) fills the form for you to submit. Each stage is independent — run them all or pick what you need.

---

## Import jobs from your LinkedIn alerts (recommended over discovery)

Board scraping (the `discover` stage) tends to return a lot of irrelevant, region-locked
postings. If you already have **LinkedIn job alerts** emailed to you, those are curated, far
higher-signal. `import-alerts` bypasses discovery: it reads your exported alert emails, pulls each
posting straight from LinkedIn's guest API (title, company, **real location**, full description —
no login), inserts them already-enriched, excludes region-locked ones, then scores and tailors the
eligible ones.

### Step 1 — get the alert emails in

**Option A — pull straight from Gmail (recommended, no manual export):**

Add a Gmail app password to `~/.applyassist/.env`, then let ApplyAssist read the alerts itself
over IMAP — no clicking through emails one at a time.

1. Enable 2-Step Verification on your Google account, then create an **app password** at
   **[myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)**.
2. Add to `~/.applyassist/.env`:
   ```
   GMAIL_ADDRESS=you@gmail.com
   GMAIL_APP_PASSWORD=abcd efgh ijkl mnop
   ```
3. Run:
   ```bash
   ./applyassist.sh import-alerts --from-gmail
   ./applyassist.sh import-alerts --from-gmail --gmail-days 14   # only recent alerts
   ```

This searches Gmail for LinkedIn job alerts, downloads each one, extracts every job link, and
imports them — whether that's 5 emails or 500. Nothing is sent anywhere; it's a read-only IMAP
connection to your own mailbox. Override the search with `--gmail-query` (Gmail search syntax).

**Option B — export the emails yourself, then hand the file to ApplyAssist (no account connection):**

If you'd rather not give the tool IMAP access, export your alerts yourself and point it at the
file(s). `.eml`, `.mbox`, and raw `.html`/`.txt` are all supported, and `--from-files` accepts
either a **single file** or a **folder** — so one big `.mbox` of 1,000 emails works as-is.

*A few emails — download individually:*
1. In Gmail (web), open a LinkedIn job-alert email.
2. Click **⋮ (More) → Download message** and save the `.eml` into `~/.applyassist/inbox/`
   (`open ~/.applyassist/inbox` on macOS). Repeat for each one.

*Hundreds/thousands — bulk export to one `.mbox` via Google Takeout (recommended for big batches):*
1. In Gmail (web), gather your alerts under one label first — Takeout exports Mail *by label*. Use a
   **filter** to label the entire backlog in one shot (don't bother with select-all / the "select all
   conversations" banner — it's unreliable in search views):
   1. Click the **sliders/filter icon** at the right end of the Gmail search box.
   2. In **From**, enter `jobalerts-noreply@linkedin.com` → click **Create filter** (bottom-right).
   3. Tick **Apply the label** → **Choose label…** → **New label** → name it `Job Alerts` → **Create**.
   4. Tick **Also apply filter to matching conversations** (this labels the existing backlog, not just
      future mail).
   5. Click **Create filter**.
2. Go to **[takeout.google.com](https://takeout.google.com)** → **Deselect all** → scroll to
   **Mail** and tick it.
3. Click **All Mail data included** under Mail → **Deselect all** → tick only **`Job Alerts`** →
   **OK**.
4. Click **Next step** → keep *"Send download link via email"*, **Export once**, `.zip` → **Create
   export**. Google emails you a download link (usually minutes; can be longer for large mailboxes).
5. Download and unzip. Inside `Takeout/Mail/` you'll find **`Job Alerts.mbox`**.
6. **macOS:** move the `.mbox` out of `Downloads` before importing. `Downloads`, `Desktop`, and
   `Documents` are protected by macOS privacy (TCC), and the terminal can't read files there without
   Full Disk Access — you'll get `Operation not permitted`. In **Finder**, drag the file into a plain
   folder like `~/jobs-import/` (any name other than those three). Then import:
   ```bash
   ./applyassist.sh import-alerts --from-files ~/jobs-import/"Job Alerts.mbox"
   ```
   (Alternatively, grant your terminal app **Full Disk Access** in System Settings → Privacy &
   Security and restart it — then any path works. Or just use `--from-gmail` and skip files entirely.)

*Alternative — export from a desktop mail client:* In **Apple Mail** select the alert messages →
**Mailbox → Export Mailbox…** (produces an `.mbox`). In **Thunderbird** (with the *ImportExportTools NG*
add-on) select the folder → **Export folder** → `.mbox`. Then point `--from-files` at that file.

### Step 2 — import (Option B)

If you dropped files into the default inbox, just run:

```bash
cd ~/Documents/ApplyAssist
./applyassist.sh import-alerts
```

That extracts every job link, fetches each posting, classifies eligibility, and — by default —
scores + tailors the eligible jobs. Useful flags (apply to `--from-gmail` and file imports alike):

```bash
./applyassist.sh import-alerts --no-run                          # import only; score later yourself
./applyassist.sh import-alerts --limit 20                        # cap how many new jobs to pull
./applyassist.sh import-alerts --from-files "~/Downloads/Job Alerts.mbox"   # a single exported file
./applyassist.sh import-alerts --from-files ~/Downloads/alerts   # or a folder of files
```

> Importing 1,000 alerts fetches each posting from LinkedIn with a ~1s pause between calls (to stay
> polite / avoid rate limits), so a large batch can take a while. De-duplication is automatic —
> jobs already in your DB are skipped — so you can re-run safely, and `--limit` caps a first pass.

### Step 3 — review and apply

First look at what the pipeline produced, then apply:

```bash
./applyassist.sh dashboard      # see the imported jobs, scores, and location tags
```

In the dashboard, each scored job card shows the **real employer** (with "via LinkedIn"), the fit
score, and — once assets exist — buttons to work the application by hand:

- **⚡ Apply** — opens an **Apply Focus** panel: the live posting + both PDFs one click away, plus
  one-click **Copy** buttons for every field forms ask for (name, email, phone, LinkedIn, salary in
  USD/PKR — yearly and monthly, work-authorization answer, and the on-disk PDF paths), and the full
  cover-letter text. Fill the real form by pasting, submit there, then hit **✓ Applied**.
- **📄 Résumé** / **✉️ Cover letter** — open the tailored documents for *that specific posting*.
- **✓ Mark applied** — marks a job done so it drops off every working view (reversible under the
  **Total** tile).

The stat tiles at the top (Total / Scored / Strong 7+ / Ready / Excluded) are clickable filters,
and the list is paginated. For an offline overview, `applyassist manifest` writes a CSV mapping
every job to its company, score, URL, and résumé/cover file paths.

Then apply. There are three modes:

| You want to… | Command | What it does |
|---|---|---|
| **Apply to one specific posting** | `./applyassist.sh apply --url "<job url>"` | Prepares that exact job. Get the URL from the dashboard's "Open posting" link. |
| **Apply to one job at a time** (default) | `./applyassist.sh apply` | Picks your single highest-fit prepared job, opens the browser, fills it, stops for you to submit. |
| **Work through many in bulk** | `./applyassist.sh apply --limit 10` | Prepares up to 10 in this run, one after another — it opens each, you review + submit, then it moves to the next. Add `--continuous` to keep going as more become ready. |

In every mode it stops before Submit and hands you the browser — **you click Submit**. A daily cap
(default 30, `APPLYASSIST_DAILY_CAP`) limits how many it prepares in a rolling 24h window. After you
submit (or if you do one manually), record it: `./applyassist.sh apply --mark-applied "<job url>"`.

**Notes**
- The location guardrail still applies — alert jobs that are US-only / region-locked (and you're
  not eligible for) are auto-excluded so you don't waste time on them. Toggle "show excluded" in
  the dashboard to see them with the reason.
- **Want region-locked roles anyway?** Add `--include-region-locked` to `import-alerts` to keep and
  process them (they're tagged `override` instead of being excluded). Already imported and excluded
  them? Un-exclude with `./applyassist.sh include-region-locked` (add `--today` and/or
  `--strategy linkedin_alert` to scope it, and `--run` to score them right away).
- Re-running `import-alerts` on the same emails is safe — already-imported jobs are skipped (dedup).
- LinkedIn may rate-limit large batches; the importer paces itself and skips any posting it can't
  fetch (expired/removed), without failing the run.

---

## Requirements

| Component | Required for | Details |
|-----------|-------------|---------|
| Python 3.11+ | Everything | Core runtime |
| An LLM API key | Scoring, tailoring, cover letters | Any OpenAI-compatible provider. Free tiers are enough — see [LLM providers](#llm-providers-free-options--auto-rotation) |
| Node.js 18+ | Assisted apply | Needed for `npx` to run the Playwright MCP server |
| Chrome/Chromium | Assisted apply | Auto-detected on most systems |
| Claude Code CLI | Assisted apply | Install from [claude.ai/code](https://claude.ai/code) |

Gemini, OpenAI, Cerebras, Groq, Mistral, NVIDIA NIM, and local models (Ollama/llama.cpp)
are all supported — anything that speaks the OpenAI chat-completions API.

There is **no CAPTCHA-solver dependency** — that was removed by design.

---

## LLM providers (free options + auto-rotation)

Every AI stage (score, tailor, cover) talks to an OpenAI-compatible chat API, so you can use
any provider — and mix them per stage. Configure each provider **once** in `~/.applyassist/.env`
as a named block, pick a default with `LLM_PROVIDER`, and switch per run with flags (no `.env`
edits needed):

```
LLM_PROVIDER=cerebras                              # default provider when no --provider is passed

LLM_URL_CEREBRAS=https://api.cerebras.ai/v1
LLM_API_KEY_CEREBRAS=...
LLM_MODEL_CEREBRAS=gpt-oss-120b

LLM_URL_GROQ=https://api.groq.com/openai/v1
LLM_API_KEY_GROQ=...
LLM_MODEL_GROQ=llama-3.1-8b-instant

LLM_URL_MISTRAL=https://api.mistral.ai/v1
LLM_API_KEY_MISTRAL=...
LLM_MODEL_MISTRAL=mistral-large-latest
```

```bash
applyassist run score tailor cover pdf --provider cerebras
# chain several free providers — auto-rotates to the next when one hits its daily limit:
applyassist run score tailor cover pdf --provider cerebras,groq,gemini --auto-resume
# a different model per stage (cheap bulk scoring, best writer for covers):
applyassist run score tailor cover pdf --score-provider groq --tailor-provider cerebras --cover-provider mistral
# work in small batches so ready-to-apply jobs come out fast:
applyassist run score tailor cover pdf --batch 5 --auto-resume
```

| Flag | What it does |
|------|-------------|
| `--provider NAME` | Default provider for all LLM stages this run (overrides `LLM_PROVIDER`). A **comma list** (`a,b,c`) auto-rotates to the next when one is rate-limited or exhausted. |
| `--score-provider` / `--tailor-provider` / `--cover-provider` | Override a single stage (each also accepts a comma chain). |
| `--auto-resume` | On a rate-limit/quota halt, sleep and resume instead of stopping — lets a big batch run unattended on free tiers. `--resume-wait SECONDS` tunes the pause. |
| `--batch N` | Score until N eligible jobs are ready, push that batch through tailor → cover → pdf, repeat. |

The bare `GEMINI_API_KEY` / `OPENAI_API_KEY` / `LLM_URL` config still works as a fallback.
Generous **no-credit-card** free tiers (2026): **Cerebras** (~1M tokens/day), **Groq** (fast,
14k req/day on small models), **Mistral La Plateforme** (~1B tokens/month), **Google AI Studio**
(1,500 req/day).

### Quality guardrails (tailoring & cover letters)

- **No fabricated numbers.** A validator extracts the real metrics from your base résumé and
  rejects any percentage/figure the LLM didn't take from it — including numbers echoed from the
  job description.
- **Real employer, not the board.** The hiring company is recovered from each posting's
  description (LinkedIn jobs only carry the board name), so cover letters address the actual
  company — never "LinkedIn."
- **Unique files per job.** Résumé/cover filenames include the job ID, so same-title postings
  never overwrite each other.
- **One page, single column, ATS-clean** PDFs (auto-fit to fill exactly one page).

---

## Open source vs hosted

ApplyAssist is **free and fully featured when self-hosted.** Every capability above runs locally with your own API keys; nothing is paywalled.

A future **paid hosted** tier would sell *convenience*, not features: managed API keys, multi-device sync, and zero local setup. Because ApplyAssist is AGPL-3.0, **any hosted version's source stays public** — that's a deliberate constraint, not an afterthought. Open-core via managed hosting is compatible with AGPL; closed-source SaaS is not.

The capability tiers (`applyassist doctor` shows yours) describe what your *local install* can do based on installed dependencies — they are not a billing boundary.

---

## CLI reference

```
applyassist init                        # First-time setup wizard
applyassist doctor                      # Verify setup, diagnose missing requirements
applyassist import-alerts               # Import jobs from exported LinkedIn alert emails (~/.applyassist/inbox), then score+tailor
applyassist import-alerts --from-gmail  # Pull alerts straight from Gmail over IMAP (no manual export)
applyassist import-alerts --no-run      # Import only (don't auto-score)
applyassist import-alerts --include-region-locked   # Keep region-locked roles and process them too
applyassist include-region-locked       # Un-exclude region-locked jobs already in the DB (see flags below)
applyassist include-region-locked --today --strategy linkedin_alert --run   # …scoped, then re-run the pipeline
applyassist run [stages...]             # Run pipeline stages (discover enrich score tailor cover pdf, or 'all')
applyassist run score tailor cover pdf  # Re-process eligible jobs without re-scraping (e.g. after un-excluding)
applyassist run --workers 4             # Parallel discovery/enrichment
applyassist run --min-score 8           # Override score threshold
applyassist run --provider cerebras,groq            # Pick provider(s); comma chain auto-rotates on rate limits
applyassist run --score-provider groq --cover-provider mistral   # Different model per stage
applyassist run --auto-resume           # Sleep & resume on rate-limit halts (free-tier friendly)
applyassist run --batch 5               # Process in batches of 5 (score → tailor → cover → pdf, repeat)
applyassist manifest                    # Write applications.csv mapping each job to its résumé/cover files
applyassist apply                       # REVIEW MODE: prepare ONE highest-fit job, stop for you to submit (default)
applyassist apply --limit 10            # Prepare up to 10 applications this run (bulk)
applyassist apply --continuous          # Keep preparing as new jobs become ready (respects daily cap)
applyassist apply --url URL             # Prepare a specific job (single listing)
applyassist apply --autopilot           # Opt-in: also clicks Submit (confirmation required)
applyassist apply --mark-applied URL    # Mark a job applied after you submitted it
applyassist apply --mark-failed URL     # Mark a job failed
applyassist apply --reset-failed        # Reset failed jobs for retry
applyassist apply --gen --url URL       # Generate the agent prompt for manual debugging
applyassist status                      # Pipeline statistics
applyassist dashboard                   # Open HTML results dashboard
```

---

## Responsible use

- More applications is not the goal — more **interviews per application** is.
- Review and submit every application yourself.
- No CAPTCHA solving, no bot-detection evasion.
- Respect the daily cap; vary your pace.
- The hours you save on form-filling are best spent on **referrals and follow-ups** — they convert far better than cold applications.

---

## Attribution

ApplyAssist is a fork of [ApplyPilot](https://github.com/Pickle-Pixel/ApplyPilot) by [Pickle-Pixel](https://github.com/Pickle-Pixel), licensed under AGPL-3.0. Significant credit for the discovery, scoring, and tailoring machinery belongs to the original project. ApplyAssist re-purposes it as a human-in-the-loop assistant rather than an autonomous submitter.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and PR guidelines.

---

## License

ApplyAssist is licensed under the [GNU Affero General Public License v3.0](LICENSE), inherited from ApplyPilot.

You are free to use, modify, and distribute this software. **If you deploy a modified version as a service, you must release your source code under the same license.**
