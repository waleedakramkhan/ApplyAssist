# ApplyAssist — Quick Start

This machine is already set up. The Python venv, dependencies, Claude Code CLI, Chrome, and
Node 22 are all installed. Use the `./applyassist.sh` launcher — it activates the right Node
version and the venv for you.

## One thing left to do (yours)

Run the setup wizard. It asks for your resume file, some profile details, and a **free**
Gemini API key (get one at https://aistudio.google.com → "Get API key"):

```bash
cd ~/Documents/ApplyAssist
./applyassist.sh init
./applyassist.sh doctor      # should now show all OK / Tier 3
```

## Daily use

```bash
cd ~/Documents/ApplyAssist
./applyassist.sh run          # find jobs → score → tailor resume → cover letters → PDFs
./applyassist.sh status       # see what's ready to apply to
./applyassist.sh apply        # opens a browser, fills ONE application, then STOPS
```

### Better than `run`: import your LinkedIn alerts

Auto-discovery (`run`) pulls a lot of irrelevant jobs. Your **LinkedIn job alerts** are curated and
far better. Use them instead:

1. Add a Gmail **app password** ([myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords))
   to `~/.applyassist/.env` as `GMAIL_ADDRESS` and `GMAIL_APP_PASSWORD`.
2. Run:
   ```bash
   ./applyassist.sh import-alerts --from-gmail
   ```
   This reads your LinkedIn alerts straight from Gmail over IMAP — no exporting emails one by one.
   It pulls each posting (title, company, location, description), drops region-locked ones, and
   scores + tailors the eligible jobs. Then `./applyassist.sh dashboard` to review and
   `./applyassist.sh apply` to apply. (Full details in README → "Import jobs from your LinkedIn alerts".)

   No Gmail? Drop exported `.eml`/`.mbox` files into `~/.applyassist/inbox/` and run
   `./applyassist.sh import-alerts` instead — see README for both options.

When `apply` stops, the browser stays open with the form filled in. **You** review every
field, fix anything, and click **Submit** yourself. Then back in the terminal:

- `a` = I submitted it
- `s` = skip this one
- `q` = quit

## Notes

- **You always click Submit.** ApplyAssist fills the form; it never submits for you (that's the
  whole point — it keeps you out of ban trouble and stops bad applications going out at scale).
- **It never solves CAPTCHAs.** If one appears, it pauses and you finish that step in the browser.
- **One job per run by default.** Do several in a row with `./applyassist.sh apply --limit 5`
  (still one-at-a-time, you submit each).
- **Daily cap is 30.** Raise it if you really need to: `APPLYASSIST_DAILY_CAP=50 ./applyassist.sh apply`.
- **Quality over volume.** 30 reviewed, tailored applications beat 1,000 sprayed ones. Spend the
  time you save chasing referrals — they convert far better than cold applications.

## Running without the launcher

If you prefer, activate things manually:

```bash
source ~/Documents/ApplyAssist/.venv/bin/activate
nvm use 22
applyassist <command>
```
