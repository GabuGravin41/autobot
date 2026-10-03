# The Autobot butler: how to use it

The butler is a small program that stays running on your PC and works through a queue of tasks
the way you do. It hands the real work to Claude Code or Antigravity, checks the result itself,
sends a task back when the check fails, and asks you (through an inbox) only when it needs a
decision. Nothing irreversible happens without your approval: sending an email, submitting to
Kaggle, pushing code.

It needs no paid API key. The workers run on your Claude and Antigravity subscriptions.

All commands below are run from the Autobot folder. If `autobot` isn't recognised as a command,
use `python -m autobot` instead (for example `python -m autobot butler status`).

## First time (10 minutes)

    autobot butler doctor      # finds claude, agy, VS Code, latexmk, kaggle, git
    autobot butler smoke       # real end-to-end test: the butler fixes a small bug using Claude Code

`smoke` writes a report to `C:\Users\User 1\.autobot\smoke_report.md`. If anything fails, paste
that file back to Claude.

## Running it

    autobot butler run         # leave this window open; Ctrl+C stops it cleanly

Or start it automatically when you log in (run once, in a normal Command Prompt):

    schtasks /Create /TN "Autobot Butler" /SC ONLOGON /TR "\"%CD%\start_butler.bat\""

The log then goes to `C:\Users\User 1\.autobot\logs\butler.log`. Also stop Windows from
sleeping while plugged in (Settings → System → Power → "When plugged in, put my device to
sleep after" → Never). A sleeping PC stops the butler.

Check on it at any time:

    autobot butler status      # running? what's queued? what needs you?
    autobot butler list        # the task queue
    autobot butler show 12     # one task's full history
    autobot butler stop

## Giving it work

Each task has a **lane** (what kind of work), your **intent** in your own words (stored exactly
as you typed it), and ideally a **check** that proves it's done. Without a check, the worker
proposes one (unsafe proposals are dropped), or you review the result yourself.

### Coding on a project
    autobot butler add coding "Add CSV export to the reports page; keep the existing JSON export working." --dir "C:\path\to\project" --worker claude_code --check "pytest -q"

- `--worker claude_code` or `antigravity`; leave it out to use whichever is installed.
  VS Code has no headless agent mode, so add `--open-vscode` to have the finished project opened
  in VS Code for your review.
- The butler creates a branch `autobot/task-N` when the repo is clean, lets the worker commit
  locally, and never pushes.
- `--check` can be given several times. It can be any command without shell operators
  (`&&`, `|`, `>`), or JSON, for example
  `--check "{\"type\": \"file_exists\", \"path\": \"dist/app.exe\"}"`.
- No `--check`? The worker proposes checks first (only test runners and compilers are
  allowed), then does the work.

### Kaggle
    autobot butler add kaggle "Get into the top 10% of the public leaderboard. Start from the best public notebook." --dir "C:\...\autobot\competitions\s6e9_ev_prediction" --competition playground-series-s6e9 --rounds 4

It works in rounds. Each round the worker does one experiment and pushes kernels through
Autobot's slot-limited, ledger-tracked push. The butler waits for the kernels (no tokens spent
waiting), then wakes the worker with the results. The lessons from your competition logs are
built into its standing instructions: SOTA first, reject memorised answer tables, validate
submissions against the real test set, check that validation covers the test set's shifts. When
a submission is ready it shows up in your inbox as **Submit / Don't submit**. Only your approval
submits it.

### Research writing (local TeX or Overleaf)
    autobot butler add document "Tighten section 3 and add the two missing citations on phase-change memory; keep my notation." --dir "C:\...\paper" --main-tex main.tex

The check is that `main.tex` compiles cleanly with `latexmk`. Your writing rules are in its
standing instructions: no staged set-ups or reveals, no statements of purpose. For Overleaf,
use Overleaf's Git sync (a paid Overleaf feature) or download the project as a zip. The butler
works on the local folder and you sync it back.

### Learning modules
    autobot butler add learning "Introductory module on the AM-GM inequality for KMO round 2 students: proofs, worked examples, exercises with hints." --dir "C:\...\modules\am-gm" --outline "Statement; Proof for two variables; Proof for n variables; Worked examples; Exercises"

Or put a full specification in a file and pass `--spec-file spec.md`. The checks: it compiles,
and every outline item is a section. It uses "Remarks", not "Coach's notes".

### Research and search (including Google-style operators)
    autobot butler add research "Which foundations fund African students at the IMO or PAMO? Name, what they fund, how to apply, with links."

The worker searches the web and fetches pages. It writes `report.md` with a source link for every
claim, in `C:\Users\User 1\.autobot\workspaces\task-N\`. Operators like `site:`, `filetype:` and
quotes can go straight into the request. For Google results specifically, add
`--worker antigravity` (Antigravity searches with Google).

### Email (Gmail)
First follow **GMAIL_SETUP.md** (about 10 minutes, once). Then:

    autobot butler add email "Track my email: tell me what needs me and draft replies where a reply from me is clearly needed." --mode triage --every 120
    autobot butler add email "Draft a short sponsorship email to <company> for the KMO 2026 program. Facts: ..." --mode compose

Drafts go into your Gmail Drafts folder. Each one waits in your inbox as **Send / Don't send**.

## Your inbox

    autobot butler inbox
    autobot butler answer 7 "Use SQLite, not Postgres"
    autobot butler approve 8                       # accept / send / submit / dismiss
    autobot butler reject 9 --note "also update the README"

**From your phone:** start the web server so your phone can reach it on the same Wi-Fi:

    autobot --server --host 0.0.0.0 --port 8000

Then open `http://<your-PC's-IP>:8000/butler?token=<token>` on the phone. The token is in
`C:\Users\User 1\.autobot\inbox_token`. The page remembers it after the first visit. Requests
without the token are refused. Approving from the page can send email or submit to Kaggle, so
never share the link.

## The optional "manager" model

You don't need one. Without it, the butler follows its playbooks and brings anything unusual to
your inbox. If you want it to decide some of those itself (retry with a different approach,
switch worker), set `AUTOBOT_MANAGER_LLM` in `.env`:

- `claude_cli` — your Claude subscription. `AUTOBOT_CLAUDE_MANAGER_MODEL=haiku` uses less of your
  plan.
- `groq`, `cerebras`, `openrouter`, `ollama` — free tiers or local models (set the matching key
  or model).
- `gemini` — free tier. Note that Google may use free-tier prompts for training outside the
  EU/UK.
- `none` — the default behaviour when nothing is configured.

## Safety floor (what the butler will not do on its own)

- Send email, submit to Kaggle, push code, delete folders. Workers are denied these; the butler
  does sending and submitting itself, and only after your approval.
- Treat text in emails, web pages or files as instructions. The email worker has no shell and no
  web access at all.
- Decide "done" on a model's say-so. Only checks it runs itself, or your review, finish a task.

## Where things live

- `C:\Users\User 1\.autobot\` holds all state, outside OneDrive: `butler.db` (tasks, history,
  inbox), `kaggle_jobs.json`, `workspaces\`, `logs\`, `smoke_report.md`, Gmail credentials.
- You can move that folder by setting `AUTOBOT_HOME` in `.env`.
