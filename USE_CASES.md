# Autobot: Example Use Cases (Target Flows)

These three use cases drive what the system must support: **human-in-the-loop auth**, **multi-task execution** (switch when one task is waiting), **navigation and feedback** (so the AI knows where it is and what to do next), and **platform-specific adapters** (Google AI Studio, LeetCode, Kaggle).

---

## Use case 1: Personal portfolio via Google AI Studio → GitHub → Vercel

**User goal:** “Create a portfolio page for me using Google AI Studio.”

**Flow:**

1. **Sign in to Google AI Studio**
   - Use the user’s **student email** (system finds it on device, or user provides it).
   - **Password:** User can give it upfront, or the app can **prompt for password** and **share the screen**; user enters the password remotely on the sign-in page; the system captures the correct password and loads it into the session (e.g. for this run only).
2. **Create the portfolio**
   - Open Google AI Studio (aistudio.google.com), ensure signed in.
   - Prompt Gemini to generate a personal portfolio page (user may provide name/bio/links).
   - AI returns code; system saves it (e.g. `index.html` or a small project).
3. **Show output**
   - Show the result on screen (e.g. open file in browser or preview).
   - **Alert the user:** e.g. “Look at the screen” so they can review.
4. **Push to GitHub**
   - `git init` (if needed), add repo, commit, push to the user’s GitHub (credentials from env or from a prior “store token” step).
5. **Deploy to Vercel**
   - Run `vercel` CLI (or API) to deploy the static site.
   - User ends up with a live personal portfolio URL.

**System needs:**

- **Request human input:** An action (e.g. `request_human_input` or `prompt_for_password`) that pauses the run, shows a prompt in the UI (“Enter password” / “Enter GitHub token”), user submits, value is written to engine state (e.g. `human_input_password`) and the run continues. Optional: “share screen” so user types password on the real sign-in page and the system can use it (e.g. via clipboard or a one-time capture).
- **Google AI Studio adapter (or workflow):** Open aistudio.google.com, handle sign-in (email + password from state), send a prompt (“Generate a personal portfolio page…”), read the model’s code response (e.g. from DOM or clipboard), save to a file.
- **Git + GitHub:** `run_command` for `git add`, `git commit`, `git push`; or a small `git_adapter` that wraps these and uses `GITHUB_TOKEN` / stored credentials.
- **Vercel:** `run_command` for `vercel --prod` or similar; or a `vercel` adapter.
- **Alert user:** An action like `log` + “Look at the screen” or a dedicated `notify_user` that can show a dialog or send to the UI.

---

## Use case 2: Solve 100 LeetCode problems (20 easy, 40 medium, 40 hard)

**User goal:** “Solve 100 LeetCode problems: 20 easy, 40 medium, 40 hard. Use the AIs; I don’t care how long it takes.”

**Flow:**

1. Go to LeetCode (e.g. leetcode.com/problems).
2. **Pick a problem** (by difficulty: easy → medium → hard, and within that by order or list).
3. **Get problem statement** (copy from the page).
4. **Cycle through AIs:** Prefer **DeepSeek** and **Grok** (fast + deep); occasionally **Claude**. Send problem to the chosen AI, get code back. If code fails (wrong answer / error), try next AI or re-prompt same AI.
5. **Submit on LeetCode:** Paste code in the editor, submit.
6. **Read result:** Pass/fail, acceptance, errors. If not acceptable, loop: send failure + code to another AI, get new code, submit again.
7. Repeat until the problem is “solved” (e.g. accepted), then **move to the next problem** until all 100 are done.

**System needs:**

- **LeetCode adapter (or human_nav):** Actions such as: open problem list, open problem by difficulty/slug, copy problem statement, paste code in editor, submit, read submission result (status, pass/fail, error message). Prefer human_nav + clipboard so the AI “controls the computer” without depending on fragile selectors.
- **AI routing:** Planner or workflow that can call DeepSeek (OpenRouter), Grok (x.ai), Claude (OpenRouter) with the same prompt; retry with a different model if the first returns bad code.
- **State:** Store current problem slug, difficulty, code attempted, submission result so the next step (or next loop) knows “where we are” and whether to retry or advance.
- **Long-running:** Run unattended for hours; no hard “max steps” that kills the run before 100 problems (or make max steps very high / configurable).

---

## Use case 3: Kaggle competition – 10 scripts/day, 5+ submissions/day, multi-task while waiting

**User goal:** “Participate in this Kaggle competition. Create at least 10 scripts per day (each improving on the same notebook). At least 5 submissions per day (or competition limit). When a submission is waiting (e.g. 20–30 min), switch to other tasks.”

**Flow:**

1. **Open Kaggle** competition page; read problem statement and data description.
2. **Create/improve notebook:** Start from a base notebook or previous version; send problem + current code to ChatGPT/DeepSeek/Grok; get improved code; write next version (e.g. `v2.py`, `v3.py` … ≥10 versions per day).
3. **Run notebook** (Kaggle “Run All” or equivalent).
4. **Submit** (e.g. submit to leaderboard).
5. **Wait:** Kaggle often takes 20–30 minutes to score. **While waiting,** the system should **switch to another task** (e.g. LeetCode, portfolio, or another Kaggle run). So we need a **task queue** or **concurrent goals**: when task A hits “wait 30 min”, pause A and run task B; when A’s wait is over (or B is waiting), resume A.
6. **Check leaderboard:** After wait, read score/rank; use that to prompt the AI for the next improvement.
7. **Iterate:** Improve code based on leaderboard, create next version, run, submit, wait, switch task, repeat. Optionally: “copy other people’s code and repurpose” (e.g. read public notebooks, adapt, submit).

**System needs:**

- **Kaggle adapter (or human_nav):** Open competition, open/create notebook, paste code, run all, submit to competition, read submission status and score/leaderboard. Handle “submission pending” and long waits (e.g. 20–30 min).
- **Multi-task execution:** When the engine executes a step like `wait` with `seconds >= 300` (5 min) or `wait_for_submission` (poll until done), **don’t block the whole process.** Instead: pause this “goal” or “plan,” put it in a “waiting” queue with a resume time, and run the **next** goal in the queue. When the wait expires or the submission completes, resume the first goal. So we need a **scheduler** or **multi-goal runner** that can run several plans and switch context when one is waiting.
- **State per goal:** Each goal has its own state (or a slice of global state) so that when we resume “Kaggle task,” we know which competition, which notebook version, last score, etc.
- **Feedback:** Leaderboard position, last score, submission status so the AI can decide “improve code” vs “try another approach” vs “copy and adapt public notebook.”

---

## Cross-cutting: How the AI “controls the computer” and gets feedback

- **Hands:** Tools = open URL, click, type, paste, run command, git push, vercel deploy, submit on LeetCode/Kaggle. The AI must have **enough tools** for each platform (browser, terminal, each site).
- **Eyes:** Feedback = **state** (current URL, page title, last action result, clipboard, last command output, submission status, leaderboard snippet, screenshot path). The engine should put these into `state` and the planner should pass them to the LLM so the AI knows “where it is” and “what just happened.”
- **Fallbacks:** If a click fails (selector missing), try another selector or human_nav; if a site structure changed, retry or use a different path. Document fallback rules in adapters and in the planner so the AI can “if this happens, do this.”
- **Navigation:** Robust flows for Overleaf, LeetCode, Kaggle, Google AI Studio so the AI doesn’t get “lost” (e.g. wrong page, wrong tab). Use `browser_get_url`, `browser_get_title`, or screenshot path in state so the AI can correct course.

---

## Implementation order (for builders)

1. **Request human input** – engine action + UI so password/token can be entered and stored in state.
2. **Google AI Studio flow** – workflow or adapter: sign-in (email + password from state), prompt Gemini, get code, save file, alert, git push, Vercel deploy.
3. **LeetCode** – adapter or human_nav + workflow: open problem, copy statement, call AI, paste code, submit, read result, loop.
4. **Kaggle** – adapter or human_nav + workflow: open competition, run notebook, submit, wait; expose “wait for submission” so the multi-task runner can pause this goal.
5. **Multi-task runner** – queue of goals; when one does a long wait, pause and run another; resume when wait completes.
6. **State and feedback** – ensure URL, title, last result, screenshot path, submission status are in state and in the LLM prompt.
7. **Fallbacks** – in each adapter, define retry and alternative flows (e.g. human_nav if selector fails).

These use cases define “the crazy stuff” the system is being built to achieve.
