Keith and Scott aren't running a secret search algorithm. Our 4.04% most likely comes from changes we copied from Scott's notebook, plus about 6.5 unused hours of the 9-hour run. The full write-up and Exp 4 spec are in `C:\Users\User 1\.claude\plans\you-are-claude-opus-cosmic-perlis.md`. I wrote no code; the only file created is that plan.

**1. Why we're at 4.04% and Keith is at 8.62%**
- **Same solver.** Keith's notebook says the Duck prompts, tool-use loop and game policy are unchanged, and he only changed model serving. Two other public forks differ from his by one line (`analyzer_timeout`). None of them use MCTS, beam search or an action verifier.
- **We copied Scott, not Keith.** Exp 3 is Scott Le Grand's notebook with `analyzer_timeout` raised to 1200. It differs from Keith's in three ways:
  - **Half the context:** 16K instead of 32K (`LOCAL_ANALYZER_CONTEXT_WINDOW=16384`), to fit 16 model slots instead of 8. This is my top suspect: Duck's prompts are long, so a lot of history gets thrown away.
  - **AgentFix patches are on.** Their own notes record regressions (e.g. "29% fewer actions per 2h"), and nothing shows they were ever tested on the leaderboard.
  - **Longer timeout:** 1200s instead of 900s.
- **Luck matters too.** Scott's per-game scores range from 0 to 47.6, and three games make up about half his mean, so one submission can swing by roughly ±2 points.
- **Hardware correction:** the RTX Pro 6000 machine is Blackwell with 96 GB, not Ada with 48 GB. The NVFP4 weights need Blackwell and take 81.8 GiB, leaving only about 5 GiB for the context cache. That's why context length and slot count trade off directly.

**2. The biggest lever: most of the 9 hours goes unused**
- Each game is capped at 7,920s (2.2h) and up to 28 run at once. If the hidden set has 28 games or fewer, the notebook finishes about 2.5h into its 9h. Exp 2's log confirms it ended after 2h12m.
- Games are limited by time, not ability. All 22 of Scott's public runs hit the time cap after only 33–154 actions, most still on level 1–2.
- Fix: set each game's time budget once the hidden game list is known, so the runs fill the 9 hours. It falls back to Keith's 2.2h if the hidden set is large, and the existing soft deadline still protects the 9h limit. This also covers your "fast-exit easy puzzles" idea: games that finish early free up model capacity for the rest.

**3. Your three avenues**
- **Code-executing transforms:** Duck already does this. The model writes Python against the segmented board and is told to search candidate action sequences. The next step is better helper functions in that sandbox, not a new approach.
- **Best-of-N, rollback, MCTS in the real game:** a poor fit. The game can't be saved and restored, and every action (including RESET) lowers the efficiency score. Search has to happen inside the model's own Python simulation of the game.
- **Checking actions before committing:** belongs in that same simulation. It's prompt and helper work for after Exp 4.

**4. Exp 4 spec: "Keith-exact + Budget-Fill"**
Start from Keith's notebook, not ours:
- Keith's serving profile unchanged: 8 slots, 32K context.
- No AgentFix patches.
- `analyzer_timeout` back to 900.
- One new change: the dynamic per-game time budget.
- Keep Keith's offline run on the 25 public games. Exp 3 skipped it, so we have no per-game data for our current best submission.

I expect reverting to Keith's setup to get back to about 8.6 ± 2. With the budget fix, a realistic target is 8–11% if the hidden set has 28 games or fewer. After that, test AgentFix memory, the action-effect ledger and sandbox helper functions one at a time on the public games.

**Questions for you:**
1. What leaderboard score did Scott's `taaf-flashnext-sheetu12b-0922` get? About 4% would confirm the 16K context and AgentFix as the cause.
2. Do we know how many games are in the hidden set? That decides how much the budget fix gains.
3. How many submissions per day and how much weekly GPU quota are left? The offline check on the 25 public games uses about 9 GPU-hours.