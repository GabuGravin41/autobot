The brief describes the wrong metric, and that changes the strategy. The leaderboard number isn't a correlation or R² score on incident shockwaves. According to `official_repo/docs/SCORING_SPEC.md`, it's a weighted mix of four tasks:

```
S_total = 0.35·S_state + 0.30·S_queue + 0.15·S_physics + 0.20·S_ODME
```

- **Task 1 (state):** masked-cell error on speed and per-lane flow.
- **Task 2 (queue):** overlap (IoU) between predicted and true queue cells within each forecast window.
- **Task 3 (physics):** computed from our Task 1 answers. There is nothing separate to submit.
- **Task 4 (ODME):** estimating path flows between origins and destinations.

I only read the code; nothing was run or submitted. The plan is in `C:\Users\User 1\.claude\plans\you-are-claude-opus-wondrous-yeti.md`.

## 1. Where the gap actually is
- **Task 1 has no learned model.** `output_exp11/src/task1_state.py` only interpolates in time and space, blends the two 85/15, falls back to historical averages, and applies soft physics corrections. It scores about 0.90 locally.
- **LightGBM exists only in Task 2, and it barely acts.** It was trained on about 80 windows (0.44 IoU locally). It needs a probability of 0.80 before it predicts a queue, so it rarely changes anything.
  - Onset windows predict a hard-coded cluster of bottlenecks, chosen by hour of day, at T+30 only.
  - Ongoing windows just carry forward the noisy observed speed at the start of the forecast.
- **What each task can add:** we need +0.075.
  - Task 1 from 0.90 to 0.95 is worth about +0.018.
  - ODME is worth about +0.01.
  - Task 2 can plausibly rise 0.15–0.25, worth +0.045 to +0.075. It's the only lever big enough to close most of the gap alone.
- **The leaders' edge is probably not graph networks.** Each corridor is a single chain of links, so a graph convolution adds nothing a 1-D convolution wouldn't. The leaders most likely have a well-calibrated queue forecaster and a learned imputer.

## 2. What to build
**A. Queue forecasting (Task 2) – highest value.**
- **Stop using a fixed threshold.** Each window's score is a ratio of overlapping cells, so one global threshold (0.80) is poorly calibrated by design. Instead, within each window, pick the set of cells that maximises expected IoU given the model's probabilities. This is the cheapest single test.
- **Clean the starting state.** The official answer is based on the true underlying speed, not the noisy observations. Smooth the last hour of history before deciding which links are queued now.
- **Train on far more data.** Build thousands of synthetic training windows from the train days using the official onset and ongoing definitions, instead of 80.
- **Add features that follow shockwave physics:**
  - how fast the queue tail should move, from upstream flow versus flow inside the queue;
  - where the tail will be at each forecast step compared with this link's position;
  - discharge rate at the head of the queue (capacity drop) versus upstream demand;
  - how long the queue has existed, and how often this link queues at this time of day;
  - speeds on upstream and downstream links a few steps back.

**B. State reconstruction (Task 1) – your "wave alignment" idea.**
- **Add a standard physics-based smoother as a baseline estimate.** The Adaptive Smoothing Method (Treiber–Helbing) spreads information downstream at about +80 km/h in free flow and upstream at about −15 to −20 km/h in congestion, weighted by speed. That is exactly the "directional upstream/downstream lag" idea.
- **Train a LightGBM model to correct that estimate.** It would learn the remaining error, using nearby observed values in time and along the corridor, lagged by how long a wave takes to travel between them.
- **Train it on masks that copy `MASK_SPEC.md` exactly.** Random masks would give a misleading validation score.
- **Then re-tune `physics_corrections` against Tasks 1 and 3 together**, since they share half the total weight.

**C. Target choice.** Predict speed as a fraction of free-flow speed, and flow per lane (the unit the scorer uses). Don't model density separately: the scorer derives it from flow and speed.

**D. Not now: physics-informed neural networks or graph diffusion.** With a 4-core CPU, no GPU and 5 submissions a day, the risk is high. Task 3 already rewards physical consistency through our Task 1 answers.

## 3. Experiment 12, in stages
0. **Measure first.** Build a local version of the full score using the official scorers in `official_repo/src/task*/score_*.py`, on 20 or more held-out train days. It should roughly reproduce Exp 11's 0.81 and show exactly which task is losing points.
1. **12a:** the per-window cell selection plus the cleaned starting state, using the current model.
2. **12b:** the fully retrained queue model with the physics features.
3. **12c:** the smoother plus LightGBM correction for Task 1, then physics re-tuning.
4. **Submit** only when the local score beats the Exp 11 replica by more than 0.01.

The target is roughly +0.15 on the queue score and 0.94 or better on state, which projects to about 0.87–0.90 overall. That projection is based on local estimates in code comments, not measured per-task scores; step 0 will confirm or correct it.