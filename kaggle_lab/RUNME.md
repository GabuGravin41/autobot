# Kaggle CLI lab — Titanic

Why this exists: I (Claude, via Cowork) discovered I cannot type into your
terminal — computer-use grants terminals "click-only" access (I can see and
click, never type or send keys), and this desktop has no shell-execution
bridge either. So I can't drive the Kaggle CLI on your machine directly.
What I *can* do is write every script, so the actual running is copy/paste
of a handful of commands — either by you, or by Antigravity, which does
have native execution access.

Run these from the `autobot/` project root, in order.

## 0. Verify auth

```
python kaggle_lab/00_verify.py
```

Should print `AUTHENTICATED OK` and a few competition refs. If this fails,
stop here — nothing below will work either.

## 1. Download data

```
python kaggle_lab/01_download.py
```

Pulls Titanic's train/test CSVs into `kaggle_lab/data/`.

## 2. Train baseline + generate submission

```
pip install pandas scikit-learn
python kaggle_lab/02_baseline.py
```

Prints 5-fold cross-validation accuracy, then writes
`kaggle_lab/data/submission.csv`. This is a real baseline (feature
engineering + random forest), not a placeholder — expect roughly
0.77-0.78 accuracy once scored.

## 3. Leaderboard method diagnostic (separate from the competition flow)

```
python kaggle_lab/03_leaderboard_diag.py
```

Paste the full output back in chat — this is what's blocking the fix to
`autobot/computer/kaggle_tool.py`'s `get_leaderboard()` (the old method it
called, `competition_view_leaderboard`, no longer exists in your installed
SDK version).

## 4. Submit — STOP, do not run this until you've told me to

This is the one irreversible step, so it's deliberately not in a script
you'd run by habit. When you're ready:

```
python -c "from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate(); print(api.competition_submit('kaggle_lab/data/submission.csv', 'Autobot lab baseline: RF on engineered features', 'titanic'))"
```

Titanic allows effectively unlimited daily submissions and has no
deadline, which is exactly why it was picked as the first live target —
there's no real cost to getting this step wrong.

## 5. Check your score

Kaggle scores Titanic submissions immediately. Check either on
kaggle.com/competitions/titanic/leaderboard, or once
`get_leaderboard()` is fixed (step 3), through `kaggle_tool.py` itself —
which is the more useful version of this step, since it's Autobot's own
code being exercised for real, not just the raw API.
