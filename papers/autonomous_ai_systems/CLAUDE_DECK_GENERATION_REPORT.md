I rewrote `papers/autonomous_ai_systems/generate_deck.py`, but I haven't run it. Running `python` was blocked because this session can't approve commands, so the `.pptx` has not been rebuilt and I can't confirm it runs with zero errors. I checked the python-pptx calls and the positions of every element against the slide by hand instead. Please run:

```
python papers/autonomous_ai_systems/generate_deck.py
```

The old script would have crashed: it used `COLOR_AMBER` on slide 5 without defining it. That's fixed.

## Deck architecture (10 slides, 13.333 × 7.5 in, dark navy)

**Slides 2–7 all use the same three-part layout:**
- **Header:** a small category tag, a short title (28pt, under ~35 characters), and one large metric on the right (36pt, e.g. **18.4 h**, **864×**, **Top 4.88%**).
- **Figure:** takes up about 55–65% of the slide. Each figure is scaled to fit without distortion, and its built-in title is cropped off so it doesn't repeat the slide title.
- **Bottom row:** two metric badges (value on top, small label underneath) and one outlined takeaway line of 40–45 characters, e.g. "→ Cheap loop notices. Expensive loop thinks."

| # | Slide | Main visual | Large metric |
|---|---|---|---|
| 1 | Title | 3 circles sized by time horizon (2 m, 30 m, 12 h) + 3 large stat cards | 12+ h · 6 · $5 |
| 2 | Horizon gap | `fig_horizon_gap.png` | 360× |
| 3 | Dual-loop | `fig_supervision_architecture.png` | 2 m / 30 m |
| 4 | Budget trap | `fig2_swe_runtime_math.png` | 18.4 h |
| 5 | Boundary drift | `fig_boundary_drift.png` | Hour 2.1 |
| 6 | Standings | `fig_multidomain_standings.png` | Top 4.88% |
| 7 | Economics | `fig1_cost_comparison.png` | 864× |
| 8 | Production | 2 × 3 grid of number tiles (50.9 ms, $150, FHIR R4…) | — |
| 9 | Three laws | 3 numbered cards, each led by a 44pt number (5 : 1, ≤ 12 h, Every hop) | — |
| 10 | Roadmap | Arrow-shaped Measure → Generalize → Prove steps, each with 3 short item chips, plus a closing line | — |

**Design choices:**
- **Detail lives in speaker notes.** Everything from the old side cards (the 8.0 min × 120 = 18.4 h working, CIEL codes, cost breakdown) moved there, so it's available while presenting but not on the slides.
- **Styling:** Calibri throughout, with text inset inside the cards. I removed the decorative cyan bar from the title slide.
- **Paths** are now relative to the script instead of a hard-coded absolute path.
- **Locked file:** a PowerPoint lock file (`~$…`) is sitting next to the deck, which suggests it's open. If the save fails because of that, the script writes `…_new.pptx` and prints a warning.

**Things to check:**
- **The cost title changed.** The old title said "1,000x advantage", but your figure says 864×. I used **864×** so they match.
- **One figure has a cut-off label.** In `fig_multidomain_standings.png`, the Traffic Flow label runs off the right edge ("0.79155 SOTA Anch…"). That's inside the image, so it has to be fixed in `generate_pitch_figures.py` (widen the x-limit or the right margin) and regenerated.
- **Slides 2 and 7 are a bit under the 65% target.** Their figures are very wide (horizon) or fairly tall (cost), so after scaling they fill about 55% of the slide. Counting the bottom row brings it to about 62%.
- **I haven't looked at the rendered slides.** After you run the script, check for text overflow, especially the takeaway chips on slides 2–7 and the tiles on slide 8.