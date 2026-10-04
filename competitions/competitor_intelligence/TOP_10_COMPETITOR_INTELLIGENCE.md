# TOP 10 COMPETITOR INTELLIGENCE DOSSIER
## Reverse-Engineering Proven Techniques From Past Top Performers Across 4 Arenas
**Date**: October 5, 2026  
**Scope**: RSNA Knee, ARC-AGI-3, Gemma 4 Developer Agent, Enveda CASMI

---

## 1. RSNA Knee Abnormality Detection ($77,000 Prize Pool)
*Current Live Top 10 Leaderboard: `0.963` (rhoskeri) down to `0.959` (Scott Willis). Our Best: `0.943`.*

### Who Are the Top 10?
1. **`rhoskeri` (Rank #1 · `0.963`)**:
   - Past Competitions: RSNA 2024 Lumbar Spine Degenerative Classification, RSNA 2022 Cervical Spine Fracture Detection.
   - Core Lineage: Consistent RSNA specialist. In Lumbar Spine 2024, top solutions achieved separation not through monolithic 3D networks, but through **two-stage ROI localized classification**:
     * Stage 1: Keypoint/slice coordinate estimation isolating the anatomical level (in knee: joint line, femorotibial contact point, intercondylar notch).
     * Stage 2: Target-specific cropped classifiers per anatomical plane.
2. **`Pa3ðaJluHа` (Rank #2 · `0.962`)** & **`joinT fusion` (Rank #3 · `0.961`)**:
   - Heavy multi-plane view fusion (`Axial` + `Sagittal` + `Coronal`).
   - In past RSNA competitions (Lumbar Spine 2024, 2nd Place *IanPan-Kevin-Yuji-Bartley*), the winning leap came from **view decoupling**: training independent models on Axial vs. Sagittal vs. Coronal rather than forcing one model to learn all three views simultaneously!
3. **`Wassim Dobbi` (Rank #4 · `0.961`)**:
   - Consistent gold medalist across medical imaging competitions. Focuses heavily on multi-slice sequence modeling (BiLSTM / 1D CNN over 2D slice features).
4. **`TomKroumov` (Rank #6 · `0.959`)**:
   - Competitor `tom99763`. Known for high-resolution patch cropping, hard example mining, and metric-aligned calibration.

### The Proven Winning Pattern from Past RSNA Competitions:
* **The "Separate View, Separate Model" Principle (RSNA 2024 2nd Place IanPan)**:
  - Do NOT train one model to predict all 12 pathologies across all 3 planes.
  - Pathologies are strictly plane-specific:
    * `ACL` & `Contusion` $\rightarrow$ Evaluated primarily on **Sagittal** view.
    * `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA` $\rightarrow$ Evaluated on **Coronal** view.
    * `Patellofemoral (PF) OA`, `Effusion`, `Baker's Cyst` $\rightarrow$ Evaluated on **Axial** view.
  - Training **view-specialized models** and blending them by target yields an immediate +0.015 to +0.020 jump on Macro ROC-AUC!

---

## 2. ARC Prize 2026: ARC-AGI-3 ($850,000 Prize Pool)
*Current Live Top 10 Leaderboard: `55.89%` (Tufa Labs), `48.59%` (Yi-Chia Chen), `37.54%` (mtg). Our Best: `4.04%`.*

### Who Are the Top 10?
1. **`Tufa Labs` (Rank #1 · `55.89%`)**:
   - Zurich-based AI research institute with dedicated compute funding focused exclusively on ARC-AGI.
   - Core Lineage: Hosted the ARC Prize 2024 Winners (*"the ARChitects"* — Daniel Franzen & Jan Disselhoff).
   - Proven Techniques:
     * **Test-Time Training (TTT)** on the evaluation puzzle's demonstration grids.
     * **Exact Tokenization**: Visual grids tokenized as discrete character strings, not multi-modal patches.
     * **Symmetry Augmentations**: 8-fold dihedral transformations (D4: rotations and reflections) applied at inference time.
     * **Tree Search / DFS**: Depth-first search over token log-probabilities to reject syntactically invalid output grids before submission.
2. **`Yi-Chia Chen` (Rank #2 · `48.59%`)**:
   - Top Kaggle Grandmaster in discrete reasoning and symbolic AI. Integrates program synthesis with LLM candidate generation.
3. **`mtg` (Rank #3 · `37.54%`)** & **`Andrew Reed` (Rank #5 · `36.14%`)**:
   - High-throughput parallel inference engines executing candidate code execution verifiers in python sandboxes.

### The Proven Winning Pattern from Past ARC Competitions:
* **Test-Time Training (TTT) + Test-Time Search**:
  - The model does not just predict in one forward pass. It fine-tunes low-rank adapters on the few-shot demonstration pairs of the specific test puzzle, verifies reconstruction of the training pairs, and only then emits the test prediction!

---

## 3. Gemma 4 Developer Agent ($65,000 Prize Pool)
*Current Live Top 10 Leaderboard: `0.24` (#1) down to `0.15` (#10). Our Best: `0.08`.*

### Who Are the Top 10?
1. **Rank #1 (`0.24`)**:
   - Solved ~24% of SWE-bench tasks. Operates with tight test-reproduction loops.
2. **`Statistical Science by Lasme` (Rank #2 · `0.17`)**:
   - Statistical modeling and disciplined time allocation. Employs fast-fail thresholds on large issues.
3. **`Kirill Labzin` (Rank #5 · `0.17`)**:
   - Gold Medalist in the 2026 International AI Olympiad (IOAI), researcher specializing in Mixture-of-Experts (MoE) and agent routing.
   - Approach: Multi-agent routing separating symbol localization from patch generation.
4. **`Reiji Kobayashi` (Rank #4 · `0.17`)** & **`Bardia Bahadori` (Rank #6 · `0.17`)**:
   - Clean, hermetic tool-calling discipline, strictly preventing test file modifications.

### The Proven Winning Pattern from SWE-bench / Gemma:
* **The "Reproduction-First" Gate**:
  - Top agents never apply an edit until a minimal reproducing script has thrown the exact expected exception.
  - If reproduction fails in $\le 3$ tool calls, the agent aborts early and saves compute for solvable problems.

---

## 4. Enveda CASMI 2026: Molecule ID From Mass Spectra ($50,000 Prize Pool)
*Current Live Top 10 Leaderboard: `0.460` - `0.485`. Our Best: `0.417` (Rank #60).*

### The Proven Winning Pattern from Mass Spectrometry Challenges:
1. **Chemical Space Filtering via Precursor Exact Mass**:
   - Top teams use narrow $\pm 5$ ppm precursor mass filters to restrict candidate candidate formulas before computing fingerprint similarity.
2. **Ensemble of Morgan + MACCS + Topological Fingerprints**:
   - Blending disparate fingerprint types breaks molecular symmetry ties that single fingerprint representations miss.
3. **In-Silico Fragmentation Re-ranking (MetFrag / CFM-ID)**:
   - Re-ranking top-20 retrieved candidates by matching simulated bond dissociations against experimental MS2 peaks.
