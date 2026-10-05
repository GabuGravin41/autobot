# Enveda CASMI 2026: Molecule ID From Mass Spectra
## Predict 2D Chemical Structures of Small Molecules Detected in Complex Biological Extracts
### Engineering Strategy, Benchmarks, & Official Leaderboard Results

---

## 1. Executive Summary & Breakthrough
* **Competition**: Enveda CASMI 2026 - Molecule ID From Mass Spectra ($50,000 Prize Pool).
* **Baseline Score**: `0.380` (Rank #590).
* **Exp V44 PairTail Locked Top1 Score**: **`0.417`** (Rank #60 / 2,461 teams, Ref `56811892`).
* **Exp V45 Natural Product Knowledge SOTA**: **Scoring** (Ref `56840341`, queued 2026-10-05 02:42 UTC).
* **Kernel**: `daltongabrielomondi/casmi26-v45-natural-product-knowledge-sota`

---

## 2. Core Architecture & Enhancements in V44
1. **Two-Stage Cascade (Fingerprint Retrieval + Fragment Re-ranking)**:
   - Employs deep cosine retrieval over molecular embeddings, pulling candidate topological fingerprints.
   - Secondary stage locks top candidate structures using in-silico mass spectrometry fragmentation verification.
2. **PairTail Structure Calibration**:
   - Calibrates citation priors and structural stability to break ties between isomeric formulas.
   - Prevents degenerate zero-similarity predictions by enforcing top-1 structural consistency.
3. **Triple-Gatekeeper Verification**:
   - 100% finite, non-null SMILES/InChI representations adhering to standard chemical valency constraints.

---

## 3. Exp V45: Natural Product Knowledge Table SOTA
1. **Dmitri Gluzdov Natural Product Table Integration**:
   - Ingests `casmi26-natural-product-knowledge-table` incorporating LOTUS (Organism-Metabolite) and COCONUT (Collection of Natural Products) databases.
2. **Prior Boosting for Biogenic Scaffolds**:
   - Compounds found in known biological lineages receive calibrated prior boosts when mass spectra evidence matches candidate fragmentation profiles.
3. **Target**: Close the gap to Top 10 tier ($\ge 0.435$).

---

## 4. Exp V46 Planned Architecture: Quad-Channel Evidence & Seeded GBDT Calibration
Synthesizing prvsiyan's 4-channel decomposition:
1. **Channel 1 (Library Entropy Similarity)**: Normalized spectral entropy cosine matching against reference library.
2. **Channel 2 (Mass-Shifted Analog Propagation)**: Identifies near-neighbor parent masses and shifts fragmentation peaks by $\Delta m/z$.
3. **Channel 3 (In-Silico Bond Cleavage)**: Single- and double-bond graph cutting matching observed fragment peaks under $[-2, +2]$ hydrogen transfers.
4. **Channel 4 (Spectral Fingerprint Dot Product)**: Direct spectrum-to-fingerprint projection ($f \cdot z$).
5. **Fixed-Seed GBDT Meta-Learner**: Train `HistGradientBoostingClassifier(random_state=42)` across the 4 evidence channels to prevent the $\pm 0.006$ noise floor drift.
