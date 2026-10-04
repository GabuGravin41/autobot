# Enveda CASMI 2026: Molecule ID From Mass Spectra
## Predict 2D Chemical Structures of Small Molecules Detected in Complex Biological Extracts
### Engineering Strategy, Benchmarks, & Official Leaderboard Results

---

## 1. Executive Summary & Breakthrough
* **Competition**: Enveda CASMI 2026 - Molecule ID From Mass Spectra ($50,000 Prize Pool).
* **Baseline Score**: `0.380` (Rank #590).
* **Exp V44 PairTail Locked Top1 Score**: **`0.417`**
* **Official Leaderboard Rank**: **#60** out of 2,461 teams worldwide (Top 2.4% tier — updated live).
* **Submission ID**: `56811892`
* **Kernel**: `daltongabrielomondi/casmi26-v44-pairtail-locked-top1` (Version 1)
* **Date Completed**: `2026-10-04 04:55 UTC`.

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
