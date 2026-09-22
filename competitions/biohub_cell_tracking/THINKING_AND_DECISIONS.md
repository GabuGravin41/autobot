# Autobot Autonomous Kaggle Playbook: Biohub - Cell Tracking During Development

> **Competition**: [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)  
> **Evaluation Metric**: Tracking Metric (F1 Detection + Mitotic Division Recall + Lineage Association)  
> **Prize Pool**: $60,000 USD | **Deadline**: 2026-09-29 (8 days remaining!)  
> **Submission Allowance**: 5 submissions / day (High-Conviction Precision Mode)  
> **Execution Strategy**: 100% Remote Kaggle Cloud Execution on GPU (T4 / P100)  

---

## 1. Competition Overview & Problem Topology
Detect and track thousands of developing zebrafish cells through 4D volumetric space and time (Zarr 3D volumes across multiple developmental time points). The challenge combines:
1. **Volumetric Cell Center Detection**: Identifying cell centroids (z, y, x) from high-noise 3D light-sheet fluorescence microscopy.
2. **Temporal Frame-to-Frame Relinking**: Linking cells across time frames despite cell migration, deformation, and microscopy motion blur.
3. **Mitotic Cell Division Detection**: Detecting when a single mother cell divides into two distinct daughter cells (lineage branching).

---

## 2. Public SOTA Intelligence & Baseline Dissection
- **Champion Benchmark**: `reyhanksatria/biohub-cell-tracking-0-947-lb` (**0.947 LB**)
- **Architecture Stack**:
  1. **Pre-Trained Volumetric 3D U-Net Checkpoints**: Mounts `reyhanksatria/biohub-deepcenterunet3d-center-prior-v1` and `reyhanksatria/biohub-tracking-support-pack`.
  2. **Harmonic Probability Fusion**: Computes bidirectional forward-backward association probabilities between consecutive frames.
  3. **Integer Linear Programming (ILP) Tracking**: Solves global optimal multi-target tracking constraints with explicit penalty terms for cell appearance, disappearance, and division branching.
  4. **Short-Track Rescue & DeepCenter Gap Veto**: Filters false-positive spurious tracks while rescuing real daughter cells through adaptive neighborhood density checks.

---

## 3. Autobot Experiment Registry

| Exp ID | Kernel Slug | Approach | Model Family | Hardware | Status | Local CV Score | Public LB |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 0 (Previous)** | `autobot-biohub-baseline-july` | Baseline Tracker | Tracking | GPU | Complete | - | **0.898** (Rank #2,247) |
| **Exp 1 (Completed & Verified)** | `autobot-biohub-exp1-sota-repro` | DeepCenter 3D U-Net + Harmonic ILP Fusion (0.947 SOTA Repro) | 3D U-Net / Dual-Seed / ILP | 2x GPU (T4) | Complete & Verified | **0.9511** (Held-Out Proxy) | Target: **0.947+** |

---

## 4. Experiment 1 Detailed Execution Log & Discoveries

### Root-Cause Analysis & Fixes:
1. **Dataset Mounting Resolution**:
   - The initial metadata specified invalid dataset sources under `reyhanksatria/*` which failed mounting.
   - The actual Kaggle datasets were authored by `pilkwang`:
     - `pilkwang/biohub-tracking-support-pack-50ep-v1` (containing all offline dependency wheels and primary UNet weights)
     - `pilkwang/biohub-deepcenter-unet3d-center-prior-v1` (DeepCenter UNet3D checkpoint)
     - `pilkwang/biohub-temporal-unet3d-seed314159-v1` (Secondary temporal model checkpoint)
   - Patched `kernel-metadata.json` and `main.py` path lookups to bind directly to the mounted `/kaggle/input/` paths.

2. **60-Second Liveness Verification Rule**:
   - Initial push failed within 30s due to missing dataset paths. The rule caught this immediately, preventing wasted cloud GPU time.
   - After updating dataset slugs and path resolutions, re-pushed kernel version 2 which achieved `KernelWorkerStatus.RUNNING` stably.
   - Total pipeline runtime was **99.8 minutes** on dual GPU, executing 8-view D4 spatial TTA, dual-seed temporal association, validation post-processing sweep (`ppsweep`), and ILP multi-target tracking.

3. **Output & Submission Integrity Verification**:
   - Generated outputs downloaded to `competitions/biohub_cell_tracking/output_exp1_sota/`.
   - **`submission.csv`**:
     - Total rows: **241,356**
     - Total nodes: **122,808**
     - Total edges: **118,548**
     - NaN count: **0** (Zero NaNs across all columns)
     - Schema: exact match `['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']`
     - Contiguous IDs: `0` to `241,355`
     - Test dataset coverage: 100% complete across all 4 test sets (`44b6_0113de3b`, `44b6_0b24845f`, `6bba_05b6850b`, `6bba_05db0fb1`)
     - Biological graph topology:
       - Max in-degree = 1 (no cell fusion)
       - Max out-degree = 2 (at most binary mitotic division)
       - Total mitotic division parents detected: **124** (55 in `44b6_0113de3b`, 25 in `44b6_0b24845f`, 10 in `6bba_05b6850b`, 34 in `6bba_05db0fb1`)
   - **`ppsweep_selected.json`**:
     - Automatically evaluated candidates on held-out validation movies.
     - Selected candidate: `tight55` (`MOTION_RELINK_TIGHT_UM: 5.5`).
     - Proxy score increased from baseline **0.9490** to **0.9511**.

4. **Code Competition Submission Protocol**:
   - Biohub is a strict **Kaggle Code Competition**; manual CSV upload via API fails with `400: Submission not allowed: This competition only accepts Submissions from Notebooks.`
   - Code submissions mandate that internet access must be disabled (`enable_internet: false`).
   - All wheels are pre-packaged offline in `pilkwang/biohub-tracking-support-pack-50ep-v1`, allowing 100% offline execution.
   - Pushed Version 3 with `"enable_internet": "false"` to enable submission triggering.
   - **Submission CLI Command**: Once Version 3 completes, submit to Kaggle leaderboard via:
     `kaggle competitions submit biohub-cell-tracking-during-development -k daltongabrielomondi/autobot-biohub-exp1-sota-repro -v 3 -f submission.csv -m "Autobot Exp 1: DeepCenter 3D U-Net + Harmonic Probability ILP SOTA Baseline"`

---

## 5. Cross-Competition Prior Transfer (The Grandmaster Archive)
Key analogies from past medical & biological vision competitions:
- **RSNA 3D CT/MRI Challenges**: 3D volumetric slice aggregation, Test-Time Augmentation (TTA) with flip/rotation on z-planes.
- **Sartorius / HuBMAP**: Watershed seed placement and morphological border erosion to prevent cell coalescence.
- **Cell Tracking Challenge (CTC)**: Graph Hungarian bipartite matching with lookahead window (gap <= 2) for cell division recovery.

---

## 6. Submission Economics Policy
- Daily Quota: **5 submissions / day**.
- **Rule**: Submissions must verify exact Zarr test set ID coverage, biological graph validity (in-degree <= 1, out-degree <= 2), and monotonic improvement against the 0.947 anchor.