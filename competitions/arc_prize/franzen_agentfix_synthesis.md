# ARC Prize 2026: Synthesis of Daniel Franzen's M2 Rotation Canon & AgentFix

## 1. Problem Formulation: Spatial Orientation Noise in Grid Worlds
In ARC-AGI-3 grid-world tasks, objects frequently rotate (e.g. arrows pointing in 4 cardinal directions, Tetris-like tetrominoes, rotating obstacles).
Existing harnesses (including Tufa Labs Duck harness and standard AgentFix) hash objects by:
$$\text{norm}(cells) = \text{sort}((r - \min r, c - \min c))$$
$$\text{hash} = \text{sha1}(color, \text{norm}(cells))$$

### Limitations of Standard Hashing:
1. **Rotation Blindness**: A rotated object receives a completely distinct ID from its original form, forcing the LLM reasoner to re-learn its identity and behavior from scratch.
2. **False Symmetry Attribution**: Choosing reference orientations based on colored pixels makes the canonical orientation dependent on color, confounding shape with hue.

---

## 2. Daniel Franzen's Canonical Object Signature Innovation
From our audit of Franzen's Milestone 2 winning codebase:
1. **Color-Free Shape Rotation**:
   ```python
   norm = _normalize_cells(cells)
   variants = [norm]
   for _ in range(3):
       variants.append(_rotate_cells_90(variants[-1]))
   shape_hashes = [_cells_hash(v, "__shape__") for v in variants]
   k = shape_hashes.index(min(shape_hashes))  # Orientation chosen purely from geometry!
   ```
2. **Four-Tuple Geometric Invariant**:
   - `canonical_hash`: Identical for all rotated copies of the same shape + color.
   - `rotation`: Degrees clockwise from canonical orientation ($0^\circ, 90^\circ, 180^\circ, 270^\circ$).
   - `rotational_symmetry`: Order of symmetry ($4$ for squares, $2$ for dominoes/bars, $1$ for chiral shapes).
   - `pose_hash`: Classic rotation-sensitive hash of the observed pose.

---

## 3. Integration into AgentFix World Model
In our current Exp 5 architecture (`daltongabrielomondi/autobot-arc3-exp5-balanced-agentfix-sota`):
- `AGENTFIX_MEMORY=1` retains the persistent world model across frame resets.
- By augmenting the object state dictionary with Franzen's `(canonical_hash, rotation, symmetry)`:
  - The reasoning agent immediately knows when an obstacle or agent has simply turned $90^\circ$ without losing track of its identity.
  - The vLLM context token footprint is preserved (compact integers instead of duplicate object descriptions).

---

## 4. Next Iteration Candidate: Exp 6 (Franzen Rotation-Canon + AgentFix MTP SOTA)
- **Engine**: FlashNext NVFP4 MTP 3 Speculative Engine (`kv5-bf16-mtp3-c8-cg32`).
- **Context**: 16K context window with 16 concurrent sequences.
- **Segmentation**: Daniel Franzen D4 rotation canonicalization with color-free orientation selection.
- **Action Masking**: HUD action filtering + game budget allocation.
