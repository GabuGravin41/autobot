# Cutting-Edge AI Mathematical Breakthroughs: 10 Educational Notebooks

A curated curriculum of 10 publication-grade interactive Jupyter Notebooks dissecting the latest mathematical, algorithmic, and architectural breakthroughs in modern Deep Learning and Large Language Models.

---

## Curriculum Overview

| # | Notebook | Core Mathematical Concepts | Key Equations / Theorems |
|---|---|---|---|
| **01** | [`01_deepseek_mla_and_auxiliary_loss_free_moe.ipynb`](./01_deepseek_mla_and_auxiliary_loss_free_moe.ipynb) | Multi-Head Latent Attention (MLA), KV Compression, Decoupled RoPE, Auxiliary-Loss-Free MoE Load Balancing | $\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t$, $e_{i,t} = \text{TopK}(\mathbf{s}_t + \mathbf{b}_t)$ |
| **02** | [`02_test_time_compute_and_grpo_reasoning.ipynb`](./02_test_time_compute_and_grpo_reasoning.ipynb) | Test-Time Compute Scaling, Group Relative Policy Optimization (GRPO), Critic-Free RL, Reasoning Tokens | $\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G + \epsilon}$, $\mathcal{J}_{\text{GRPO}}(\theta)$ |
| **03** | [`03_mamba2_state_space_duality.ipynb`](./03_mamba2_state_space_duality.ipynb) | State Space Duality (SSD), Continuous SSMs to 1-D Masked Attention, Semi-Separable Matrices, Block-Decomposition | $M_{j,i} = C_j \left(\prod_{k=i+1}^j A_k\right) B_i$, $Y = (M \circ (Q K^T)) V$ |
| **04** | [`04_flashattention3_and_fp8_math.ipynb`](./04_flashattention3_and_fp8_math.ipynb) | Hardware-Aware Tiling, FP8 (E4M3 vs E5M2) Dynamic Scaling, Asynchronous Tensor Core Pipelines, Warp Specialization | $L_i = \max(L_{i-1}, M_i) + \log(e^{L_{i-1} - \dots} + \dots)$ |
| **05** | [`05_flow_matching_and_rectified_flows.ipynb`](./05_flow_matching_and_rectified_flows.ipynb) | Continuous Normalizing Flows, Rectified Flow ODEs, Optimal Transport Vector Fields, Straight-Trajectory Simulation | $\psi_t(x_0, x_1) = (1-t)x_0 + t x_1$, $v_t = x_1 - x_0$ |
| **06** | [`06_representation_engineering_steering_vectors.ipynb`](./06_representation_engineering_steering_vectors.ipynb) | Contrastive Activation Addition (CAA), Steering Vectors, Concept Direction Extraction, Linear Subspace Control | $\mathbf{v}_{\text{steer}} = \frac{1}{|P|}\sum h_p - \frac{1}{|N|}\sum h_n$ |
| **07** | [`07_direct_preference_optimization_dpo_math.ipynb`](./07_direct_preference_optimization_dpo_math.ipynb) | Closed-Form Bradley-Terry Optimization, Implicit Reward Re-parameterization, Reference Policy Regularization | $\mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)$ |
| **08** | [`08_rope_scaling_and_yarn_long_context.ipynb`](./08_rope_scaling_and_yarn_long_context.ipynb) | Rotary Position Embedding (RoPE), NTK-Aware Interpolation, YaRN (Yet another RoPE extensioN), Attention Temperature Ramp | $\theta_i' = \theta_i \cdot s^{-d/(d-2)}$, $\lambda(r) = \text{ramp}(r)$ |
| **09** | [`09_dinov2_self_supervised_vision_foundations.ipynb`](./09_dinov2_self_supervised_vision_foundations.ipynb) | Self-Distillation with No Labels (DINO), Swapped Cross-Entropy, Vision Transformer Register Tokens, Patch Invariance | $\mathcal{L}_{\text{DINO}} = - \sum p_t \log p_s$, $p(z) = \text{softmax}(z / \tau)$ |
| **10** | [`10_sparse_autoencoders_mechanistic_interpretability.ipynb`](./10_sparse_autoencoders_mechanistic_interpretability.ipynb) | Linear Representation Hypothesis, Superposition Resolution, Monosemantic Dictionary Learning, L1 / TopK Sparsity | $\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda \|f\|_1$, $\hat{x} = W_{\text{dec}} f + b_{\text{dec}}$ |

---

## Structure & Methodology
Every notebook is engineered according to three strict standards:
1. **Mathematical Rigor**: Derivations from first principles with full LaTeX mathematical proofs.
2. **From-Scratch PyTorch Implementations**: Zero black-box dependencies; every core layer, forward pass, and loss function is implemented directly in pure PyTorch and NumPy.
3. **Reproducible Numerical Experiments**: Standalone verification blocks demonstrating tensor shape transformations, numerical stability, and optimization behavior.
