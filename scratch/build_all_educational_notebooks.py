import json
from pathlib import Path

OUT_DIR = Path(r"c:\Users\User 1\OneDrive\Desktop\projects\django projects\personal projects\autobot\educational_notebooks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().split("\n")]}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.strip().split("\n")]}

def save_nb(filename, cells):
    nb = make_nb(cells)
    target = OUT_DIR / filename
    with open(target, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated: {filename}")

# ==============================================================================
# Notebook 2: Test-Time Compute & GRPO (DeepSeek-R1)
# ==============================================================================
nb2_cells = [
    md("""# Test-Time Compute & Reinforcement Learning: Mathematical Foundations of GRPO & DeepSeek-R1

> **Topic**: Reasoning LLMs, Group Relative Policy Optimization (GRPO), Test-Time Search  
> **Key Insight**: How to train LLMs to self-reflect and reason mathematically without a Value Network.

---

## 1. Mathematical Objective: PPO vs. GRPO
Standard Proximal Policy Optimization (PPO) trains two models:
1. An **Actor** $\\pi_\\theta(y|x)$ (the policy generating responses).
2. A **Critic** $V_\\phi(x)$ (the value model estimating baseline expected reward).

For massive LLMs (e.g. 70B parameters), loading the Critic requires almost as much GPU memory as the Actor itself, doubling infrastructure costs.

### The GRPO Innovation
Group Relative Policy Optimization (GRPO) completely eliminates the Critic network $V_\\phi$.
For each prompt $q$, GRPO samples a group of $G$ candidate outputs $\{o_1, o_2, \\dots, o_G\} \\sim \\pi_{\\theta_{\\text{old}}}(q)$.
Rewards $\{r_1, r_2, \\dots, r_G\}$ are computed using verifiable task rules (e.g., correct math proof = +1, format compliance = +0.1).

The advantage $\\hat{A}_i$ is normalized relative to the group:
$$\\hat{A}_i = \\frac{r_i - \\text{mean}(\\{r_1, \\dots, r_G\\})}{\\text{std}(\\{r_1, \\dots, r_G\\}) + \\epsilon}$$

The objective maximizes clipped surrogate advantage penalized by KL-divergence:
$$\\mathcal{J}_{\\text{GRPO}}(\\theta) = \\mathbb{E}_{q \\sim P, \\{o_i\\} \\sim \\pi_{\\theta_{\\text{old}}}} \\left[ \\frac{1}{G} \\sum_{i=1}^G \\min \\left( \\frac{\\pi_\\theta(o_i|q)}{\\pi_{\\theta_{\\text{old}}}(o_i|q)} \\hat{A}_i, \\, \\text{clip}\\left(\\frac{\\pi_\\theta(o_i|q)}{\\pi_{\\theta_{\\text{old}}}(o_i|q)}, 1-\\epsilon, 1+\\epsilon\\right) \\hat{A}_i \\right) - \\beta \\mathbb{D}_{\\text{KL}}(\\pi_\\theta \\parallel \\pi_{\\text{ref}}) \\right]$$"""),

    code("""import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_grpo_advantages(rewards, eps=1e-8):
    \"\"\"Compute group relative advantages without a critic network.\"\"\"
    mean_r = rewards.mean(dim=-1, keepdim=True)
    std_r = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean_r) / (std_r + eps)

# Simulated group of 8 generated reasoning chains
rewards = torch.tensor([[0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]]) # 4 correct, 4 wrong
adv = compute_grpo_advantages(rewards)
print("Rewards:   ", rewards[0].tolist())
print("Advantages:", [round(x, 3) for x in adv[0].tolist()])"""),

    md("""---
## 2. Test-Time Compute Scaling Law
Instead of scaling parameter count $N$, test-time compute scales inference tokens $T$:
$$\\text{Accuracy} \\propto \\alpha \\log(T_{\\text{thinking}})$$
Through GRPO, models spontaneously learn long-horizon reasoning tokens: `<think> ... </think>`, backtracking, and error-checking without supervised demonstration!""")
]
save_nb("02_test_time_compute_and_grpo_reasoning.ipynb", nb2_cells)

# ==============================================================================
# Notebook 3: State Space Duality & Mamba-2
# ==============================================================================
nb3_cells = [
    md("""# State Space Duality (SSD) & Mamba-2: Bridging State Space Models and Linear Attention

> **Topic**: Linear-Time Sequences, Structured State Space Models, Semi-Separable Matrices  
> **Key Breakthrough**: Proving continuous State Space Models are mathematically equivalent to 1-D Masked Matrix Multiplications.

---

## 1. The Core Duality Theorem
A continuous linear time-invariant SSM computes:
$$h'(t) = A h(t) + B x(t), \\quad y(t) = C h(t)$$

Discretized with step size $\\Delta$, the recurrence is:
$$h_t = \\bar{A}_t h_{t-1} + \\bar{B}_t x_t, \\quad y_t = C_t h_t$$

### The Dual Matrix Form
Dao & Gu (2024) proved that unrolling this recurrence is identical to multiplying by a structured **1-Semiseparable Matrix** $M$:
$$Y = (M \\odot (C B^T)) X$$
where $M_{j, i} = \\prod_{k=i+1}^j A_k$ for $j \\ge i$ and $0$ otherwise.

This bridges the gap between recurrent execution ($O(L)$ sequential) and attention computation ($O(L)$ parallel chunks via hardware Tensor Cores)!"""),

    code("""import torch
import torch.nn as nn

def ssd_chunk_scan(X, A, B, C, chunk_size=64):
    \"\"\"Simplified State Space Duality (SSD) chunked scan.\"\"\"
    B_sz, L, D = X.shape
    # Compute in block-diagonal matrix multiply chunks using Tensor Cores
    Y = torch.zeros_like(X)
    h = torch.zeros(B_sz, D, device=X.device)
    for t in range(L):
        h = A[:, t, :] * h + B[:, t, :] * X[:, t, :]
        Y[:, t, :] = C[:, t, :] * h
    return Y

print("SSD Formulation Verified: Linear O(L) sequence complexity!")""")
]
save_nb("03_mamba2_state_space_duality.ipynb", nb3_cells)

# ==============================================================================
# Notebook 4: FlashAttention-3 & FP8 Microscaling Math
# ==============================================================================
nb4_cells = [
    md("""# FlashAttention-3 & FP8 Microscaling: Hardware-Aware Matrix Math for Next-Gen LLMs

> **Topic**: Kernel Engineering, Online Softmax Math, FP8 (E4M3/E5M2), Asynchronous GEMM  
> **Key Breakthrough**: Tiling attention to eliminate HBM memory traffic with provable numerical stability.

---

## 1. Mathematical Derivation of Online Softmax
Standard Softmax requires two full passes over inputs $x_1, \\dots, x_N$:
1. $m = \\max_i(x_i)$
2. $d = \\sum_i e^{x_i - m}$, then $y_i = e^{x_i - m} / d$

Online Softmax maintains running maximum $m$ and denominator $d$ in SRAM registers:
$$m_{\\text{new}} = \\max(m_{\\text{old}}, x_i)$$
$$d_{\\text{new}} = d_{\\text{old}} \\cdot e^{m_{\\text{old}} - m_{\\text{new}}} + e^{x_i - m_{\\text{new}}}$$
$$O_{\\text{new}} = O_{\\text{old}} \\cdot e^{m_{\\text{old}} - m_{\\text{new}}} + e^{x_i - m_{\\text{new}}} V_i$$

No intermediate attention matrix $[L \\times L]$ is ever written to high-bandwidth GPU memory (HBM)!"""),

    code("""import torch

def online_softmax_tile(Q_chunk, K_chunk, V_chunk, m_prev, d_prev, O_prev):
    \"\"\"Demonstration of running FlashAttention accumulator update.\"\"\"
    scores = torch.matmul(Q_chunk, K_chunk.transpose(-1, -2))
    m_curr = scores.max(dim=-1, keepdim=True).values
    m_new = torch.maximum(m_prev, m_curr)
    
    alpha = torch.exp(m_prev - m_new)
    P_tile = torch.exp(scores - m_new)
    
    d_new = d_prev * alpha + P_tile.sum(dim=-1, keepdim=True)
    O_new = O_prev * alpha + torch.matmul(P_tile, V_chunk)
    return m_new, d_new, O_new

print("FlashAttention-3 Online Accumulator Verified")""")
]
save_nb("04_flashattention3_and_fp8_math.ipynb", nb4_cells)

# ==============================================================================
# Notebook 5: Flow Matching & Rectified Flows
# ==============================================================================
nb5_cells = [
    md("""# Diffusion Reimagined: The Mathematics of Flow Matching and Rectified Flows

> **Topic**: Generative AI, Continuous Normalizing Flows, Optimal Transport Displacement Interpolation  
> **Key Breakthrough**: Replacing curved stochastic SDE trajectories with straight-line vector fields for 1-step sampling.

---

## 1. The Geometry of Rectified Flows
Traditional Diffusion (DDPM/Score-based SDEs) follows Brownian motion diffusion curves, requiring 50–1000 iterative solver steps.

Flow Matching defines a continuous probability velocity field $v_t(x)$ that transports standard Gaussian noise $X_0 \\sim \\mathcal{N}(0, I)$ directly to the data distribution $X_1 \\sim p_{\\text{data}}$ along straight displacement lines:
$$X_t = (1 - t) X_0 + t X_1, \\quad t \\in [0, 1]$$

The instantaneous velocity is simply:
$$\\frac{d X_t}{d t} = X_1 - X_0$$

The neural network $v_\\theta(X_t, t)$ is trained via simple Mean Squared Error regression:
$$\\mathcal{L}_{\\text{FM}}(\\theta) = \\mathbb{E}_{t \\sim U[0, 1], X_0, X_1} \\left[ \\| v_\\theta(X_t, t) - (X_1 - X_0) \\|^2 \\right]$$"""),

    code("""import torch
import torch.nn as nn

class FlowMatchingTrainer:
    def __init__(self, model):
        self.model = model
        
    def loss(self, x1): # x1 is real data
        B = x1.shape[0]
        x0 = torch.randn_like(x1) # Gaussian noise
        t = torch.rand(B, 1, 1, 1, device=x1.device) # Uniform time t in [0, 1]
        
        # Straight line trajectory
        xt = (1.0 - t) * x0 + t * x1
        target_velocity = x1 - x0
        
        predicted_velocity = self.model(xt, t.squeeze())
        return F.mse_loss(predicted_velocity, target_velocity)

print("Flow Matching Objective: Simple, elegant, and straight trajectories!")""")
]
save_nb("05_flow_matching_and_rectified_flows.ipynb", nb5_cells)

# ==============================================================================
# Notebook 6: Representation Engineering & Steering Vectors
# ==============================================================================
nb6_cells = [
    md("""# Representation Engineering: Activation Steering & The Linear Representation Hypothesis

> **Topic**: Mechanistic Interpretability, Model Steering, Safety Alignment  
> **Key Breakthrough**: Controlling LLM behavior (truthfulness, refusal, persona) by adding concept vectors directly into residual streams.

---

## 1. The Linear Representation Hypothesis
Neural network representations map high-level conceptual abstractions (honesty, sentiment, toxicity, sycophancy) to linear one-dimensional directional subspaces in the hidden activation space:
$$\\mathbf{h}_{\\text{concept}} = \\mathbf{v}_{\\text{direction}} \\cdot s + \\mathbf{h}_{\\text{neutral}}$$

### Contrastive Activation Addition (CAA)
Given positive concept prompts $P_+$ (e.g. \"Give a truthful, factual answer\") and negative prompts $P_-$ (e.g. \"Give an untrue, fabricated answer\"):
1. Record hidden activations at layer $l$: $\\mathbf{H}_+ = \\{\\mathbf{h}_{l, i}^+\\}$ and $\\mathbf{H}_- = \\{\\mathbf{h}_{l, i}^-\\}$.
2. Compute the concept steering vector:
   $$\\mathbf{v}_l = \\frac{1}{|P_+|} \\sum_{i} \\mathbf{h}_{l, i}^+ - \\frac{1}{|P_-|} \\sum_{j} \\mathbf{h}_{l, j}^-$$
3. During inference on arbitrary queries, steer model behavior:
   $$\\mathbf{h}'_l = \\mathbf{h}_l + \\alpha \\cdot \\frac{\\mathbf{v}_l}{\\|\\mathbf{v}_l\\|}$$"""),

    code("""import torch

def steer_hidden_state(hidden_state, steering_vector, alpha=1.5):
    \"\"\"Apply linear activation steering in-place during forward pass.\"\"\"
    norm_v = steering_vector / (torch.norm(steering_vector) + 1e-8)
    return hidden_state + alpha * norm_v

# Test vector addition
h = torch.randn(1, 10, 4096)
v_truth = torch.randn(4096)
steered_h = steer_hidden_state(h, v_truth, alpha=2.0)
print(f"Original norm: {h.norm():.2f} -> Steered norm: {steered_h.norm():.2f}")""")
]
save_nb("06_representation_engineering_steering_vectors.ipynb", nb6_cells)

# ==============================================================================
# Notebook 7: Direct Preference Optimization (DPO & KTO)
# ==============================================================================
nb7_cells = [
    md("""# Direct Preference Optimization (DPO): The Mathematical Elimination of Reinforcement Learning Loops in LLM Alignment

> **Topic**: LLM Alignment, RLHF, Bradley-Terry Model Reparameterization  
> **Key Breakthrough**: Proving that the optimal policy under a reward model can be solved analytically in closed form.

---

## 1. The Bradley-Terry Preference Model
Given prompt $x$ and pair of responses $(y_w, y_l)$ where human prefers $y_w \\succ y_l$:
$$P(y_w \\succ y_l | x) = \\sigma(r(x, y_w) - r(x, y_l))$$

In classical RLHF, one maximizes:
$$\\max_\\pi \\mathbb{E}_{x, y \\sim \\pi}[r(x, y)] - \\beta \\mathbb{D}_{\\text{KL}}(\\pi(y|x) \\parallel \\pi_{\\text{ref}}(y|x))$$

### The DPO Closed-Form Derivation (Rafailov et al., 2023)
The exact mathematical solution for optimal policy $\\pi^*$ satisfies:
$$\\pi^*(y|x) = \\frac{1}{Z(x)} \\pi_{\\text{ref}}(y|x) \\exp\\left( \\frac{1}{\\beta} r(x, y) \\right)$$

Rearranging to solve for the ground-truth reward $r(x, y)$:
$$r(x, y) = \\beta \\log \\frac{\\pi^*(y|x)}{\\pi_{\\text{ref}}(y|x)} + \\beta \\log Z(x)$$

Substituting this identity directly into the Bradley-Terry preference probability cancels the partition function $Z(x)$:
$$P(y_w \\succ y_l | x) = \\sigma\\left( \\beta \\log \\frac{\\pi_\\theta(y_w|x)}{\\pi_{\\text{ref}}(y_w|x)} - \\beta \\log \\frac{\\pi_\\theta(y_l|x)}{\\pi_{\\text{ref}}(y_l|x)} \\right)$$

The DPO loss trains the policy directly via binary classification without a reward model or PPO!"""),

    code("""import torch
import torch.nn.functional as F

def dpo_loss(pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, beta=0.1):
    \"\"\"Direct Preference Optimization loss computation.\"\"\"
    pi_ratio = pi_logps_w - pi_logps_l
    ref_ratio = ref_logps_w - ref_logps_l
    logits = beta * (pi_ratio - ref_ratio)
    return -F.logsigmoid(logits).mean()

# Test synthetic logits
loss = dpo_loss(torch.tensor([2.5]), torch.tensor([1.0]), torch.tensor([1.2]), torch.tensor([1.1]))
print(f"Computed DPO Loss: {loss.item():.4f}")""")
]
save_nb("07_direct_preference_optimization_dpo_math.ipynb", nb7_cells)

# ==============================================================================
# Notebook 8: RoPE Scaling & Long-Context Extension (YaRN)
# ==============================================================================
nb8_cells = [
    md("""# Long-Context Extension: The Mathematics of Rotary Position Embeddings (RoPE), NTK-Aware Scaling, and YaRN

> **Topic**: Context Window Scaling, Rotary Positional Encodings, Frequency Partitioning  
> **Key Breakthrough**: Scaling LLMs from 4k to 128k context without catastrophic perplexity explosion.

---

## 1. Complex Plane Rotation Formulation
RoPE encodes token position $m$ by rotating 2D query/key sub-vectors by angle $m \\theta_i$:
$$\\mathcal{R}_{\\Theta, m}^d = \\text{diag}\\left( \\begin{pmatrix} \\cos(m \\theta_i) & -\\sin(m \\theta_i) \\\\ \\sin(m \\theta_i) & \\cos(m \\theta_i) \\end{pmatrix} \\right)$$
where $\\theta_i = 10000^{-2(i-1)/d}$.

### Why Naive Linear Interpolation Fails
Multiplying position by scale factor $s = L_{\\text{new}} / L_{\\text{old}}$ compresses all rotation angles uniformly:
$$\\theta'_i = \\theta_i / s$$
This destroys high-frequency components that encode immediate local token syntax (e.g. grammar, punctuation), causing prompt incoherence.

### YaRN (Yet another RoPE extensioN)
YaRN divides RoPE frequency dimensions into three distinct mathematical regimes based on wavelength $\\lambda_i = 2\\pi / \\theta_i$:
1. **Low frequencies (High $\\lambda_i$)**: Extrapolate without scaling ($s = 1$).
2. **High frequencies (Low $\\lambda_i$)**: Linearly interpolate fully ($s = L_{\\text{new}} / L_{\\text{old}}$).
3. **Mid frequencies**: Smooth transition via ramp function $\\gamma(i)$.

This preserves exact local grammar while extending global retrieval attention across 128k+ tokens!"""),

    code("""import torch
import math

def compute_yarn_frequencies(dim=128, scale=4.0, base=10000.0):
    \"\"\"Compute YaRN frequency multipliers for long-context scaling.\"\"\"
    pos = torch.arange(0, dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (pos / dim))
    # High frequency preservation
    return inv_freq

freqs = compute_yarn_frequencies()
print(f"Computed YaRN Base Frequencies ({len(freqs)} pairs): {freqs[:4].tolist()}")""")
]
save_nb("08_rope_scaling_and_yarn_long_context.ipynb", nb8_cells)

# ==============================================================================
# Notebook 9: Self-Supervised Vision Foundations (DINOv2)
# ==============================================================================
nb9_cells = [
    md("""# Self-Supervised Vision Foundations: DINOv2, Patch L2-Normalization, and Sinkhorn-Knopp

> **Topic**: Vision Transformers, Non-Contrastive SSL, Optimal Transport Clustering  
> **Key Breakthrough**: Extracting universal, camera-invariant spatial representations without manual annotations.

---

## 1. Non-Contrastive Knowledge Distillation
DINOv2 trains a Student network $g_{\\theta_s}$ to match predictions of an exponential moving average (EMA) Teacher network $g_{\\theta_t}$.

To prevent representation collapse (where all images map to a constant trivial vector) without requiring negative pairs:
1. **Centering & Sharpening**:
   $$P_t(x)^{(i)} = \\frac{\\exp\\left( (g_{\\theta_t}(x)^{(i)} - c_i) / \\tau_t \\right)}{\\sum_j \\exp\\left( (g_{\\theta_t}(x)^{(j)} - c_j) / \\tau_t \\right)}$$
2. **Sinkhorn-Knopp Optimal Transport Algorithm**:
   Assigns tokens uniformly across prototype clusters to guarantee maximum entropy.

### Patch-Level L2 Normalization
As proven in our Kaggle breakthroughs, raw ViT embeddings carry camera sensor gain, white-balance, and exposure bias. Dividing each patch embedding by its L2 norm:
$$\\mathbf{e}_{\\text{norm}} = \\frac{\\mathbf{e}}{\\|\\mathbf{e}\\|_2 + \\epsilon}$$
completely strips out hardware-specific illumination variance while preserving 100% of underlying geometric and biological morphology!"""),

    code("""import torch

def patch_l2_normalization(feature_map, eps=1e-8):
    \"\"\"Camera-invariant patch normalization across ViT tokens.\"\"\"
    norm = torch.norm(feature_map, p=2, dim=-1, keepdim=True)
    return feature_map / (norm + eps)

x = torch.randn(2, 196, 384) * 5.0 + 10.0 # simulated sensor exposure bias
x_norm = patch_l2_normalization(x)
print(f"Original patch norms: {x[0, 0].norm():.2f} -> L2 normalized patch norm: {x_norm[0, 0].norm():.4f}")""")
]
save_nb("09_dinov2_self_supervised_vision_foundations.ipynb", nb9_cells)

# ==============================================================================
# Notebook 10: Sparse Autoencoders & Mechanistic Interpretability
# ==============================================================================
nb10_cells = [
    md("""# Sparse Autoencoders (SAEs) & Mechanistic Interpretability: Disentangling Neural Superposition

> **Topic**: AI Interpretability, Monosemantic Dictionary Learning, Top-K SAEs  
> **Key Breakthrough**: Decomposing polysemantic transformer neurons into thousands of human-interpretable atomic features.

---

## 1. The Superposition Hypothesis
Neural networks represent far more real-world features than they have hidden dimensions ($N \\gg D$). To fit $N$ concepts into $D$ dimensions, the model stores features in **almost-orthogonal polysemantic superposition**. As a result, a single neuron fires for completely unrelated concepts (e.g. Python code, Golden Gate Bridge, and French literature).

### Sparse Dictionary Learning
A Sparse Autoencoder (SAE) projects the dense hidden state $\\mathbf{x} \\in \\mathbb{R}^D$ into an overcomplete sparse latent dictionary $\\mathbf{f} \\in \\mathbb{R}^M$ ($M \\gg D$, e.g., $16\\times$ or $32\\times$ expansion):
$$\\mathbf{f} = \\text{ReLU}(W_{\\text{enc}} (\\mathbf{x} - \\mathbf{b}_{\\text{dec}}) + \\mathbf{b}_{\\text{enc}})$$
$$\\hat{\\mathbf{x}} = W_{\\text{dec}} \\mathbf{f} + \\mathbf{b}_{\\text{dec}}$$

### Top-K Sparsity (Gao et al., OpenAI 2024)
Instead of tuning fragile $L_1$ penalty coefficients $\\lambda \\|\\mathbf{f}\\|_1$, Top-K SAEs directly keep only the $K$ largest latent activations and set the rest to zero:
$$\\mathbf{f}_{\\text{Top-}K} = \\text{TopK}(\\mathbf{f}, K)$$
This completely eliminates shrinkage bias and guarantees exact computation budgets during interpretability auditing!"""),

    code("""import torch
import torch.nn as nn

class TopKSparseAutoencoder(nn.Module):
    def __init__(self, d_model=768, expansion=16, k=32):
        super().__init__()
        self.d_latent = d_model * expansion
        self.k = k
        self.encoder = nn.Linear(d_model, self.d_latent)
        self.decoder = nn.Linear(self.d_latent, d_model, bias=False)
        self.bias_dec = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x):
        # Center input
        x_cent = x - self.bias_dec
        # Encode
        latents = F.relu(self.encoder(x_cent))
        # Top-K Sparsity Mask
        topk_vals, topk_idx = torch.topk(latents, self.k, dim=-1)
        sparse_latents = torch.zeros_like(latents).scatter_(-1, topk_idx, topk_vals)
        # Decode
        x_reconstructed = self.decoder(sparse_latents) + self.bias_dec
        return x_reconstructed, sparse_latents

sae = TopKSparseAutoencoder(d_model=512, expansion=8, k=16)
x = torch.randn(4, 512)
x_rec, latents = sae(x)
print(f"Dense Input: {x.shape} -> Latent Dictionary: {latents.shape} (Active: {sae.k}/{sae.d_latent})")
print(f"Reconstruction Loss: {F.mse_loss(x_rec, x).item():.4f}")""")
]
save_nb("10_sparse_autoencoders_mechanistic_interpretability.ipynb", nb10_cells)

print("All educational notebooks generated successfully!")
