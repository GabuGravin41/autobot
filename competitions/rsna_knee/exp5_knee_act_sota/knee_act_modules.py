"""
Knee-ACT: Knee Anatomical Cross-Plane Transformer Modules.
Implements:
1. Gated-MIL Focal Slice Attention (replaces mean pooling; preserves focal tears).
2. Anatomical Cross-Plane Target Router (routes 12 clinical targets to diagnostic planes).
3. Biomechanical Triad Coupling Head (Unhappy Triad, Pivot-Shift Contusion, Joint Capsule).
4. Symmetric Cross-Entropy (SCE) Loss (Noise-tolerant learning from report-derived labels).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

TARGET_NAMES = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture"
]

# Anatomical Mask: [12 targets, 3 planes (0: Sagittal, 1: Coronal, 2: Axial)]
# 0 = allowed, -1e4 = masked out
# Sagittal=0, Coronal=1, Axial=2
ANATOMICAL_PRIOR_MASK = torch.tensor([
    [0.0,   0.0, -1e4],  # 0: ACL (Sagittal + Coronal)
    [-1e4,  0.0,   0.0], # 1: MCL (Coronal + Axial)
    [0.0,   0.0, -1e4],  # 2: Medial Meniscus (Sagittal + Coronal)
    [0.0,   0.0, -1e4],  # 3: Lateral Meniscus (Sagittal + Coronal)
    [0.0,   0.0, -1e4],  # 4: Medial OA (Coronal + Sagittal)
    [0.0,   0.0, -1e4],  # 5: Lateral OA (Coronal + Sagittal)
    [0.0,  -1e4,   0.0], # 6: PF OA (Axial + Sagittal)
    [0.0,   0.0,   0.0], # 7: Effusion (All planes)
    [0.0,   0.0,   0.0], # 8: Synovitis (All planes)
    [0.0,  -1e4,   0.0], # 9: Baker's Cyst (Axial + Sagittal)
    [0.0,   0.0,   0.0], # 10: Contusion (All planes)
    [0.0,   0.0,   0.0], # 11: Fracture (All planes)
], dtype=torch.float32)

# Target Noise Weights (downweights high-disagreement report labels)
TARGET_LOSS_WEIGHTS = torch.tensor([
    1.0,  # ACL (high agreement)
    0.6,  # MCL (high discordance with reports)
    0.9,  # Medial Meniscus
    0.9,  # Lateral Meniscus
    0.8,  # Medial OA
    0.8,  # Lateral OA
    0.7,  # PF OA
    1.0,  # Effusion (clear signal)
    0.8,  # Synovitis
    0.5,  # Baker's Cyst (frequently omitted in dictations)
    1.0,  # Contusion
    1.0   # Fracture (high agreement)
], dtype=torch.float32)


class GatedMILSpatialAttention(nn.Module):
    """
    Gated Multiple Instance Learning (Gated-MIL) Slice Pooling.
    Isolates focal tears (ACL, meniscus) on 1-2 slices without signal dilution,
    while naturally broadening for diffuse findings (effusion, OA).
    
    Formula:
        a_s = softmax(w^T (tanh(V h_s) * sigmoid(U h_s)))
        z = sum(a_s * h_s)
    """
    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.v_proj = nn.Linear(in_features, hidden_dim)
        self.u_proj = nn.Linear(in_features, hidden_dim)
        self.w_proj = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (B, S, D) where S is number of slices, D is feature dim.
        Returns:
            pooled: (B, D) aggregated plane representation.
            attn_weights: (B, S, 1) normalized attention distribution over slices.
        """
        v = torch.tanh(self.v_proj(x))
        u = torch.sigmoid(self.u_proj(x))
        gated = v * u
        scores = self.w_proj(gated)  # (B, S, 1)
        weights = F.softmax(scores, dim=1)  # (B, S, 1)
        pooled = torch.sum(weights * x, dim=1)  # (B, D)
        return pooled, weights


class AnatomicalCrossPlaneRouter(nn.Module):
    """
    Clinical Target Query Cross-Attention Router.
    12 learnable target query embeddings attend across the 3 canonical plane tokens
    (Sagittal, Coronal, Axial), constrained by the clinical anatomical prior mask.
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_targets = 12
        
        # 12 learnable query embeddings for the 12 clinical targets
        self.target_queries = nn.Parameter(torch.randn(1, self.num_targets, embed_dim) * 0.02)
        
        # Cross-attention block
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        
        # Linear classifier per target: map attended token to scalar logit
        self.target_classifiers = nn.Linear(embed_dim, 1)
        
        # Register the static prior mask buffer
        self.register_buffer("prior_mask", ANATOMICAL_PRIOR_MASK)

    def forward(self, plane_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            plane_tokens: Tensor of shape (B, 3, D) containing pooled features for
                          [Sagittal, Coronal, Axial].
        Returns:
            base_logits: Tensor of shape (B, 12) unconstrained target logits.
            attended_features: Tensor of shape (B, 12, D) target-specific feature representations.
        """
        B = plane_tokens.size(0)
        q = self.target_queries.expand(B, -1, -1)  # (B, 12, D)
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(plane_tokens)  # (B, 3, D)
        
        # Mask shape: (12, 3) broadcast across batch and heads in PyTorch MHA
        # attn_mask in MHA: (L, S) where L=12, S=3
        out, attn_weights = self.cross_attn(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            attn_mask=self.prior_mask  # (12, 3)
        )
        attended = self.norm_out(q + out)  # (B, 12, D)
        
        # Classification logits per target
        logits = self.target_classifiers(attended).squeeze(-1)  # (B, 12)
        return logits, attended


class BiomechanicalTriadHead(nn.Module):
    """
    Kinematic & Pathological Co-Occurrence Refinement Layer.
    Injects orthopedic domain constraints:
    1. Pivot-Shift Bone Bruise: ACL Tear <-> Bone Contusion.
    2. O'Donoghue's Unhappy Triad: ACL Tear <-> MCL Tear <-> Medial Meniscus Tear.
    3. Joint Capsule Reactive Effusion: Effusion <-> Synovitis.
    4. Chronic Meniscal Derangement: Medial Meniscus <-> Medial OA.
    """
    def __init__(self, num_targets: int = 12, hidden_dim: int = 64):
        super().__init__()
        # Interaction feature dimension: 4 clinical interactions
        self.interaction_dim = 4
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.interaction_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_targets)
        )
        # Initialize final projection near zero so initial predictions match base logits
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

    def forward(self, base_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            base_logits: Tensor of shape (B, 12) from cross-plane router.
        Returns:
            refined_logits: Tensor of shape (B, 12) with biomechanical coupling.
        """
        probs = torch.sigmoid(base_logits)
        
        # 0: ACL, 1: MCL, 2: MedMen, 3: LatMen, 4: MedOA, 5: LatOA,
        # 6: PFOA, 7: Effusion, 8: Synovitis, 9: Bakers, 10: Contusion, 11: Fracture
        p_acl = probs[:, 0:1]
        p_mcl = probs[:, 1:2]
        p_med_men = probs[:, 2:3]
        p_med_oa = probs[:, 4:5]
        p_effusion = probs[:, 7:8]
        p_synovitis = probs[:, 8:9]
        p_contusion = probs[:, 10:11]
        
        # Interactions
        inter_pivot_shift = p_acl * p_contusion                           # (B, 1)
        inter_unhappy_triad = p_acl * p_mcl * p_med_men                   # (B, 1)
        inter_capsule_reactive = p_effusion * p_synovitis                 # (B, 1)
        inter_root_wear = p_med_men * p_med_oa                            # (B, 1)
        
        interactions = torch.cat([
            inter_pivot_shift,
            inter_unhappy_triad,
            inter_capsule_reactive,
            inter_root_wear
        ], dim=1)  # (B, 4)
        
        residual = self.residual_mlp(interactions)  # (B, 12)
        return base_logits + residual


class KneeACTModel(nn.Module):
    """
    Complete Knee-ACT Model:
    Backbone (per-slice feature extractor) -> Gated-MIL (per-plane) ->
    Anatomical Router (per-target) -> Biomechanical Triad Coupling.
    """
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 3 independent Gated-MIL pooling blocks for Sagittal, Coronal, Axial
        self.mil_sagittal = GatedMILSpatialAttention(feature_dim, hidden_dim)
        self.mil_coronal = GatedMILSpatialAttention(feature_dim, hidden_dim)
        self.mil_axial = GatedMILSpatialAttention(feature_dim, hidden_dim)
        
        # Anatomical Cross-Plane Target Router
        self.router = AnatomicalCrossPlaneRouter(embed_dim=feature_dim, num_heads=num_heads)
        
        # Biomechanical Triad Coupling Head
        self.triad_head = BiomechanicalTriadHead(num_targets=12, hidden_dim=64)

    def forward(
        self,
        feat_sag: torch.Tensor,
        feat_cor: torch.Tensor,
        feat_ax: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            feat_sag: (B, S_sag, D) slice feature embeddings for Sagittal.
            feat_cor: (B, S_cor, D) slice feature embeddings for Coronal.
            feat_ax:  (B, S_ax, D) slice feature embeddings for Axial.
        Returns:
            Dictionary containing:
                "logits": (B, 12) final calibrated logits.
                "base_logits": (B, 12) pre-triad logits.
                "attn_sag": (B, S_sag, 1) slice attention weights for Sagittal.
                "attn_cor": (B, S_cor, 1) slice attention weights for Coronal.
                "attn_ax":  (B, S_ax, 1) slice attention weights for Axial.
        """
        # Step 1: Gated-MIL slice pooling per plane
        pool_sag, attn_sag = self.mil_sagittal(feat_sag)  # (B, D)
        pool_cor, attn_cor = self.mil_coronal(feat_cor)  # (B, D)
        pool_ax, attn_ax = self.mil_axial(feat_ax)        # (B, D)
        
        # Step 2: Assemble canonical plane tokens
        plane_tokens = torch.stack([pool_sag, pool_cor, pool_ax], dim=1)  # (B, 3, D)
        
        # Step 3: Anatomical Cross-Plane Routing
        base_logits, attended_features = self.router(plane_tokens)  # (B, 12)
        
        # Step 4: Biomechanical Triad Refinement
        final_logits = self.triad_head(base_logits)  # (B, 12)
        
        return {
            "logits": final_logits,
            "base_logits": base_logits,
            "attn_sag": attn_sag,
            "attn_cor": attn_cor,
            "attn_ax": attn_ax,
        }


class SymmetricCrossEntropyLoss(nn.Module):
    """
    Symmetric Cross-Entropy (SCE) Loss with Target Weighting.
    Neutralizes report-label noise by combining Standard CE with Reverse CE (RCE):
        L_SCE = alpha * CE(p, y) + beta * RCE(y, p)
    Prevents gradient explosions on discordant report extractions (MCL, Baker's).
    """
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        eps: float = 1e-7,
        use_weights: bool = True
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.use_weights = use_weights
        self.register_buffer("weights", TARGET_LOSS_WEIGHTS)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 12) raw model logits.
            targets: (B, 12) continuous soft labels in [0, 1].
        """
        probs = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
        targets = targets.clamp(self.eps, 1.0 - self.eps)
        
        # Standard Cross-Entropy (CE)
        ce = - (targets * torch.log(probs) + (1.0 - targets) * torch.log(1.0 - probs))
        
        # Reverse Cross-Entropy (RCE)
        rce = - (probs * torch.log(targets) + (1.0 - probs) * torch.log(1.0 - targets))
        
        # Symmetric Combination
        sce = self.alpha * ce + self.beta * rce  # (B, 12)
        
        if self.use_weights:
            sce = sce * self.weights.unsqueeze(0)
            
        return sce.mean()
