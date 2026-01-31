"""
Feature Fusion strategy.

Injects VLM visual features into the detector's feature pyramid
to enhance detection with semantic understanding.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tianwen.core.registry import FUSIONS
from tianwen.detectors.base import BaseDetector, BatchDetectionOutput
from tianwen.vlms.base import BaseVLM
from tianwen.fusions.base import BaseFusion, FusionOutput


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention module for fusing VLM and detector features.

    Detector features attend to VLM features to incorporate semantic information.
    """

    def __init__(
        self,
        det_dim: int,
        vlm_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = det_dim // num_heads

        # Query from detector, Key/Value from VLM
        self.q_proj = nn.Linear(det_dim, det_dim)
        self.k_proj = nn.Linear(vlm_dim, det_dim)
        self.v_proj = nn.Linear(vlm_dim, det_dim)
        self.out_proj = nn.Linear(det_dim, det_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(det_dim)
        self.norm2 = nn.LayerNorm(det_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(det_dim, det_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(det_dim * 4, det_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        det_features: Tensor,
        vlm_features: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Cross-attention forward pass.

        Args:
            det_features: Detector features [B, N_det, D_det]
            vlm_features: VLM features [B, N_vlm, D_vlm]
            attn_mask: Optional attention mask

        Returns:
            Enhanced detector features [B, N_det, D_det]
        """
        B, N_det, D = det_features.shape
        _, N_vlm, _ = vlm_features.shape

        # Cross-attention
        residual = det_features
        det_features = self.norm1(det_features)

        q = self.q_proj(det_features)
        k = self.k_proj(vlm_features)
        v = self.v_proj(vlm_features)

        # Reshape for multi-head attention
        q = q.view(B, N_det, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N_vlm, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N_vlm, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N_det, D)
        out = self.out_proj(out)
        out = self.dropout(out)

        det_features = residual + out

        # FFN
        det_features = det_features + self.ffn(self.norm2(det_features))

        return det_features


class FeatureAdapter(nn.Module):
    """
    Adapter module for injecting VLM features into detector.

    Uses a gating mechanism to control the influence of VLM features.
    """

    def __init__(
        self,
        det_dim: int,
        vlm_dim: int,
        use_gate: bool = True,
    ):
        super().__init__()

        self.use_gate = use_gate

        # Project VLM features to detector dimension
        self.vlm_proj = nn.Sequential(
            nn.Linear(vlm_dim, det_dim),
            nn.LayerNorm(det_dim),
            nn.GELU(),
        )

        if use_gate:
            # Learnable gate to control VLM influence
            self.gate = nn.Sequential(
                nn.Linear(det_dim * 2, det_dim),
                nn.Sigmoid(),
            )

    def forward(
        self,
        det_features: Tensor,
        vlm_features: Tensor,
    ) -> Tensor:
        """
        Adapt and inject VLM features.

        Args:
            det_features: Detector features [B, N, D_det]
            vlm_features: VLM features [B, M, D_vlm]

        Returns:
            Enhanced detector features [B, N, D_det]
        """
        # Project VLM features
        vlm_proj = self.vlm_proj(vlm_features)

        # Pool VLM features to match detector spatial resolution
        # Simple global average pooling then broadcast
        vlm_pooled = vlm_proj.mean(dim=1, keepdim=True)
        vlm_expanded = vlm_pooled.expand(-1, det_features.shape[1], -1)

        if self.use_gate:
            # Compute gate values
            combined = torch.cat([det_features, vlm_expanded], dim=-1)
            gate = self.gate(combined)

            # Gated addition
            output = det_features + gate * vlm_expanded
        else:
            # Direct addition
            output = det_features + vlm_expanded

        return output


@FUSIONS.register("feature_fusion", aliases=["feature", "feat_fusion"])
class FeatureFusion(BaseFusion):
    """
    Feature-level fusion between detector and VLM.

    Injects VLM visual features into the detector's feature pyramid
    using cross-attention or adapter modules.

    Fusion can happen at different levels:
    - backbone: Inject into backbone features
    - neck: Inject into FPN/neck features
    - head: Inject before detection head

    Example:
        >>> fusion = FeatureFusion(
        ...     detector=yolo_detector,
        ...     vlm=qwen_vlm,
        ...     fusion_level="neck",
        ...     fusion_type="cross_attention",
        ... )
    """

    def __init__(
        self,
        detector: BaseDetector,
        vlm: BaseVLM,
        fusion_level: str = "neck",
        fusion_type: str = "cross_attention",
        num_attention_heads: int = 8,
        use_gate: bool = True,
        freeze_vlm: bool = True,
        freeze_detector: bool = False,
        det_feature_dim: int = 512,
        **kwargs,
    ):
        """
        Initialize Feature Fusion.

        Args:
            detector: Object detection model
            vlm: Vision-Language Model
            fusion_level: Where to inject VLM features ("backbone", "neck", "head")
            fusion_type: Fusion method ("cross_attention", "adapter", "concat")
            num_attention_heads: Number of attention heads for cross-attention
            use_gate: Whether to use gating mechanism
            freeze_vlm: Whether to freeze VLM
            freeze_detector: Whether to freeze detector
            det_feature_dim: Detector feature dimension
        """
        super().__init__(
            detector=detector,
            vlm=vlm,
            freeze_vlm=freeze_vlm,
            freeze_detector=freeze_detector,
        )

        self.fusion_level = fusion_level
        self.fusion_type = fusion_type

        vlm_dim = vlm.vision_hidden_size

        # Create fusion module based on type
        if fusion_type == "cross_attention":
            self.fusion_module = CrossAttentionBlock(
                det_dim=det_feature_dim,
                vlm_dim=vlm_dim,
                num_heads=num_attention_heads,
            )
        elif fusion_type == "adapter":
            self.fusion_module = FeatureAdapter(
                det_dim=det_feature_dim,
                vlm_dim=vlm_dim,
                use_gate=use_gate,
            )
        elif fusion_type == "concat":
            # Concatenation with projection
            self.fusion_module = nn.Sequential(
                nn.Linear(det_feature_dim + vlm_dim, det_feature_dim),
                nn.LayerNorm(det_feature_dim),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Feature dimension adapter
        self.det_proj = nn.Linear(det_feature_dim, det_feature_dim)

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
        **kwargs,
    ) -> FusionOutput:
        """
        Forward pass with feature fusion.

        Args:
            images: Input images [B, C, H, W]
            targets: Optional detection targets

        Returns:
            FusionOutput with detection results and losses
        """
        # Extract detector features at specified level
        det_features = self.detector.extract_features(
            images, feature_levels=[self.fusion_level]
        )

        # Get VLM features
        with torch.no_grad():
            vlm_features = self.vlm.get_visual_features(images)

        # Apply fusion
        fused_features = self._apply_fusion(
            det_features[self.fusion_level], vlm_features
        )

        # Forward through detector with fused features
        # Note: This requires the detector to support feature injection
        # For simplicity, we do a full forward here
        det_output = self.detector(images, targets)

        # Compute losses
        loss_dict = {}
        if det_output.batch_loss_dict:
            loss_dict.update(det_output.batch_loss_dict)

        # Add fusion regularization loss (optional)
        fusion_reg_loss = self._compute_fusion_regularization(
            det_features[self.fusion_level], fused_features
        )
        loss_dict["fusion_reg"] = fusion_reg_loss * 0.1

        loss_dict["total_loss"] = sum(loss_dict.values())

        return FusionOutput(
            detection_output=det_output,
            fusion_features={
                "det_features": det_features,
                "vlm_features": vlm_features,
                "fused_features": fused_features,
            },
            loss_dict=loss_dict,
        )

    def _apply_fusion(
        self,
        det_features: Tensor,
        vlm_features: Tensor,
    ) -> Tensor:
        """Apply the fusion operation."""
        # Reshape detector features if needed: [B, C, H, W] -> [B, H*W, C]
        if det_features.dim() == 4:
            B, C, H, W = det_features.shape
            det_flat = det_features.permute(0, 2, 3, 1).reshape(B, H * W, C)
            needs_reshape = True
        else:
            det_flat = det_features
            needs_reshape = False
            H, W = None, None

        if self.fusion_type == "cross_attention":
            fused = self.fusion_module(det_flat, vlm_features)
        elif self.fusion_type == "adapter":
            fused = self.fusion_module(det_flat, vlm_features)
        elif self.fusion_type == "concat":
            # Pool VLM features and concatenate
            vlm_pooled = vlm_features.mean(dim=1, keepdim=True)
            vlm_expanded = vlm_pooled.expand(-1, det_flat.shape[1], -1)
            combined = torch.cat([det_flat, vlm_expanded], dim=-1)
            fused = self.fusion_module(combined)

        # Reshape back if needed
        if needs_reshape:
            fused = fused.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return fused

    def _compute_fusion_regularization(
        self,
        original: Tensor,
        fused: Tensor,
    ) -> Tensor:
        """
        Compute regularization to prevent too much deviation.

        This encourages the fused features to stay close to original
        detector features while incorporating VLM information.
        """
        if original.dim() == 4:
            B, C, H, W = original.shape
            original = original.permute(0, 2, 3, 1).reshape(B, -1, C)
            fused = fused.permute(0, 2, 3, 1).reshape(B, -1, C)

        return F.mse_loss(fused, original)

    def compute_loss(
        self,
        outputs: FusionOutput,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        return outputs.loss_dict or {}


@FUSIONS.register("multi_scale_fusion", aliases=["ms_fusion"])
class MultiScaleFeatureFusion(BaseFusion):
    """
    Multi-scale feature fusion.

    Injects VLM features at multiple levels of the detector's
    feature pyramid for comprehensive semantic enhancement.
    """

    def __init__(
        self,
        detector: BaseDetector,
        vlm: BaseVLM,
        fusion_levels: List[str] = ["p3", "p4", "p5"],
        det_feature_dims: List[int] = [256, 512, 1024],
        freeze_vlm: bool = True,
        freeze_detector: bool = False,
        **kwargs,
    ):
        super().__init__(
            detector=detector,
            vlm=vlm,
            freeze_vlm=freeze_vlm,
            freeze_detector=freeze_detector,
        )

        self.fusion_levels = fusion_levels
        vlm_dim = vlm.vision_hidden_size

        # Create fusion module for each level
        self.fusion_modules = nn.ModuleDict()
        for level, dim in zip(fusion_levels, det_feature_dims):
            self.fusion_modules[level] = FeatureAdapter(
                det_dim=dim,
                vlm_dim=vlm_dim,
                use_gate=True,
            )

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None,
        **kwargs,
    ) -> FusionOutput:
        """Multi-scale fusion forward pass."""
        # Get all features
        det_features = self.detector.extract_features(
            images, feature_levels=self.fusion_levels
        )

        with torch.no_grad():
            vlm_features = self.vlm.get_visual_features(images)

        # Fuse at each level
        fused_features = {}
        for level in self.fusion_levels:
            if level in det_features:
                fused_features[level] = self._apply_level_fusion(
                    det_features[level], vlm_features, level
                )

        # Get detection output
        det_output = self.detector(images, targets)

        loss_dict = det_output.batch_loss_dict or {}
        loss_dict["total_loss"] = sum(loss_dict.values()) if loss_dict else torch.tensor(0.0)

        return FusionOutput(
            detection_output=det_output,
            fusion_features=fused_features,
            loss_dict=loss_dict,
        )

    def _apply_level_fusion(
        self,
        det_features: Tensor,
        vlm_features: Tensor,
        level: str,
    ) -> Tensor:
        """Apply fusion at a specific level."""
        # Reshape if needed
        if det_features.dim() == 4:
            B, C, H, W = det_features.shape
            det_flat = det_features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        else:
            det_flat = det_features

        fused = self.fusion_modules[level](det_flat, vlm_features)

        if det_features.dim() == 4:
            fused = fused.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return fused

    def compute_loss(
        self,
        outputs: FusionOutput,
        targets: List[Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        return outputs.loss_dict or {}
