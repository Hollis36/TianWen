"""Tests for loss functions in tianwen/engine/losses.py."""

import pytest
import torch
import torch.nn.functional as F

from tianwen.engine.losses import (
    CombinedDetectionLoss,
    DistillationLoss,
    FeatureAlignmentLoss,
    FocalLoss,
)


class TestDistillationLoss:
    """Tests for DistillationLoss."""

    def test_distillation_loss_soft_only(self):
        """Without hard targets the loss equals the soft KL term."""
        loss_fn = DistillationLoss(temperature=4.0, alpha=0.5)
        student = torch.randn(4, 10)
        teacher = torch.randn(4, 10)

        losses = loss_fn(student, teacher)

        assert "soft_loss" in losses
        assert "total_loss" in losses
        assert "hard_loss" not in losses
        # total_loss should equal soft_loss when no hard targets
        assert torch.allclose(losses["total_loss"], losses["soft_loss"])

    def test_distillation_loss_with_hard_targets(self):
        """With hard targets the total loss is alpha-weighted combination."""
        alpha = 0.7
        loss_fn = DistillationLoss(temperature=2.0, alpha=alpha)
        student = torch.randn(4, 10)
        teacher = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        losses = loss_fn(student, teacher, targets)

        assert "soft_loss" in losses
        assert "hard_loss" in losses
        assert "total_loss" in losses

        expected_total = alpha * losses["soft_loss"] + (1 - alpha) * losses["hard_loss"]
        assert torch.allclose(losses["total_loss"], expected_total)

    def test_distillation_loss_temperature_scaling(self):
        """KL term should scale with T^2."""
        student = torch.randn(4, 10)
        teacher = torch.randn(4, 10)

        T = 4.0
        loss_fn = DistillationLoss(temperature=T, alpha=1.0)
        losses = loss_fn(student, teacher)

        # Manually compute soft loss at T=1 and verify scaling
        raw_kl = F.kl_div(
            F.log_softmax(student / T, dim=-1),
            F.softmax(teacher / T, dim=-1),
            reduction="batchmean",
        )
        expected = raw_kl * (T**2)
        assert torch.allclose(losses["soft_loss"], expected, atol=1e-5)


class TestFeatureAlignmentLoss:
    """Tests for FeatureAlignmentLoss."""

    def test_feature_alignment_loss_mse(self):
        """MSE loss between identical features should be zero."""
        loss_fn = FeatureAlignmentLoss(loss_type="mse", normalize=False)
        feat = torch.randn(4, 64)
        result = loss_fn(feat, feat.clone())
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_feature_alignment_loss_cosine(self):
        """Cosine loss between identical (non-zero) features should be zero."""
        loss_fn = FeatureAlignmentLoss(loss_type="cosine", normalize=True)
        feat = torch.randn(4, 64)
        result = loss_fn(feat, feat.clone())
        assert result.item() == pytest.approx(0.0, abs=1e-5)

    def test_feature_alignment_loss_l1(self):
        """L1 loss between identical features should be zero."""
        loss_fn = FeatureAlignmentLoss(loss_type="l1", normalize=False)
        feat = torch.randn(4, 64)
        result = loss_fn(feat, feat.clone())
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_feature_alignment_loss_normalize(self):
        """When normalize=True features are L2-normalized before loss."""
        loss_fn = FeatureAlignmentLoss(loss_type="mse", normalize=True)
        feat = torch.randn(4, 64) * 100  # large scale
        result = loss_fn(feat, feat.clone())
        # After normalization, MSE of same vector should still be ~0
        assert result.item() == pytest.approx(0.0, abs=1e-5)

    def test_feature_alignment_loss_invalid_type(self):
        with pytest.raises(ValueError, match="Unknown loss type"):
            FeatureAlignmentLoss(loss_type="unknown")


class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_focal_loss_basic(self):
        """Focal loss should be non-negative and finite."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        preds = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        result = loss_fn(preds, targets)
        assert result.item() >= 0.0
        assert torch.isfinite(result)

    def test_focal_loss_reduction_mean(self):
        """Mean reduction returns a scalar."""
        loss_fn = FocalLoss(reduction="mean")
        preds = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        result = loss_fn(preds, targets)
        assert result.shape == torch.Size([])

    def test_focal_loss_reduction_sum(self):
        """Sum reduction returns a scalar."""
        loss_fn = FocalLoss(reduction="sum")
        preds = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        result = loss_fn(preds, targets)
        assert result.shape == torch.Size([])

    def test_focal_loss_reduction_none(self):
        """'none' reduction returns per-sample loss."""
        loss_fn = FocalLoss(reduction="none")
        preds = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        result = loss_fn(preds, targets)
        assert result.shape == torch.Size([8])


class TestCombinedDetectionLoss:
    """Tests for CombinedDetectionLoss."""

    def _make_inputs(self, batch=4, num_classes=10, use_focal=True):
        cls_preds = torch.randn(batch, num_classes)
        box_preds = torch.randn(batch, 4)
        obj_preds = torch.randn(batch, 1)
        cls_targets = torch.randint(0, num_classes, (batch,))
        box_targets = torch.randn(batch, 4)
        obj_targets = torch.randint(0, 2, (batch, 1)).float()
        return cls_preds, box_preds, obj_preds, cls_targets, box_targets, obj_targets

    def test_combined_detection_loss(self):
        """All expected keys present and total_loss is positive."""
        loss_fn = CombinedDetectionLoss()
        cls_p, box_p, obj_p, cls_t, box_t, obj_t = self._make_inputs()
        losses = loss_fn(cls_p, box_p, obj_p, cls_t, box_t, obj_t)

        assert "cls_loss" in losses
        assert "box_loss" in losses
        assert "obj_loss" in losses
        assert "total_loss" in losses
        assert losses["total_loss"].item() > 0.0

    def test_combined_detection_loss_without_obj(self):
        """Without objectness preds/targets, obj_loss should not appear."""
        loss_fn = CombinedDetectionLoss()
        cls_p, box_p, _, cls_t, box_t, _ = self._make_inputs()
        losses = loss_fn(cls_p, box_p, None, cls_t, box_t, None)

        assert "cls_loss" in losses
        assert "box_loss" in losses
        assert "obj_loss" not in losses
        assert "total_loss" in losses
