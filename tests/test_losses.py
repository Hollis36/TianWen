"""Tests for loss functions in tianwen.engine.losses."""

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

    def test_soft_loss_only(self):
        """Without hard targets, only soft_loss and total_loss are returned."""
        loss_fn = DistillationLoss(temperature=4.0, alpha=0.5)
        student = torch.randn(4, 10)
        teacher = torch.randn(4, 10)

        losses = loss_fn(student, teacher)

        assert "soft_loss" in losses
        assert "total_loss" in losses
        assert "hard_loss" not in losses
        assert losses["total_loss"].item() >= 0

    def test_soft_and_hard_loss(self):
        """With hard targets all three loss keys are present."""
        loss_fn = DistillationLoss(temperature=4.0, alpha=0.5)
        student = torch.randn(4, 10)
        teacher = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))

        losses = loss_fn(student, teacher, targets)

        assert "soft_loss" in losses
        assert "hard_loss" in losses
        assert "total_loss" in losses

    def test_total_is_weighted_combination(self):
        """total_loss == alpha*soft + (1-alpha)*hard."""
        alpha = 0.3
        loss_fn = DistillationLoss(temperature=2.0, alpha=alpha)
        torch.manual_seed(0)
        student = torch.randn(4, 5)
        teacher = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))

        losses = loss_fn(student, teacher, targets)

        expected = alpha * losses["soft_loss"] + (1 - alpha) * losses["hard_loss"]
        assert torch.allclose(losses["total_loss"], expected, atol=1e-6)

    def test_identical_student_teacher_low_soft_loss(self):
        """When student and teacher logits are identical the soft loss is near 0."""
        loss_fn = DistillationLoss(temperature=4.0)
        logits = torch.randn(4, 10)

        losses = loss_fn(logits, logits)
        assert losses["soft_loss"].item() < 1e-4

    def test_temperature_scaling(self):
        """Higher temperature should reduce the magnitude of soft loss gradients."""
        torch.manual_seed(42)
        student = torch.randn(4, 10, requires_grad=True)
        teacher = torch.randn(4, 10)

        loss_low_T = DistillationLoss(temperature=1.0)(student, teacher)["soft_loss"]
        loss_high_T = DistillationLoss(temperature=10.0)(student, teacher)["soft_loss"]

        # Both should be non-negative
        assert loss_low_T.item() >= 0
        assert loss_high_T.item() >= 0


class TestFeatureAlignmentLoss:
    """Tests for FeatureAlignmentLoss."""

    @pytest.mark.parametrize("loss_type", ["mse", "l1", "cosine"])
    def test_forward_returns_scalar(self, loss_type):
        """All loss types should return a scalar tensor."""
        loss_fn = FeatureAlignmentLoss(loss_type=loss_type)
        student = torch.randn(4, 256)
        teacher = torch.randn(4, 256)

        loss = loss_fn(student, teacher)
        assert loss.shape == ()

    def test_identical_features_near_zero_mse(self):
        """MSE between identical features should be (near) zero."""
        loss_fn = FeatureAlignmentLoss(loss_type="mse", normalize=True)
        feat = torch.randn(4, 128)

        loss = loss_fn(feat, feat)
        assert loss.item() < 1e-5

    def test_identical_features_near_zero_cosine(self):
        """Cosine loss between identical features should be (near) zero."""
        loss_fn = FeatureAlignmentLoss(loss_type="cosine", normalize=True)
        feat = torch.randn(4, 128)

        loss = loss_fn(feat, feat)
        assert loss.item() < 1e-5

    def test_unknown_loss_type_raises(self):
        """Unsupported loss types should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            FeatureAlignmentLoss(loss_type="unknown")

    def test_normalization_flag(self):
        """normalize=False should also produce a valid scalar."""
        loss_fn = FeatureAlignmentLoss(loss_type="mse", normalize=False)
        loss = loss_fn(torch.randn(4, 64), torch.randn(4, 64))
        assert loss.shape == ()


class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_forward_returns_scalar(self):
        """Forward pass should return a scalar loss."""
        loss_fn = FocalLoss()
        preds = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))

        loss = loss_fn(preds, targets)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_high_confidence_correct_lower_loss(self):
        """High confidence on correct class should give lower focal loss than random."""
        loss_fn = FocalLoss(alpha=1.0, gamma=2.0)

        # High-confidence predictions
        preds_good = torch.zeros(4, 3)
        preds_good[:, 0] = 10.0  # very confident on class 0
        targets = torch.zeros(4, dtype=torch.long)  # class 0 is correct

        # Low-confidence / random predictions
        preds_bad = torch.randn(4, 3)

        loss_good = loss_fn(preds_good, targets).item()
        loss_bad = loss_fn(preds_bad, targets).item()

        assert loss_good < loss_bad

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_reduction_modes(self, reduction):
        """All reduction modes should run without error."""
        loss_fn = FocalLoss(reduction=reduction)
        preds = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))

        loss = loss_fn(preds, targets)
        if reduction == "none":
            assert loss.shape == (4,)
        else:
            assert loss.shape == ()


class TestCombinedDetectionLoss:
    """Tests for CombinedDetectionLoss."""

    def _make_inputs(self, n=8, num_classes=5):
        cls_preds = torch.randn(n, num_classes)
        box_preds = torch.randn(n, 4)
        cls_targets = torch.randint(0, num_classes, (n,))
        box_targets = torch.randn(n, 4)
        return cls_preds, box_preds, cls_targets, box_targets

    def test_output_keys_without_objectness(self):
        """Without objectness inputs, loss dict should have cls_loss, box_loss, total_loss."""
        loss_fn = CombinedDetectionLoss(use_focal=False)
        cls_preds, box_preds, cls_targets, box_targets = self._make_inputs()

        losses = loss_fn(cls_preds, box_preds, None, cls_targets, box_targets, None)

        assert "cls_loss" in losses
        assert "box_loss" in losses
        assert "total_loss" in losses
        assert "obj_loss" not in losses

    def test_output_keys_with_objectness(self):
        """With objectness inputs, obj_loss should also be present."""
        loss_fn = CombinedDetectionLoss(use_focal=True)
        cls_preds, box_preds, cls_targets, box_targets = self._make_inputs()
        obj_preds = torch.randn(8)
        obj_targets = torch.randint(0, 2, (8,)).float()

        losses = loss_fn(
            cls_preds, box_preds, obj_preds, cls_targets, box_targets, obj_targets
        )

        assert "obj_loss" in losses

    def test_total_is_sum_of_components(self):
        """total_loss should equal the weighted sum of component losses."""
        loss_fn = CombinedDetectionLoss(
            cls_weight=2.0, box_weight=3.0, use_focal=False
        )
        cls_preds, box_preds, cls_targets, box_targets = self._make_inputs()

        losses = loss_fn(cls_preds, box_preds, None, cls_targets, box_targets, None)

        expected = losses["cls_loss"] + losses["box_loss"]
        assert torch.allclose(losses["total_loss"], expected, atol=1e-6)

    def test_all_losses_non_negative(self):
        """All returned losses should be non-negative."""
        loss_fn = CombinedDetectionLoss()
        cls_preds, box_preds, cls_targets, box_targets = self._make_inputs()

        losses = loss_fn(cls_preds, box_preds, None, cls_targets, box_targets, None)

        for name, val in losses.items():
            assert val.item() >= 0, f"{name} was negative"
