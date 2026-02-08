"""Tests for performance improvements."""

import pytest
import numpy as np
import torch

from tianwen.utils.metrics import compute_iou, compute_ap


class TestComputeIoU:
    """Test vectorized IoU computation."""

    def test_identical_boxes(self):
        """Identical boxes should have IoU of 1.0."""
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        iou = compute_iou(boxes, boxes)
        assert iou.shape == (1, 1)
        assert torch.isclose(iou[0, 0], torch.tensor(1.0), atol=1e-5)

    def test_no_overlap(self):
        """Non-overlapping boxes should have IoU of 0.0."""
        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        boxes2 = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
        iou = compute_iou(boxes1, boxes2)
        assert torch.isclose(iou[0, 0], torch.tensor(0.0), atol=1e-5)

    def test_partial_overlap(self):
        """Partially overlapping boxes should have IoU between 0 and 1."""
        boxes1 = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
        boxes2 = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
        iou = compute_iou(boxes1, boxes2)
        assert 0 < iou[0, 0].item() < 1

    def test_matrix_shape(self):
        """IoU should return [N, M] matrix."""
        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                                [5.0, 5.0, 15.0, 15.0]])
        boxes2 = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                                [10.0, 10.0, 20.0, 20.0],
                                [20.0, 20.0, 30.0, 30.0]])
        iou = compute_iou(boxes1, boxes2)
        assert iou.shape == (2, 3)


class TestComputeAP:
    """Test AP computation."""

    def test_perfect_precision_recall(self):
        """Perfect classifier should have AP close to 1.0."""
        recalls = np.array([0.0, 0.5, 1.0])
        precisions = np.array([1.0, 1.0, 1.0])
        ap = compute_ap(recalls, precisions)
        assert ap > 0.95

    def test_zero_ap(self):
        """All wrong predictions should have AP close to 0."""
        recalls = np.array([0.0])
        precisions = np.array([0.0])
        ap = compute_ap(recalls, precisions)
        assert ap < 0.1


class TestBenchmarkComputeMetrics:
    """Test vectorized compute_metrics from benchmark.py."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield precision=1, recall=1."""
        from tools.benchmark import compute_metrics

        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        labels = torch.tensor([0])
        scores = torch.tensor([0.9])

        predictions = [{"boxes": boxes, "labels": labels, "scores": scores}]
        targets = [{"boxes": boxes, "labels": labels}]

        metrics = compute_metrics(predictions, targets)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] > 0.99

    def test_no_predictions(self):
        """No predictions should give precision=0, recall=0."""
        from tools.benchmark import compute_metrics

        predictions = [{"boxes": torch.zeros((0, 4)),
                         "labels": torch.zeros(0, dtype=torch.long),
                         "scores": torch.zeros(0)}]
        targets = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                     "labels": torch.tensor([0])}]

        metrics = compute_metrics(predictions, targets)
        assert metrics["recall"] == 0.0
        assert metrics["fn"] == 1

    def test_wrong_label_not_matched(self):
        """Prediction with wrong label should not match ground truth."""
        from tools.benchmark import compute_metrics

        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        predictions = [{"boxes": boxes, "labels": torch.tensor([1]),
                         "scores": torch.tensor([0.9])}]
        targets = [{"boxes": boxes, "labels": torch.tensor([0])}]

        metrics = compute_metrics(predictions, targets)
        assert metrics["tp"] == 0
        assert metrics["fp"] == 1
        assert metrics["fn"] == 1


class TestCocoBenchmarkIoUMatrix:
    """Test vectorized IoU matrix computation in coco_benchmark.py."""

    def test_iou_matrix_shape(self):
        """compute_iou_matrix should return correct shape."""
        import sys
        sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
        from tools.experiments.coco_benchmark import compute_iou_matrix

        boxes1 = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float64)
        boxes2 = np.array([[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]], dtype=np.float64)

        result = compute_iou_matrix(boxes1, boxes2)
        assert result.shape == (2, 3)

    def test_iou_matrix_identical(self):
        """Identical box should yield IoU=1.0."""
        from tools.experiments.coco_benchmark import compute_iou_matrix

        boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
        result = compute_iou_matrix(boxes, boxes)
        assert np.isclose(result[0, 0], 1.0, atol=1e-5)

    def test_iou_matrix_matches_scalar(self):
        """Vectorized IoU should match scalar IoU computation."""
        from tools.experiments.coco_benchmark import compute_iou_matrix, compute_iou

        boxes1 = np.array([[0, 0, 20, 20], [5, 5, 25, 25]], dtype=np.float64)
        boxes2 = np.array([[10, 10, 30, 30], [0, 0, 15, 15]], dtype=np.float64)

        matrix = compute_iou_matrix(boxes1, boxes2)
        for i in range(len(boxes1)):
            for j in range(len(boxes2)):
                scalar = compute_iou(boxes1[i], boxes2[j])
                assert np.isclose(matrix[i, j], scalar, atol=1e-5), \
                    f"Mismatch at [{i},{j}]: matrix={matrix[i,j]}, scalar={scalar}"


class TestCocoBenchmarkComputeAP:
    """Test vectorized AP computation in coco_benchmark.py."""

    def test_ap_perfect(self):
        """Perfect classifier should yield high AP."""
        from tools.experiments.coco_benchmark import compute_ap

        recalls = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        precisions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        ap = compute_ap(recalls, precisions)
        assert ap > 0.95

    def test_ap_decreasing_precision(self):
        """AP should handle decreasing precision correctly."""
        from tools.experiments.coco_benchmark import compute_ap

        recalls = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        precisions = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        ap = compute_ap(recalls, precisions)
        assert 0 < ap < 1


class TestColorJitterCaching:
    """Test that ColorJitter transform is cached."""

    def test_cached_transform_exists(self):
        """ColorJitter should have _transform attribute after init."""
        from tianwen.datasets.transforms import ColorJitter
        jitter = ColorJitter(brightness=0.2, contrast=0.2)
        assert hasattr(jitter, '_transform')

    def test_same_transform_reused(self):
        """The same _transform instance should be reused across calls."""
        from tianwen.datasets.transforms import ColorJitter
        from PIL import Image

        jitter = ColorJitter(brightness=0.2, contrast=0.2)
        transform_id = id(jitter._transform)

        # Call it
        img = Image.new('RGB', (32, 32), color=(128, 128, 128))
        boxes = torch.zeros((0, 4))
        labels = torch.zeros(0, dtype=torch.long)
        jitter(img, boxes, labels)

        # Should still be same object
        assert id(jitter._transform) == transform_id


class TestDecisionFusionVectorized:
    """Test vectorized score fusion in DecisionFusion."""

    def test_parse_vlm_response_binary(self):
        """Test VLM response parsing."""
        from tianwen.fusions.decision_fusion import DecisionFusion
        from unittest.mock import MagicMock

        # Create a mock detector and vlm
        mock_detector = MagicMock()
        mock_vlm = MagicMock()
        mock_vlm.vision_hidden_size = 768

        fusion = DecisionFusion.__new__(DecisionFusion)
        fusion.verification_mode = "binary"

        assert fusion._parse_vlm_response("yes, it is") == 1.0
        assert fusion._parse_vlm_response("no, it is not") == 0.0
        assert fusion._parse_vlm_response("maybe") == 0.5

    def test_parse_vlm_response_confidence(self):
        """Test VLM confidence parsing."""
        from tianwen.fusions.decision_fusion import DecisionFusion

        fusion = DecisionFusion.__new__(DecisionFusion)
        fusion.verification_mode = "confidence"

        assert abs(fusion._parse_vlm_response("85") - 0.85) < 0.01
        assert abs(fusion._parse_vlm_response("I'd say 95%") - 0.95) < 0.01
