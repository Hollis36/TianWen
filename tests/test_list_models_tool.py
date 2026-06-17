"""Tests for the list_models tool."""

from tools import list_models


def test_list_models_main_outputs_registered_components(capsys):
    """Test list_models prints registered component groups."""
    list_models.main()

    captured = capsys.readouterr()
    output = captured.out

    assert "Detectors" in output
    assert "Vision-Language Models" in output
    assert "Fusion strategies" in output
    assert "Datasets" in output
    assert "yolov8" in output
    assert "qwen_vl" in output
    assert "decision_fusion" in output
    assert "coco" in output
