"""Tests for the registry system."""

import pytest
import torch

from tianwen.core.registry import Registry, DETECTORS, VLMS, FUSIONS


class TestRegistry:
    """Test Registry class."""

    def test_register_decorator(self):
        """Test registering a class with decorator."""
        reg = Registry("test")

        @reg.register("test_class")
        class TestClass:
            pass

        assert "test_class" in reg
        assert reg.get("test_class") is TestClass

    def test_register_with_alias(self):
        """Test registering with aliases."""
        reg = Registry("test")

        @reg.register("main_name", aliases=["alias1", "alias2"])
        class TestClass:
            pass

        assert reg.get("main_name") is TestClass
        assert reg.get("alias1") is TestClass
        assert reg.get("alias2") is TestClass

    def test_build_from_config(self):
        """Test building instance from config."""
        reg = Registry("test")

        @reg.register("buildable")
        class Buildable:
            def __init__(self, value=10):
                self.value = value

        instance = reg.build({"type": "buildable", "value": 42})
        assert instance.value == 42

    def test_build_unknown_type(self):
        """Test building unknown type raises error."""
        reg = Registry("test")

        with pytest.raises(KeyError):
            reg.build({"type": "unknown"})

    def test_list_available(self):
        """Test listing available modules."""
        reg = Registry("test")

        @reg.register("a")
        class A:
            pass

        @reg.register("b")
        class B:
            pass

        available = reg.list_available()
        assert "a" in available
        assert "b" in available


class TestGlobalRegistries:
    """Test global registry instances."""

    def test_detectors_registry_exists(self):
        """Test DETECTORS registry is available."""
        assert DETECTORS.name == "detectors"

    def test_vlms_registry_exists(self):
        """Test VLMS registry is available."""
        assert VLMS.name == "vlms"

    def test_fusions_registry_exists(self):
        """Test FUSIONS registry is available."""
        assert FUSIONS.name == "fusions"


class TestRegisteredComponents:
    """Test that components are properly registered."""

    def test_yolo_registered(self):
        """Test YOLO detector is registered."""
        # Import to trigger registration
        from tianwen.detectors import yolo

        assert "yolov8" in DETECTORS
        assert "yolo" in DETECTORS  # alias

    def test_qwen_vl_registered(self):
        """Test Qwen-VL is registered."""
        from tianwen.vlms import qwen_vl

        assert "qwen_vl" in VLMS
        assert "qwen-vl" in VLMS  # alias

    def test_distillation_registered(self):
        """Test distillation fusion is registered."""
        from tianwen.fusions import distillation

        assert "distillation" in FUSIONS
        assert "kd" in FUSIONS  # alias
