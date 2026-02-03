"""Vision-Language Models for TianWen framework."""

from tianwen.vlms.base import BaseVLM, VLMOutput


# Import concrete implementations to trigger registration
def _register_vlms():
    from tianwen.vlms import qwen_vl  # noqa: F401
    from tianwen.vlms import internvl  # noqa: F401

_register_vlms()

from tianwen.vlms.qwen_vl import QwenVLModel  # noqa: E402
from tianwen.vlms.internvl import InternVLModel  # noqa: E402

__all__ = [
    "BaseVLM",
    "VLMOutput",
    "QwenVLModel",
    "InternVLModel",
]
