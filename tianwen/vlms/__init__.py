"""Vision-Language Models for TianWen framework."""

from tianwen.vlms.base import BaseVLM, VLMOutput

__all__ = ["BaseVLM", "VLMOutput"]

# Import concrete implementations to trigger registration
def _register_vlms():
    from tianwen.vlms import qwen_vl
    from tianwen.vlms import internvl

_register_vlms()
