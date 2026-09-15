"""Agent harness implementations and their small lazy factories."""

from .factory import create_harness, normalize_harness_name
from .identity import HARNESS_DESCRIPTORS, HarnessDescriptor, get_harness_descriptor

__all__ = [
    "HARNESS_DESCRIPTORS",
    "HarnessDescriptor",
    "create_harness",
    "get_harness_descriptor",
    "normalize_harness_name",
]
