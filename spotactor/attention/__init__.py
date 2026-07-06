"""Attention manipulation modules for SpotActor.

This package provides the core attention-level components:
- Manipulator: manages attention mode switching and spatial guidance masks
- AttnProcessor: custom attention processor that routes through the Manipulator
"""

from .manipulator import Manipulator
from .processor import SpotActorAttnProcessor

__all__ = ["Manipulator", "SpotActorAttnProcessor"]
