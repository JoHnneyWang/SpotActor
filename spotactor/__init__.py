"""SpotActor: Training-Free Layout-Controlled Consistent Image Generation.

A training-free pipeline for layout-guided consistent multi-object image
generation, featuring dual energy guidance in a semantic-latent space.
Published at AAAI 2025.

Key Features:
- Multi-object layout guidance via attention manipulation
- Cross-scene identity consistency through Regional Interconnection Self-Attention (RISA)
- Layout control via Semantic Fusion Cross-Attention (SFCA)
- Two-stage generation protocol (source → target)
- Compatible with Stable Diffusion XL

Example:
    >>> from spotactor import SpotActorXLPipeline
    >>> pipe = SpotActorXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
    >>> pipe = pipe.to("cuda")
"""

__version__ = "1.0.0"

from .pipeline import SpotActorXLPipeline
from .attention import Manipulator, SpotActorAttnProcessor
from .utils import AdaptiveScheduler

__all__ = ["SpotActorXLPipeline", "Manipulator", "SpotActorAttnProcessor", "AdaptiveScheduler"]
