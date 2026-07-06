"""Custom Attention Processor for SpotActor.

Intercepts attention computation in the UNet and routes it through
the Manipulator for layout/consistency guidance.
"""

import torch
from diffusers.models.attention_processor import Attention

from .manipulator import Manipulator


class SpotActorAttnProcessor:
    """Attention processor that integrates with SpotActor's Manipulator.

    Replaces the default attention processor in the UNet to enable:
    - Cross-attention prompt embedding replacement (CF mode)
    - Attention map storage for backward layout guidance (B mode)
    - Cross-batch self-attention for consistency (F mode)

    Args:
        manipulator: The Manipulator instance controlling attention behavior.
        place_in_unet: Position identifier ("down", "mid", or "up").
    """

    def __init__(self, manipulator: Manipulator, place_in_unet: str):
        super().__init__()
        self.manipulator = manipulator
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Process attention with optional manipulation.

        Args:
            attn: The Attention module.
            hidden_states: Input hidden states [batch, seq_len, dim].
            encoder_hidden_states: Conditioning embeddings (None for self-attn).
            attention_mask: Optional attention mask.

        Returns:
            Processed hidden states.
        """
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None

        # In CF (Consistency Forward) mode, replace cross-attention embeddings
        # with blended prompt embeddings for identity consistency
        if is_cross and "CF" in self.manipulator.mode:
            encoder_hidden_states = self.manipulator.new_prompt_embeds

        encoder_hidden_states = (
            encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Route through manipulator
        hidden_states = self.manipulator.attention_process_forward(
            attn, query, key, value, attention_mask, is_cross
        )

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)  # Linear projection
        hidden_states = attn.to_out[1](hidden_states)  # Dropout

        return hidden_states
