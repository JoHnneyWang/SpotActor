"""Attention Manipulator for SpotActor pipeline.

The Manipulator orchestrates attention-level operations during the diffusion
denoising loop, including:
- Layout backward guidance (loss-based spatial control)
- Layout forward guidance (mask-based attention re-weighting)
- Consistency forward guidance (cross-batch self-attention sharing)
- Automatic mask generation via Otsu binarization
"""

from typing import List, Optional

import numpy as np
import torch
from PIL import Image


class Manipulator:
    """Central controller for attention manipulation in SpotActor.

    Manages mode switching between different guidance strategies and maintains
    the spatial masks used for layout and consistency control.

    Modes:
        D   - Default (no manipulation)
        LB  - Layout Backward only
        LCB - Layout + Consistency Backward
        LF  - Layout Forward only
        CF  - Consistency Forward only
        LCF - Layout + Consistency Forward

    Args:
        batch_size: Number of scenes generated simultaneously.
        obj_indices: Per-batch list of token indices for each object.
        sce_indices: Per-batch list of token indices for scene tokens.
        all_indices: Per-batch list of all token index ranges.
        bboxes: Per-batch list of normalized bounding boxes [x_min, y_min, x_max, y_max].
        is_source: Whether this is source-stage generation (skips mask init).
        dtype: Tensor data type.
        device: Computation device.
    """

    # Attention layer position ranges (SDXL UNet architecture)
    _LAYER_RANGES = {
        "down4096": (0, 9),
        "down1024": (9, 49),
        "mid1024": (49, 69),
        "up1024": (69, 129),
        "up4096": (129, 141),
    }

    def __init__(
        self,
        batch_size: int = 2,
        obj_indices: Optional[List[List[int]]] = None,
        sce_indices: Optional[List[List[int]]] = None,
        all_indices: Optional[List] = None,
        bboxes: Optional[List[List[List[float]]]] = None,
        is_source: bool = False,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.mode = "D"
        self.batch_size = batch_size
        self.obj_num = len(obj_indices[0]) if obj_indices else 0
        self.dtype = dtype
        self.device = device

        # Configurable layer selections
        self.layout_selected_layer = ["up1024"]
        self.consistent_selected_layer = ["down4096", "down1024", "mid1024", "up1024", "up4096"]
        self.automask_selected_layer = ["up1024"]

        # Attention layer tracking
        self.num_att_layers = -1  # Set by register_attention_control
        self.cur_att_layer = 0

        # Storage for attention maps
        self.A_CA_store: List[torch.Tensor] = []
        self.A_SA_store: List[torch.Tensor] = []
        self.A_CA_accumulate_automask = torch.zeros(
            (batch_size * 20 * 2, 1024, 77), dtype=dtype, device=device
        )

        # Consistency masks (initialized as fully-open)
        self.masks_1024 = torch.ones((batch_size, 1024 * batch_size), dtype=dtype, device=device)
        self.masks_4096 = torch.ones((batch_size, 4096 * batch_size), dtype=dtype, device=device)

        # Object/scene indices
        self.obj_indices = obj_indices
        self.sce_indices = sce_indices
        self.all_indices = all_indices
        self.is_source = is_source

        # Initialize spatial masks (only for target stage)
        if not is_source and bboxes is not None:
            self._build_sdsa_layout_masks(bboxes)
            self._build_forward_layout_masks(bboxes)

    # ------------------------------------------------------------------
    # Layer position detection
    # ------------------------------------------------------------------

    def check_position(self) -> str:
        """Determine which UNet block the current attention layer belongs to."""
        for name, (start, end) in self._LAYER_RANGES.items():
            if self.cur_att_layer < end:
                return name
        return "default_process"

    # ------------------------------------------------------------------
    # Core attention routing
    # ------------------------------------------------------------------

    def attention_process_forward(
        self,
        attn,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_cross: bool,
    ) -> torch.Tensor:
        """Route attention computation based on current mode.

        Args:
            attn: The Attention module instance.
            q: Query tensor [heads, res, dim].
            k: Key tensor [heads, res, dim].
            v: Value tensor [heads, res, dim].
            attn_mask: Optional attention mask.
            is_cross: Whether this is cross-attention (True) or self-attention (False).

        Returns:
            Hidden states after attention computation.
        """
        self.cur_att_layer += 1
        position = self.check_position()

        if "B" in self.mode:
            return self._backward_mode(attn, q, k, v, attn_mask, is_cross, position)
        elif "F" in self.mode:
            return self._forward_mode(attn, q, k, v, attn_mask, is_cross, position)

        # Default: standard attention
        attention_probs = attn.get_attention_scores(q, k, attn_mask)
        return torch.bmm(attention_probs, v)

    def _backward_mode(self, attn, q, k, v, attn_mask, is_cross, position):
        """Handle backward guidance mode (storing attention maps for loss)."""
        attention_probs = attn.get_attention_scores(q, k, attn_mask)

        if "L" in self.mode:
            if is_cross and position in self.layout_selected_layer:
                self.A_CA_store.append(attention_probs)

        if "C" in self.mode:
            if is_cross and position in self.automask_selected_layer:
                self.A_CA_accumulate_automask += attention_probs.detach()

        return torch.bmm(attention_probs, v)

    def _forward_mode(self, attn, q, k, v, attn_mask, is_cross, position):
        """Handle forward guidance mode (active attention manipulation)."""
        if "L" in self.mode and is_cross and not self.is_source:
            # Layout forward: re-weight cross-attention with spatial masks
            attention_probs = attn.get_attention_scores(q, k, attn_mask)
            attention_probs = self._apply_layout_forward(attention_probs)
            return torch.bmm(attention_probs, v)

        elif "C" in self.mode:
            if is_cross and position in self.automask_selected_layer:
                attention_probs = attn.get_attention_scores(q, k, attn_mask)
                self.A_CA_accumulate_automask += attention_probs.detach()
                return torch.bmm(attention_probs, v)
            elif not is_cross and position in self.consistent_selected_layer:
                return self._apply_consistency_forward(attn, q, k, v, attn_mask)

        # Fallback: standard attention
        attention_probs = attn.get_attention_scores(q, k, attn_mask)
        return torch.bmm(attention_probs, v)

    # ------------------------------------------------------------------
    # Consistency guidance
    # ------------------------------------------------------------------

    def _apply_consistency_forward(self, attn, q, k, v, attn_mask=None):
        """Cross-batch self-attention sharing for identity consistency.

        Concatenates k/v from all batch items so each scene attends to all others,
        with spatial masking to restrict attention to object regions.
        """
        head = int(k.shape[0] / 2 / self.batch_size)
        seq_len = k.shape[1]
        dim = k.shape[2]

        # Reshape and broadcast k/v across batch items
        k_plus = (
            k.reshape(2, self.batch_size, head, seq_len, dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(2, head, -1, dim)
        )
        k_plus = (
            k_plus.unsqueeze(2)
            .repeat(1, 1, self.batch_size, 1, 1)
            .permute(0, 2, 1, 3, 4)
            .reshape(-1, self.batch_size * seq_len, dim)
        )

        v_plus = (
            v.reshape(2, self.batch_size, head, seq_len, dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(2, head, -1, dim)
        )
        v_plus = (
            v_plus.unsqueeze(2)
            .repeat(1, 1, self.batch_size, 1, 1)
            .permute(0, 2, 1, 3, 4)
            .reshape(-1, self.batch_size * seq_len, dim)
        )

        # Apply spatial layout mask (log-space for additive attention bias)
        if seq_len == 1024:
            log_mask = torch.log(
                self.layout_mask32.unsqueeze(1).unsqueeze(1).unsqueeze(0)
                .repeat(2, 1, head, 1024, 1)
                .reshape(-1, 1024, self.batch_size * 1024)
                + 1e-10
            )
        else:
            log_mask = torch.log(
                self.layout_mask64.unsqueeze(1).unsqueeze(1).unsqueeze(0)
                .repeat(2, 1, head, 4096, 1)
                .reshape(-1, 4096, self.batch_size * 4096)
                + 1e-10
            )

        attention_probs = attn.get_attention_scores(q, k_plus, log_mask)
        return torch.bmm(attention_probs, v_plus)

    # ------------------------------------------------------------------
    # Layout forward guidance
    # ------------------------------------------------------------------

    def _apply_layout_forward(self, attn: torch.Tensor) -> torch.Tensor:
        """Re-weight cross-attention to encourage objects within their bounding boxes.

        Args:
            attn: Attention probabilities [head, res, 77].

        Returns:
            Modified attention probabilities.
        """
        _, res, _ = attn.shape
        attn_changed = attn.clone()
        for bs in range(self.batch_size):
            for obj in range(self.obj_num):
                obj_ind = self.obj_indices[bs][obj]
                if res == 1024:
                    mask = self.layout_forward_masks32[bs, obj].unsqueeze(0)
                else:
                    mask = self.layout_forward_masks64[bs, obj].unsqueeze(0)
                attn_changed[:, :, obj_ind] = attn_changed[:, :, obj_ind] * mask
        return attn_changed

    # ------------------------------------------------------------------
    # Prompt embedding manipulation
    # ------------------------------------------------------------------

    def get_new_prompt_embeds(self, prompt_embeds: torch.Tensor) -> torch.Tensor:
        """Create mixed prompt embeddings for consistency guidance.

        Blends object token embeddings from the source (batch item 0) into
        target batch items to encourage semantic alignment.

        Args:
            prompt_embeds: Original prompt embeddings [2*bs, seq_len, dim].

        Returns:
            Modified prompt embeddings stored as self.new_prompt_embeds.
        """
        lamb = 0.5
        bs = self.batch_size
        new_prompt_embeds = prompt_embeds.clone()
        for bs_id, obj_indices_bs in enumerate(self.obj_indices):
            if bs_id == 0:
                continue
            for obj_id, obj_index in enumerate(obj_indices_bs):
                source_index = self.obj_indices[0][obj_id]
                new_prompt_embeds[bs + bs_id, obj_index] = (
                    prompt_embeds[bs + bs_id, obj_index] * (1 - lamb)
                    + prompt_embeds[bs, source_index] * lamb
                )
        self.new_prompt_embeds = new_prompt_embeds
        return self.new_prompt_embeds

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def process_between_steps(self):
        """Reset per-step storage between denoising steps."""
        self.A_CA_store = []
        self.A_SA_store = []
        self.cur_att_layer = 0

    def set_mode(self, mode: str = "D"):
        """Set the current manipulation mode."""
        self.mode = mode

    @torch.no_grad()
    def process_automask(self):
        """Process accumulated attention maps for auto-mask generation.

        Currently resets the accumulator. Can be extended to build
        dynamic object masks via Otsu binarization.
        """
        self.A_CA_accumulate_automask.zero_()

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------

    def _build_sdsa_layout_masks(self, bboxes: List[List[List[float]]], offset: float = 0.1):
        """Build spatial masks for cross-batch self-attention (consistency guidance).

        Controls which spatial regions of other batch items each scene can attend to.
        Target scenes only attend to object regions of the source scene.

        Args:
            bboxes: Per-batch bounding boxes.
            offset: Spatial expansion offset for mask boundaries.
        """
        mask_list32 = []
        mask_list64 = []
        for bs in range(self.batch_size):
            layout_mask32 = torch.zeros((32, 32), dtype=self.dtype, device=self.device)
            layout_mask64 = torch.zeros((64, 64), dtype=self.dtype, device=self.device)
            for obj in range(self.obj_num):
                obj_box = bboxes[bs][obj]
                # 32x32 resolution
                x_min = max((obj_box[0] - offset), 0.0) * 32
                y_min = max((obj_box[1] - offset), 0.0) * 32
                x_max = min((obj_box[2] + offset), 1.0) * 32
                y_max = min((obj_box[3] + offset), 1.0) * 32
                layout_mask32[round(y_min):round(y_max), round(x_min):round(x_max)] = 1
                # 64x64 resolution
                x_min = max((obj_box[0] - offset), 0.0) * 64
                y_min = max((obj_box[1] - offset), 0.0) * 64
                x_max = min((obj_box[2] + offset), 1.0) * 64
                y_max = min((obj_box[3] + offset), 1.0) * 64
                layout_mask64[round(y_min):round(y_max), round(x_min):round(x_max)] = 1
            mask_list32.append(layout_mask32.view(-1))
            mask_list64.append(layout_mask64.view(-1))

        layout_masks32 = torch.stack(mask_list32)
        layout_masks64 = torch.stack(mask_list64)

        mask_ones_32 = torch.ones_like(layout_masks32)
        mask_ones_64 = torch.ones_like(layout_masks64)
        mask_zeros_32 = torch.zeros_like(layout_masks32)
        mask_zeros_64 = torch.zeros_like(layout_masks64)

        # Source: attends to self only (zeros for target regions)
        # Target: attends to source object regions + self fully
        self.layout_mask32 = torch.stack([
            torch.cat((mask_ones_32[0], mask_zeros_32[1]), dim=0),
            torch.cat((layout_masks32[0], mask_ones_32[1]), dim=0),
        ])
        self.layout_mask64 = torch.stack([
            torch.cat((mask_ones_64[0], mask_zeros_64[1]), dim=0),
            torch.cat((layout_masks64[0], mask_ones_64[1]), dim=0),
        ])

    def _build_forward_layout_masks(self, bboxes: List[List[List[float]]], offset: float = 0.0):
        """Build per-object spatial masks for layout forward guidance.

        Each object's mask is the complement of other objects' bounding boxes,
        encouraging attention for each object to concentrate in its own region.

        Args:
            bboxes: Per-batch bounding boxes.
            offset: Spatial expansion offset.
        """
        mask_list32 = []
        mask_list64 = []
        for bs in range(self.batch_size):
            for obj in range(self.obj_num):
                layout_mask32 = torch.ones((32, 32), dtype=self.dtype, device=self.device)
                layout_mask64 = torch.ones((64, 64), dtype=self.dtype, device=self.device)
                for other_obj in range(self.obj_num):
                    if obj == other_obj:
                        continue
                    obj_box = bboxes[bs][other_obj]
                    # Zero out other objects' regions at 32x32
                    x_min = max((obj_box[0] - offset), 0.0) * 32
                    y_min = max((obj_box[1] - offset), 0.0) * 32
                    x_max = min((obj_box[2] + offset), 1.0) * 32
                    y_max = min((obj_box[3] + offset), 1.0) * 32
                    layout_mask32[round(y_min):round(y_max), round(x_min):round(x_max)] = 0
                    # Zero out other objects' regions at 64x64
                    x_min = max((obj_box[0] - offset), 0.0) * 64
                    y_min = max((obj_box[1] - offset), 0.0) * 64
                    x_max = min((obj_box[2] + offset), 1.0) * 64
                    y_max = min((obj_box[3] + offset), 1.0) * 64
                    layout_mask64[round(y_min):round(y_max), round(x_min):round(x_max)] = 0
                mask_list32.append(layout_mask32.view(-1))
                mask_list64.append(layout_mask64.view(-1))

        self.layout_forward_masks32 = torch.stack(mask_list32).reshape(self.batch_size, self.obj_num, -1)
        self.layout_forward_masks64 = torch.stack(mask_list64).reshape(self.batch_size, self.obj_num, -1)

    # ------------------------------------------------------------------
    # Utility: Otsu binarization (for auto-mask extension)
    # ------------------------------------------------------------------

    @staticmethod
    def otsu_binarization(image: torch.Tensor, device: str = "cuda", dtype=torch.float16, offset: int = 5) -> torch.Tensor:
        """Apply Otsu's binarization method to a single-channel image tensor.

        Args:
            image: Input 2D tensor (values in [0, 255]).
            device: Computation device.
            dtype: Output tensor dtype.
            offset: Threshold offset for adjustment.

        Returns:
            Binary mask tensor.
        """
        pixel_counts = image.view(-1).to(torch.float32)
        hist = torch.histc(pixel_counts, bins=256, min=0, max=255).to(device)
        total_pixels = pixel_counts.size(0)
        sum_total = torch.dot(torch.arange(256, device=device, dtype=torch.float32), hist)

        sum_bg = torch.tensor(0.0, device=device)
        weight_bg = torch.tensor(0.0, device=device)
        max_variance = torch.tensor(0.0, device=device)
        threshold = torch.tensor(0.0, device=device)

        for i in range(256):
            weight_bg += hist[i]
            if weight_bg == 0:
                continue
            weight_fg = total_pixels - weight_bg
            if weight_fg == 0:
                break
            sum_bg += i * hist[i]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if variance > max_variance:
                max_variance = variance
                threshold = torch.tensor(float(i), device=device)

        adjusted_threshold = max(threshold.item() - offset, 0)
        return (image > adjusted_threshold).to(dtype)
