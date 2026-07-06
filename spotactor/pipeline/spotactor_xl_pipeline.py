"""SpotActor SDXL Pipeline — Two-stage layout-guided consistent generation.

This pipeline extends Stable Diffusion XL with:
1. Layout Backward Guidance: Optimizes latents via attention-map-based loss
   to place objects within specified bounding boxes.
2. Layout Forward Guidance: Re-weights cross-attention at inference to
   spatially concentrate object tokens.
3. Consistency Forward Guidance: Shares self-attention keys/values across
   batch items (scenes) to maintain identity consistency.

Generation follows a two-stage protocol:
- Stage 1 (Source): Generate a reference image with layout control.
- Stage 2 (Target): Generate target scenes conditioned on source latents
  for cross-scene consistency.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.stable_diffusion_xl import (
    StableDiffusionXLPipeline,
    StableDiffusionXLPipelineOutput,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from PIL import Image, ImageDraw
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from ..attention import Manipulator, SpotActorAttnProcessor
from ..utils import AdaptiveScheduler

logger = logging.get_logger(__name__)


class SpotActorXLPipeline(StableDiffusionXLPipeline):
    """Stable Diffusion XL pipeline with SpotActor layout + consistency guidance.

    Inherits from StableDiffusionXLPipeline and adds:
    - Two-stage generation (source → target)
    - Attention-based layout backward/forward guidance
    - Cross-batch self-attention sharing for identity consistency
    - Automatic bounding box visualization

    Args:
        vae: Variational autoencoder model.
        text_encoder: First CLIP text encoder.
        text_encoder_2: Second CLIP text encoder with projection.
        tokenizer: First CLIP tokenizer.
        tokenizer_2: Second CLIP tokenizer.
        unet: UNet2D conditional model.
        scheduler: Diffusion scheduler.
        image_encoder: Optional CLIP image encoder.
        feature_extractor: Optional CLIP image feature extractor.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )

    # ------------------------------------------------------------------
    # Tokenization utilities
    # ------------------------------------------------------------------

    def get_token_indices(self, prompt: List[str]) -> Tuple[List[Dict], List, List[Dict]]:
        """Map tokens to their positions in the tokenized prompt.

        Args:
            prompt: List of prompt strings (one per batch item).

        Returns:
            Tuple of (token_to_id_list, length_ranges, id_to_token_list).
        """
        ids_list = self.tokenizer(prompt).input_ids
        token_to_id_list = []
        len_list = []
        id_to_token_list = []
        for ids in ids_list:
            decoded = [self.tokenizer.decode(i) for i in ids]
            token_to_id = {tok: idx for idx, tok in enumerate(decoded)}
            id_to_token = {idx: tok for idx, tok in enumerate(decoded)}
            token_to_id_list.append(token_to_id)
            len_list.append(range(len(ids)))
            id_to_token_list.append(id_to_token)
        return token_to_id_list, len_list, id_to_token_list

    @staticmethod
    def get_obj_indices(
        token_to_id: List[Dict[str, int]], tokens: List[List[str]]
    ) -> List[List[int]]:
        """Look up token indices for specified object/scene tokens.

        Args:
            token_to_id: Per-batch token-to-index mapping.
            tokens: Per-batch list of token strings to look up.

        Returns:
            Per-batch list of integer indices.
        """
        return [
            [indices_per_batch[t] for t in tokens[bs]]
            for bs, indices_per_batch in enumerate(token_to_id)
        ]

    # ------------------------------------------------------------------
    # Loss computation for layout backward guidance
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        all_indices: List,
        obj_indices: List[List[int]],
        sce_indices: List[List[int]],
        bboxes: List[List[List[float]]],
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Compute layout alignment loss from stored cross-attention maps.

        For each batch item (except source when in target mode), computes
        MSE between the object's attention map and its target bounding box mask.

        Args:
            all_indices: Per-batch token index ranges.
            obj_indices: Per-batch object token indices.
            sce_indices: Per-batch scene token indices.
            bboxes: Per-batch normalized bounding boxes.
            device: Computation device.

        Returns:
            List of per-batch loss tensors.
        """
        assert len(self.manipulator.A_CA_store) > 0, "No attention maps stored"
        loss_list = []

        attn_maps = torch.stack(self.manipulator.A_CA_store)
        n, _, res, dim = attn_maps.shape
        attn_maps = (
            attn_maps.reshape(n, 2, self.manipulator.batch_size, -1, res, dim)
            .permute(1, 2, 0, 3, 4, 5)
        )  # [2, bs, n, head, res, dim]

        # Use conditional (non-negative) attention maps
        attn_maps_bs = attn_maps.reshape(
            2, self.manipulator.batch_size, -1, res, dim
        )[1]

        for ind, attn_map in enumerate(attn_maps_bs):
            # Skip source batch item in target stage
            if not self.manipulator.is_source and ind == 0:
                continue

            b, i, j = attn_map.shape
            H = W = int(np.sqrt(i))

            # Normalize attention maps (per-head, per-token)
            ca_map = attn_map
            max_val = ca_map.max(dim=1, keepdim=True).values
            max_val = max_val.max(dim=2, keepdim=True).values
            ca_map = ca_map / (max_val + 1e-8)
            ca_map = ca_map.reshape(b, H, W, 77)

            loss = 0.0
            object_number = len(bboxes[ind])

            for obj_idx in range(object_number):
                obj_box = bboxes[ind][obj_idx]
                obj_ind = obj_indices[ind][obj_idx]
                x_min, y_min, x_max, y_max = (
                    obj_box[0] * W,
                    obj_box[1] * H,
                    obj_box[2] * W,
                    obj_box[3] * H,
                )

                # Create target mask: 1 inside bbox, 0 outside
                mask_obj = torch.zeros((H, W), dtype=torch.float16, device=device)
                mask_obj[round(y_min):round(y_max), round(x_min):round(x_max)] = 1.0
                mask_obj_expanded = mask_obj.unsqueeze(0).repeat(b, 1, 1)

                loss = loss + torch.nn.functional.mse_loss(
                    ca_map[:, :, :, obj_ind], mask_obj_expanded
                )

            loss_list.append(loss)

        return loss_list

    # ------------------------------------------------------------------
    # Attention registration
    # ------------------------------------------------------------------

    def register_attention_control(self):
        """Replace UNet attention processors with SpotActor processors."""
        attn_procs = {}
        att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                raise ValueError(f"Unknown UNet block: {name}")
            att_count += 1
            attn_procs[name] = SpotActorAttnProcessor(
                manipulator=self.manipulator, place_in_unet=place_in_unet
            )
        self.unet.set_attn_processor(attn_procs)
        self.manipulator.num_att_layers = att_count

    # ------------------------------------------------------------------
    # Visualization utilities
    # ------------------------------------------------------------------

    # Per-object color palette (distinct, high-contrast colors)
    _BOX_COLORS = [
        "#FF4D4D",  # red
        "#4DC8FF",  # sky blue
        "#FFD700",  # gold
        "#7CFC00",  # lawn green
        "#FF69B4",  # hot pink
        "#FF8C00",  # dark orange
    ]

    @staticmethod
    def draw_box(
        pil_img: Image.Image,
        bboxes: List[List[float]],
        subjects: Optional[List[str]] = None,
    ) -> Image.Image:
        """Draw bounding boxes with per-object colors and name labels.

        Args:
            pil_img: Input PIL image.
            bboxes: List of normalized bounding boxes [x_min, y_min, x_max, y_max].
            subjects: Optional list of object names (one per bbox).
                      When provided, each box gets a distinct color and a label.

        Returns:
            Image with drawn bounding boxes and optional labels.
        """
        from PIL import ImageFont

        width, height = pil_img.size
        box_width = max(3, int(min(width, height) * 0.004))
        font_size = max(20, int(min(width, height) * 0.035))

        # Try to load a bold system font; fall back to PIL default
        font = None
        for font_path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except (IOError, OSError):
                continue
        if font is None:
            font = ImageFont.load_default()

        colors = SpotActorXLPipeline._BOX_COLORS
        draw = ImageDraw.Draw(pil_img)

        for i, obj_box in enumerate(bboxes):
            color = colors[i % len(colors)]
            x_min = int(obj_box[0] * width)
            y_min = int(obj_box[1] * height)
            x_max = int(obj_box[2] * width)
            y_max = int(obj_box[3] * height)

            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=box_width)

            if subjects and i < len(subjects):
                label = subjects[i]
                # Measure text size
                try:
                    bbox_text = draw.textbbox((0, 0), label, font=font)
                    tw = bbox_text[2] - bbox_text[0]
                    th = bbox_text[3] - bbox_text[1]
                except AttributeError:
                    tw, th = draw.textsize(label, font=font)

                pad = max(3, int(font_size * 0.15))
                # Place label above box; clamp to image boundary
                lx = x_min
                ly = y_min - th - 2 * pad
                if ly < 0:
                    ly = y_min + pad  # place inside top of box if no room above

                # Semi-transparent background rectangle
                draw.rectangle(
                    [lx, ly, lx + tw + 2 * pad, ly + th + 2 * pad],
                    fill=color,
                )
                draw.text((lx + pad, ly + pad), label, fill="#000000", font=font)

        return pil_img

    # ------------------------------------------------------------------
    # Scheduler utility
    # ------------------------------------------------------------------

    def retrieve_timesteps(
        self,
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        **kwargs,
    ):
        """Retrieve timesteps from scheduler, supporting custom schedules."""
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(
                inspect.signature(scheduler.set_timesteps).parameters.keys()
            )
            if not accepts_timesteps:
                raise ValueError(
                    f"Scheduler {scheduler.__class__} does not support custom timesteps."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps

    # ------------------------------------------------------------------
    # Main generation loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        extra_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Generate images with layout and consistency guidance.

        Args:
            extra_config: SpotActor-specific configuration dict containing:
                - is_source (bool): Whether this is source-stage generation.
                - source_latent (Tensor|None): Latent from source stage (for target).
                - layout_guidance_steps (int): Steps with backward layout guidance.
                - consistent_guidance_steps (int): Steps with consistency guidance.
                - backward_iter_per_step (int|"auto"): Backward optimization iterations.
                - scale_factor (float|"auto"): Loss scaling factor.
                - loss_thres (float|"auto"): Early-stop loss threshold.
                - semantic_rescale (float|"auto"): Prompt embedding gradient scale.
                - obj_tokens (List[List[str]]): Object token strings per batch.
                - sce_tokens (List[List[str]]): Scene token strings per batch.
                - bboxes (List[List[List[float]]]): Bounding boxes per batch.
                - inference_steps (int): Number of denoising steps.
                - prompt (List[str]): Prompt list.

        Returns:
            Tuple of (images, initial_latent) when return_dict=False.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        # 0. Defaults
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Validate inputs
        self.check_inputs(
            prompt, prompt_2, height, width, callback_steps,
            negative_prompt, negative_prompt_2, prompt_embeds,
            negative_prompt_embeds, pooled_prompt_embeds,
            negative_pooled_prompt_embeds, callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False

        # 2. Batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode prompts
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )
        (
            prompt_embeds, negative_prompt_embeds,
            pooled_prompt_embeds, negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt, prompt_2=prompt_2, device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale, clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = self.retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )

        # 5. Prepare latents
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents,
            height, width, prompt_embeds.dtype, device, generator, latents,
        )

        # 6. Extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # ============ SpotActor Setup ============
        is_source = extra_config["is_source"]
        if not is_source and extra_config.get("source_latent") is not None:
            latents[:1] = extra_config["source_latent"]

        token_to_id, all_indices, id_to_token = self.get_token_indices(prompt)
        obj_indices = self.get_obj_indices(token_to_id, extra_config["obj_tokens"])
        sce_indices = self.get_obj_indices(token_to_id, extra_config["sce_tokens"])

        # --- Adaptive hyperparameter scheduling ---
        # Resolve "auto" params via AdaptiveScheduler; fixed values pass through
        adaptive_scheduler = AdaptiveScheduler(
            raw_params=extra_config,
            num_inference_steps=num_inference_steps,
            verbose=extra_config.get("adaptive_verbose", True),
        )
        if adaptive_scheduler.has_auto_params:
            resolved = adaptive_scheduler.resolve(
                bboxes_all=extra_config["bboxes"],
                is_source=is_source,
            )
            # Merge resolved numeric values into extra_config
            extra_config = {**extra_config, **resolved}
            
            self._adaptive_scheduler = adaptive_scheduler
        else:
            self._adaptive_scheduler = None

        self.manipulator = Manipulator(
            batch_size=batch_size,
            obj_indices=obj_indices,
            sce_indices=sce_indices,
            all_indices=all_indices,
            bboxes=extra_config["bboxes"],
            is_source=is_source,
        )
        self.manipulator.id_to_token = id_to_token
        self.register_attention_control()
        # =========================================

        # 7. Time embeddings
        add_text_embeds = pooled_prompt_embeds
        text_encoder_projection_dim = (
            int(pooled_prompt_embeds.shape[-1])
            if self.text_encoder_2 is None
            else self.text_encoder_2.config.projection_dim
        )
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        negative_add_time_ids = (
            self._get_add_time_ids(
                negative_original_size, negative_crops_coords_top_left,
                negative_target_size, dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
            if negative_original_size is not None and negative_target_size is not None
            else add_time_ids
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and 0 < self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(self.scheduler.config.num_train_timesteps
                      - (self.denoising_end * self.scheduler.config.num_train_timesteps))
            )
            num_inference_steps = len([ts for ts in timesteps if ts >= discrete_timestep_cutoff])
            timesteps = timesteps[:num_inference_steps]

        # Guidance scale embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        loss_thres = extra_config["loss_thres"]
        layout_forward_steps = 20

        # Prepare blended prompt embeddings for consistency
        self.manipulator.get_new_prompt_embeds(prompt_embeds)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # --- Backward Guidance Phase ---
                if i < extra_config["layout_guidance_steps"] and i < extra_config["consistent_guidance_steps"]:
                    self.manipulator.set_mode("LCB")
                elif i < extra_config["layout_guidance_steps"]:
                    self.manipulator.set_mode("LB")

                if "B" in self.manipulator.mode:
                    latents, prompt_embeds = self._backward_guidance_step(
                        latents, prompt_embeds, t, i,
                        add_text_embeds, add_time_ids, timestep_cond,
                        extra_config, all_indices, obj_indices, sce_indices,
                        loss_thres, is_source, device,
                    )

                # Save initial latent for return
                if i == 0:
                    initial_latent = latents

                # --- Forward Inference Step ---
                latent_model_input = (
                    torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                # Set forward mode
                if i < extra_config["consistent_guidance_steps"] and i < layout_forward_steps:
                    self.manipulator.set_mode("LCF")
                elif i < extra_config["consistent_guidance_steps"]:
                    self.manipulator.set_mode("CF")
                elif i < layout_forward_steps:
                    self.manipulator.set_mode("LF")
                else:
                    self.manipulator.set_mode("D")

                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if self.manipulator.mode != "D":
                    self.manipulator.process_automask()
                self.manipulator.process_between_steps()

                # Classifier-free guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = self.rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale
                    )

                # Denoise step
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # Callbacks
                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i // getattr(self.scheduler, "order", 1), t, latents)

        # 9. Decode latents
        if output_type != "latent":
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if output_type != "latent":
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, initial_latent.cpu())

        return StableDiffusionXLPipelineOutput(images=image)

    # ------------------------------------------------------------------
    # Backward guidance step (extracted for clarity)
    # ------------------------------------------------------------------

    def _backward_guidance_step(
        self, latents, prompt_embeds, t, step_idx,
        add_text_embeds, add_time_ids, timestep_cond,
        extra_config, all_indices, obj_indices, sce_indices,
        loss_thres, is_source, device,
    ):
        """Execute backward layout guidance optimization for one timestep.

        Iteratively optimizes latents and prompt embeddings to minimize
        the layout alignment loss.
        """
        with torch.enable_grad():
            latents = latents.detach().requires_grad_(True)
            prompt_embeds = prompt_embeds.detach().requires_grad_(True)
            
            for guidance_iter in range(extra_config["backward_iter_per_step"]):
                latent_model_input = (
                    torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                self.manipulator.A_CA_accumulate_automask.zero_()

                # Forward pass to collect attention maps
                self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )
                self.unet.zero_grad()
                self.manipulator.process_automask()

                # Compute layout loss
                loss_list = self._compute_loss(
                    all_indices, obj_indices, sce_indices, extra_config["bboxes"], device
                )

                if guidance_iter == 0:
                    start_loss_list = [l.item() for l in loss_list]

                # Online adaptation: report loss to adaptive scheduler
                if self._adaptive_scheduler is not None:
                    avg_loss = sum(l.item() for l in loss_list) / len(loss_list)
                    self._adaptive_scheduler.report_loss(guidance_iter, avg_loss)

                # Filter converged losses
                valid_loss = [l for l in loss_list if l.item() >= loss_thres]

                if len(valid_loss) == 0:
                    logger.info(f"Layout converged at iter {guidance_iter}, loss: {[l.item() for l in loss_list]}")
                    self.manipulator.process_between_steps()
                    del loss_list
                    break

                loss = torch.stack(valid_loss).mean() * (
                    self._adaptive_scheduler.get_scale_factor()
                    if self._adaptive_scheduler is not None
                    else extra_config["scale_factor"]
                )

                # Compute gradients
                grad_latent = torch.autograd.grad(
                    loss.requires_grad_(True), [latents], retain_graph=True
                )[0]
                grad_prompt = torch.autograd.grad(
                    loss, [prompt_embeds], retain_graph=True
                )[0]

                # Zero source gradients in target mode
                if not is_source:
                    grad_latent[0] = 0
                    grad_prompt[0] = 0

                # Update latents and embeddings
                sigma_sq = self.scheduler.sigmas[step_idx] ** 2
                latents = latents - grad_latent * sigma_sq
                prompt_embeds = prompt_embeds - extra_config["semantic_rescale"] * grad_prompt * sigma_sq

                self.manipulator.process_between_steps()
                # Explicitly free computation graph references to avoid OOM
                del loss, loss_list, valid_loss

        return latents, prompt_embeds
