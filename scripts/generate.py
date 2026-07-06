"""SpotActor — Config-driven generation script.

Generates images based on a YAML configuration file that defines subjects,
prompts, bounding boxes, and all hyperparameters.

Usage:
    python scripts/generate.py --config configs/examples/cat_dog.yaml

The script loads configs/default.yaml as base, then merges the case config
on top. Any field in the case config overrides the corresponding default.
"""

import argparse
import os
import sys
from copy import deepcopy

import torch
import yaml
from diffusers import EulerDiscreteScheduler
from diffusers.models import UNet2DConditionModel
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spotactor import SpotActorXLPipeline


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(config_path: str) -> dict:
    """Load case config."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load case config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="SpotActor: Config-driven generation")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to case YAML config file")
    parser.add_argument("--targets", type=str, default=None,
                        help="Comma-separated target indices to generate (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Extract config sections
    model_cfg = cfg["model"]
    gen_cfg = cfg["generation"]
    spot_cfg = cfg["spotactor"]
    out_cfg = cfg["output"]

    subjects = cfg["subjects"]
    central_prompt = cfg["central_prompt"]
    source_cfg = cfg["source"]
    targets_cfg = cfg["targets"]

    # Determine which targets to generate
    if args.targets:
        target_indices = [int(x) for x in args.targets.split(",")]
    else:
        target_indices = list(range(len(targets_cfg)))

    # SpotActor parameter dict (values can be numbers or "auto")
    spotactor_params = {
        "layout_guidance_steps": spot_cfg["layout_guidance_steps"],
        "consistent_guidance_steps": spot_cfg["consistent_guidance_steps"],
        "backward_iter_per_step": spot_cfg["backward_iter_per_step"],
        "scale_factor": spot_cfg["scale_factor"],
        "loss_thres": spot_cfg["loss_thres"],
        "semantic_rescale": spot_cfg["semantic_rescale"],
        "inference_steps": gen_cfg["num_inference_steps"],
        "adaptive_verbose": spot_cfg.get("adaptive_verbose", True),
    }

    negative_prompt = gen_cfg["negative_prompt"]
    guidance_scale = gen_cfg["guidance_scale"]

    # ==================== Pipeline Setup ====================
    print("Loading pipeline...")
    pipe = SpotActorXLPipeline.from_pretrained(
        model_cfg["path"],
        torch_dtype=torch.float16,
        variant=model_cfg["variant"],
    )
    pipe.unet = UNet2DConditionModel.from_pretrained(
        model_cfg["path"],
        subfolder="unet",
        torch_dtype=torch.float16,
        variant=model_cfg["variant"],
    )
    pipe = pipe.to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    output_dir = out_cfg["dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ==================== Stage 1: Source Generation ====================
    source_scene = source_cfg["scene"]
    source_bbox = source_cfg["bbox"]
    seed_source = source_cfg["seed"]
    source_prompt = f"{central_prompt}, {source_scene}"

    source_config = {
        **spotactor_params,
        "is_source": True,
        "source_latent": None,
        "consistent_guidance_steps": 0,
        "obj_tokens": [subjects],
        "sce_tokens": [source_scene.split()],
        "bboxes": [source_bbox],
        "prompt": [source_prompt],
    }

    generator = torch.Generator(device="cuda").manual_seed(seed_source)
    print(f"[Source] seed={seed_source} | scene=\"{source_scene}\"")

    source_images, source_latent = pipe(
        prompt=[source_prompt],
        negative_prompt=[negative_prompt],
        num_inference_steps=gen_cfg["num_inference_steps"],
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=1,
        return_dict=False,
        extra_config=source_config,
    )

    src_img = source_images[0]
    if out_cfg["draw_box"]:
        src_img = pipe.draw_box(src_img, source_bbox, subjects)
    src_img.save(os.path.join(output_dir, f"source_seed{seed_source}.jpg"))
    print(f"  Saved: {output_dir}/source_seed{seed_source}.jpg")

    # ==================== Stage 2: Target Generation ====================
    all_images = [src_img]

    for t_idx in target_indices:
        t_cfg = targets_cfg[t_idx]
        t_scene = t_cfg["scene"]
        t_bbox = t_cfg["bbox"]
        t_seed = t_cfg["seed"]
        target_prompt = f"{central_prompt}, {t_scene}"

        target_config = {
            **spotactor_params,
            "is_source": False,
            "source_latent": source_latent.to("cuda"),
            "obj_tokens": [subjects, subjects],
            "sce_tokens": [source_scene.split(), t_scene.split()],
            "bboxes": [source_bbox, t_bbox],
            "prompt": [source_prompt, target_prompt],
        }

        generator = torch.Generator(device="cuda").manual_seed(t_seed)
        print(f"[Target {t_idx+1}/{len(targets_cfg)}] seed={t_seed} | scene=\"{t_scene}\"")

        target_images, _ = pipe(
            prompt=[source_prompt, target_prompt],
            negative_prompt=[negative_prompt] * 2,
            num_inference_steps=gen_cfg["num_inference_steps"],
            guidance_scale=guidance_scale,
            generator=generator,
            num_images_per_prompt=1,
            return_dict=False,
            extra_config=target_config,
        )

        tgt_img = target_images[1]
        if out_cfg["draw_box"]:
            tgt_img = pipe.draw_box(tgt_img, t_bbox, subjects)
        tgt_img.save(os.path.join(output_dir, f"target{t_idx+1}_seed{t_seed}.jpg"))
        print(f"  Saved: {output_dir}/target{t_idx+1}_seed{t_seed}.jpg")
        all_images.append(tgt_img)

    # ==================== Save Overview ====================
    if out_cfg.get("save_overview", True) and len(all_images) > 1:
        n = len(all_images)
        overview = Image.new("RGB", (n * 1024, 1024))
        for i, img in enumerate(all_images):
            overview.paste(img.resize((1024, 1024)), (i * 1024, 0))
        overview.save(os.path.join(output_dir, "overview.jpg"))
        print(f"  Saved: {output_dir}/overview.jpg")

    print("Done!")


if __name__ == "__main__":
    main()
