"""Adaptive Hyperparameter Scheduler for SpotActor.

Automatically adjusts SpotActor guidance parameters that are set to "auto"
based on:
1. Static analysis: Bbox geometry (area, overlap, spacing, size ratio)
2. Dynamic feedback: Online loss convergence behavior during backward guidance

Design:
- Parameters set to a specific numeric value in config are used as-is.
- Parameters set to "auto" are dynamically computed based on layout difficulty.
- During backward guidance, scale_factor (if auto) is further adapted online
  based on loss convergence behavior (plateau → boost, oscillation → decay).

Usage:
    scheduler = AdaptiveScheduler(raw_params, num_inference_steps)
    resolved_params = scheduler.resolve(bboxes, is_source)
    # During backward loop (only if scale_factor is auto):
    scheduler.report_loss(iter_idx, loss_value)
    current_scale = scheduler.get_scale_factor()
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

import numpy as np


@dataclass
class LayoutDifficulty:
    """Geometric difficulty metrics computed from bounding boxes."""
    num_objects: int = 0
    avg_area: float = 0.0       # Mean normalized area of all bboxes
    min_area: float = 0.0       # Smallest bbox area (bottleneck)
    max_area: float = 0.0       # Largest bbox area
    size_ratio: float = 1.0     # max_area / min_area (imbalance factor)
    min_distance: float = 1.0   # Min center-to-center distance (normalized)
    max_iou: float = 0.0        # Maximum IoU between any pair
    total_coverage: float = 0.0 # Total area covered by all bboxes
    difficulty_score: float = 0.0  # Composite [0, 1], higher = harder


class AdaptiveScheduler:
    """Geometry-aware + loss-adaptive hyperparameter scheduler.

    Only adjusts parameters that are marked as "auto" in the config.
    Fixed numeric values pass through untouched.

    Args:
        raw_params: Dict from YAML config (values are either numbers or "auto").
        num_inference_steps: Total denoising steps.
        verbose: Whether to print adaptation logs.
    """

    # Empirical parameter ranges (tuned for SDXL 1024x1024)
    _PARAM_BOUNDS = {
        "layout_guidance_steps": (1, 5),
        "consistent_guidance_steps": (10, 25),
        "backward_iter_per_step": (15, 50),
        "scale_factor": (100.0, 800.0),
        "loss_thres": (0.10, 0.30),
        "semantic_rescale": (0.001, 0.008),
    }

    # The set of params that can be "auto"
    AUTO_CAPABLE_PARAMS = set(_PARAM_BOUNDS.keys())

    # Auto-derived: params whose bounds are both int → value should be int
    _INT_KEYS = frozenset(
        k for k, (lo, hi) in _PARAM_BOUNDS.items()
        if isinstance(lo, int) and isinstance(hi, int)
    )

    def __init__(
        self,
        raw_params: Dict,
        num_inference_steps: int = 30,
        verbose: bool = True,
    ):
        self.raw_params = raw_params
        self.num_inference_steps = num_inference_steps
        self.verbose = verbose

        # Identify which params are auto vs fixed
        self._auto_keys: Set[str] = set()
        self._fixed_values: Dict[str, Union[int, float]] = {}

        for key in self.AUTO_CAPABLE_PARAMS:
            val = raw_params.get(key)
            if val == "auto":
                self._auto_keys.add(key)
            elif val is not None:
                self._fixed_values[key] = int(val) if key in self._INT_KEYS else float(val)

        # Online adaptation state (for scale_factor)
        self._loss_history: List[float] = []
        self._current_scale_factor: Optional[float] = None
        self._scale_is_auto = "scale_factor" in self._auto_keys
        self._adaptation_count = 0

    @property
    def has_auto_params(self) -> bool:
        """Whether any parameters need automatic computation."""
        return len(self._auto_keys) > 0

    # ------------------------------------------------------------------
    # Static geometry analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
        """Compute IoU between two normalized boxes [x_min, y_min, x_max, y_max]."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / (union + 1e-8)

    @staticmethod
    def _box_center(box: List[float]) -> tuple:
        """Compute center of a normalized box."""
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        return cx, cy

    def analyze_difficulty(self, bboxes: List[List[float]]) -> LayoutDifficulty:
        """Analyze layout difficulty from bounding box geometry.

        Args:
            bboxes: List of normalized bboxes [x_min, y_min, x_max, y_max].

        Returns:
            LayoutDifficulty dataclass with computed metrics.
        """
        n = len(bboxes)
        if n == 0:
            return LayoutDifficulty()

        # Compute areas
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
        avg_area = np.mean(areas)
        min_area = np.min(areas)
        max_area = np.max(areas)
        size_ratio = max_area / (min_area + 1e-8)
        total_coverage = sum(areas)

        # Compute pairwise metrics
        max_iou = 0.0
        min_distance = 2.0
        for i in range(n):
            for j in range(i + 1, n):
                iou = self._compute_iou(bboxes[i], bboxes[j])
                max_iou = max(max_iou, iou)
                ci = self._box_center(bboxes[i])
                cj = self._box_center(bboxes[j])
                dist = np.sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)
                min_distance = min(min_distance, dist)

        # Composite difficulty score [0, 1]
        area_difficulty = np.clip(1.0 - min_area / 0.25, 0, 1)
        count_difficulty = np.clip((n - 1) / 4.0, 0, 1)
        proximity_difficulty = np.clip(1.0 - min_distance / 0.5, 0, 1)
        overlap_difficulty = np.clip(max_iou / 0.3, 0, 1)
        imbalance_difficulty = np.clip((size_ratio - 1) / 5.0, 0, 1)

        difficulty_score = (
            0.35 * area_difficulty
            + 0.15 * count_difficulty
            + 0.20 * proximity_difficulty
            + 0.15 * overlap_difficulty
            + 0.15 * imbalance_difficulty
        )
        difficulty_score = float(np.clip(difficulty_score, 0, 1))

        return LayoutDifficulty(
            num_objects=n,
            avg_area=float(avg_area),
            min_area=float(min_area),
            max_area=float(max_area),
            size_ratio=float(size_ratio),
            min_distance=float(min_distance),
            max_iou=float(max_iou),
            total_coverage=float(total_coverage),
            difficulty_score=difficulty_score,
        )

    # ------------------------------------------------------------------
    # Parameter computation
    # ------------------------------------------------------------------

    def _lerp(self, key: str, t: float) -> float:
        """Linear interpolation between parameter bounds based on difficulty t in [0,1]."""
        lo, hi = self._PARAM_BOUNDS[key]
        return lo + (hi - lo) * t

    def _cast(self, key: str, value) -> Union[int, float]:
        """Cast *value* to the correct type for *key* based on _PARAM_BOUNDS."""
        return int(value) if key in self._INT_KEYS else float(value)

    def _compute_auto_params(self, bboxes: List[List[float]]) -> Dict[str, Union[int, float]]:
        """Compute values for all auto parameters given bbox layout.

        Only computes for keys in self._auto_keys.
        """
        diff = self.analyze_difficulty(bboxes)
        d = diff.difficulty_score
        result = {}

        if "layout_guidance_steps" in self._auto_keys:
            result["layout_guidance_steps"] = self._cast(
                "layout_guidance_steps", np.ceil(self._lerp("layout_guidance_steps", d))
            )

        if "backward_iter_per_step" in self._auto_keys:
            result["backward_iter_per_step"] = self._cast(
                "backward_iter_per_step", np.ceil(self._lerp("backward_iter_per_step", d))
            )

        if "scale_factor" in self._auto_keys:
            # Smaller bboxes → stronger scaling needed
            area_factor = np.clip(0.15 / (diff.min_area + 0.01), 0.5, 3.0)
            base_scale = 300  # default reference
            scale_factor = np.clip(
                base_scale * area_factor * (0.7 + 0.6 * d),
                self._PARAM_BOUNDS["scale_factor"][0],
                self._PARAM_BOUNDS["scale_factor"][1],
            )
            result["scale_factor"] = self._cast("scale_factor", scale_factor)

        if "loss_thres" in self._auto_keys:
            # Harder layouts → stricter threshold (lower value)
            result["loss_thres"] = self._cast(
                "loss_thres", self._lerp("loss_thres", 1.0 - d)
            )

        if "semantic_rescale" in self._auto_keys:
            result["semantic_rescale"] = self._cast(
                "semantic_rescale", self._lerp("semantic_rescale", d * 0.7)
            )

        if "consistent_guidance_steps" in self._auto_keys:
            cgs_ratio = 0.55 + 0.20 * d  # 55%-75% of total steps
            consistent_steps = np.round(cgs_ratio * self.num_inference_steps)
            consistent_steps = np.clip(
                consistent_steps,
                self._PARAM_BOUNDS["consistent_guidance_steps"][0],
                self._PARAM_BOUNDS["consistent_guidance_steps"][1],
            )
            result["consistent_guidance_steps"] = self._cast(
                "consistent_guidance_steps", consistent_steps
            )

        if self.verbose and result:
            print(f"  [AdaptiveScheduler] difficulty={d:.3f} "
                  f"(area={diff.min_area:.3f}, dist={diff.min_distance:.3f}, "
                  f"iou={diff.max_iou:.3f}, ratio={diff.size_ratio:.1f})")
            auto_str = ", ".join(f"{k}={v}" for k, v in result.items())
            print(f"    → auto: {auto_str}")

        return result

    # ------------------------------------------------------------------
    # Main API: resolve params for a generation stage
    # ------------------------------------------------------------------

    def resolve(
        self,
        bboxes_all: List[List[List[float]]],
        is_source: bool,
    ) -> Dict[str, Union[int, float]]:
        """Resolve all spotactor params: fixed values + auto-computed values.

        Args:
            bboxes_all: Full bboxes list as passed to pipeline (per-batch).
            is_source: Whether this is source-stage generation.

        Returns:
            Dict of fully resolved numeric hyperparameters.
        """
        # Start with fixed values
        resolved = dict(self._fixed_values)

        # Compute auto params based on target bbox difficulty
        if self._auto_keys:
            target_bboxes = bboxes_all[0] if is_source else bboxes_all[-1]
            auto_vals = self._compute_auto_params(target_bboxes)
            resolved.update(auto_vals)

        # Source stage overrides
        if is_source:
            resolved["consistent_guidance_steps"] = 0

        # Set up online adaptation state
        self._loss_history = []
        self._current_scale_factor = resolved.get("scale_factor", 300)
        self._adaptation_count = 0

        return resolved

    # ------------------------------------------------------------------
    # Online adaptation (called inside backward guidance loop)
    # ------------------------------------------------------------------

    def report_loss(self, iteration: int, loss_value: float):
        """Report current loss for online scale_factor adaptation.

        Only adapts scale_factor when it was set to "auto".
        Should be called once per backward iteration.
        """
        if not self._scale_is_auto:
            return

        self._loss_history.append(loss_value)

        # Need at least 5 data points to analyze trend
        if len(self._loss_history) < 5:
            return

        # Recent convergence rate
        recent = self._loss_history[-5:]
        decay_rate = (recent[0] - recent[-1]) / (recent[0] + 1e-8)

        # Plateau detection: boost scale
        if decay_rate < 0.05 and self._adaptation_count < 3:
            old_scale = self._current_scale_factor
            self._current_scale_factor = min(
                self._current_scale_factor * 1.3,
                self._PARAM_BOUNDS["scale_factor"][1],
            )
            self._adaptation_count += 1
            if self.verbose and self._current_scale_factor != old_scale:
                print(f"    [Adapt] plateau at iter {iteration}, "
                      f"scale: {old_scale:.0f} → {self._current_scale_factor:.0f}")

        # Oscillation detection: decay scale
        if len(self._loss_history) >= 4:
            diffs = [self._loss_history[i+1] - self._loss_history[i]
                     for i in range(len(self._loss_history)-3, len(self._loss_history)-1)]
            if len(diffs) >= 2 and diffs[0] * diffs[1] < 0:
                old_scale = self._current_scale_factor
                self._current_scale_factor = max(
                    self._current_scale_factor * 0.85,
                    self._PARAM_BOUNDS["scale_factor"][0],
                )
                if self.verbose and self._current_scale_factor != old_scale:
                    print(f"    [Adapt] oscillation at iter {iteration}, "
                          f"scale: {old_scale:.0f} → {self._current_scale_factor:.0f}")

    def get_scale_factor(self) -> float:
        """Get current (potentially adapted) scale_factor."""
        return self._current_scale_factor
