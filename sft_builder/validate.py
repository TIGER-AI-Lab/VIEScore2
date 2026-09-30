"""
Validation module for SFT samples.

Must-pass validation (drop otherwise):
- score numeric
- text_reason non-empty (no generic filler)
- visual_reason present + non-empty
- visual evidence aligns with image (mask/heatmap size ok or safely resizable; bbox/polygon in-bounds)
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
from PIL import Image

from .schema import RawSample


logger = logging.getLogger(__name__)

# Generic filler phrases to reject
GENERIC_FILLER_PATTERNS = [
    r"^(good|nice|ok|fine|great)\.?$",
    r"^no (issues?|problems?)\.?$",
    r"^n/?a$",
    r"^none\.?$",
    r"^-+$",
    r"^\s*$",
]

# Minimum text reason length
MIN_TEXT_REASON_LENGTH = 10


@dataclass
class ValidationStats:
    """Statistics from validation."""
    total_input: int = 0
    passed: int = 0
    dropped_no_score: int = 0
    dropped_invalid_score: int = 0
    dropped_no_text_reason: int = 0
    dropped_generic_text: int = 0
    dropped_no_visual_reason: int = 0
    dropped_empty_visual: int = 0
    dropped_visual_size_mismatch: int = 0
    dropped_bbox_out_of_bounds: int = 0
    dropped_polygon_invalid: int = 0
    dropped_no_image: int = 0

    def summary(self) -> str:
        """Return a summary string."""
        lines = [
            f"Validation Summary:",
            f"  Total input: {self.total_input}",
            f"  Passed: {self.passed} ({100*self.passed/max(1,self.total_input):.1f}%)",
            f"  Dropped breakdown:",
            f"    - No score: {self.dropped_no_score}",
            f"    - Invalid score: {self.dropped_invalid_score}",
            f"    - No text_reason: {self.dropped_no_text_reason}",
            f"    - Generic text: {self.dropped_generic_text}",
            f"    - No visual_reason: {self.dropped_no_visual_reason}",
            f"    - Empty visual data: {self.dropped_empty_visual}",
            f"    - Visual size mismatch: {self.dropped_visual_size_mismatch}",
            f"    - Bbox out of bounds: {self.dropped_bbox_out_of_bounds}",
            f"    - Polygon invalid: {self.dropped_polygon_invalid}",
            f"    - No image: {self.dropped_no_image}",
        ]
        return "\n".join(lines)


class Validator:
    """Validates raw samples against must-pass criteria."""

    def __init__(
        self,
        require_image: bool = True,
        allow_visual_resize: bool = True,
        max_resize_ratio: float = 4.0,
        aspect_ratio_tolerance: float = 0.02,
    ):
        """
        Initialize validator.

        Args:
            require_image: Whether to require a valid image path
            allow_visual_resize: Allow visual evidence that needs resizing
            max_resize_ratio: Maximum absolute size ratio to accept. Only used
                as an upper bound safety net — the primary check is aspect
                ratio alignment, because ImagenWorld annotators often drew
                error masks on a rescaled (e.g. 640-px-short-side) view of
                the generated image, producing masks 2-3x the original size
                with the same aspect ratio.
            aspect_ratio_tolerance: Relative aspect-ratio difference allowed
                between mask and image (default 2%). A mask with the same
                aspect ratio as the image is, by definition, for the same
                image and can be safely rescaled at load time.
        """
        self.require_image = require_image
        self.allow_visual_resize = allow_visual_resize
        self.max_resize_ratio = max_resize_ratio
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.stats = ValidationStats()

    def validate(self, samples: Iterator[RawSample]) -> Iterator[RawSample]:
        """
        Validate samples, yielding only those that pass.

        Args:
            samples: Iterator of RawSample objects

        Yields:
            RawSample objects that pass validation
        """
        for sample in samples:
            self.stats.total_input += 1

            result, reason = self._validate_sample(sample)
            if result:
                self.stats.passed += 1
                yield sample
            else:
                logger.debug(f"Dropped {sample.id}: {reason}")

    def _validate_sample(self, sample: RawSample) -> Tuple[bool, str]:
        """
        Validate a single sample.

        Returns:
            (passed, reason) tuple
        """
        # Check score
        if sample.score is None:
            self.stats.dropped_no_score += 1
            return False, "no score"

        if not isinstance(sample.score, (int, float)):
            self.stats.dropped_invalid_score += 1
            return False, "score not numeric"

        if np.isnan(sample.score) or np.isinf(sample.score):
            self.stats.dropped_invalid_score += 1
            return False, "score is nan/inf"

        # Check text_reason
        if not sample.text_reason:
            self.stats.dropped_no_text_reason += 1
            return False, "no text_reason"

        if len(sample.text_reason.strip()) < MIN_TEXT_REASON_LENGTH:
            self.stats.dropped_generic_text += 1
            return False, "text_reason too short"

        if self._is_generic_filler(sample.text_reason):
            self.stats.dropped_generic_text += 1
            return False, "text_reason is generic filler"

        # Check visual_reason
        if sample.visual_reason_type is None:
            self.stats.dropped_no_visual_reason += 1
            return False, "no visual_reason type"

        if sample.visual_reason_data is None:
            self.stats.dropped_empty_visual += 1
            return False, "no visual_reason data"

        # Validate visual evidence
        visual_valid, visual_reason = self._validate_visual_evidence(sample)
        if not visual_valid:
            return False, visual_reason

        # Check image (optional based on config)
        if self.require_image:
            if not sample.image:
                self.stats.dropped_no_image += 1
                return False, "no image path"
            if not Path(sample.image).exists():
                self.stats.dropped_no_image += 1
                return False, "image file not found"

        return True, "passed"

    def _is_generic_filler(self, text: str) -> bool:
        """Check if text is generic filler."""
        text_lower = text.strip().lower()
        for pattern in GENERIC_FILLER_PATTERNS:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    def _validate_visual_evidence(self, sample: RawSample) -> Tuple[bool, str]:
        """Validate visual evidence data."""
        vr_type = sample.visual_reason_type
        vr_data = sample.visual_reason_data

        if vr_type == "mask":
            return self._validate_mask(vr_data, sample)
        elif vr_type == "heatmap":
            return self._validate_heatmap(vr_data, sample)
        elif vr_type == "bbox":
            return self._validate_bbox(vr_data, sample)
        elif vr_type == "polygon":
            return self._validate_polygon(vr_data, sample)
        elif vr_type == "svg":
            return self._validate_svg(vr_data, sample)
        else:
            self.stats.dropped_no_visual_reason += 1
            return False, f"unknown visual_reason type: {vr_type}"

    def _validate_mask(self, data, sample: RawSample) -> Tuple[bool, str]:
        """Validate mask data."""
        # If it's a path, check it exists and load dimensions
        if isinstance(data, str):
            mask_path = Path(data)
            if not mask_path.exists():
                self.stats.dropped_empty_visual += 1
                return False, "mask file not found"

            try:
                with Image.open(mask_path) as mask_img:
                    mask_w, mask_h = mask_img.size
            except Exception as e:
                self.stats.dropped_empty_visual += 1
                return False, f"cannot read mask: {e}"

            # Check size alignment with image
            if sample.image_width and sample.image_height:
                if not self._check_size_alignment(
                    mask_w, mask_h, sample.image_width, sample.image_height
                ):
                    self.stats.dropped_visual_size_mismatch += 1
                    return False, "mask size doesn't align with image"

        elif isinstance(data, np.ndarray):
            if data.size == 0:
                self.stats.dropped_empty_visual += 1
                return False, "mask array is empty"

            # Check size alignment
            if sample.image_width and sample.image_height:
                mask_h, mask_w = data.shape[:2]
                if not self._check_size_alignment(
                    mask_w, mask_h, sample.image_width, sample.image_height
                ):
                    self.stats.dropped_visual_size_mismatch += 1
                    return False, "mask size doesn't align with image"

        else:
            self.stats.dropped_empty_visual += 1
            return False, "mask data is not a path or array"

        return True, "ok"

    def _validate_heatmap(self, data, sample: RawSample) -> Tuple[bool, str]:
        """Validate heatmap data."""
        if isinstance(data, str):
            # Path to heatmap file
            heatmap_path = Path(data)
            if not heatmap_path.exists():
                self.stats.dropped_empty_visual += 1
                return False, "heatmap file not found"

        elif isinstance(data, np.ndarray):
            if data.size == 0:
                self.stats.dropped_empty_visual += 1
                return False, "heatmap array is empty"

            # Check for non-trivial heatmap (not all zeros)
            if np.max(np.abs(data)) < 1e-6:
                self.stats.dropped_empty_visual += 1
                return False, "heatmap is all zeros"

        elif isinstance(data, list):
            # Could be serialized array
            arr = np.array(data)
            if arr.size == 0:
                self.stats.dropped_empty_visual += 1
                return False, "heatmap list is empty"

        else:
            self.stats.dropped_empty_visual += 1
            return False, "heatmap data is not a path, array, or list"

        return True, "ok"

    def _validate_bbox(self, data, sample: RawSample) -> Tuple[bool, str]:
        """Validate bounding box data."""
        # Bbox should be [x, y, w, h] or [[x, y, w, h], ...]
        if not isinstance(data, (list, tuple)):
            self.stats.dropped_empty_visual += 1
            return False, "bbox is not a list/tuple"

        if len(data) == 0:
            self.stats.dropped_empty_visual += 1
            return False, "bbox list is empty"

        # Normalize to list of bboxes
        if isinstance(data[0], (int, float)):
            bboxes = [data]  # Single bbox
        else:
            bboxes = data

        # Validate each bbox
        for bbox in bboxes:
            if len(bbox) != 4:
                self.stats.dropped_empty_visual += 1
                return False, f"bbox has {len(bbox)} elements, expected 4"

            x, y, w, h = bbox

            if w <= 0 or h <= 0:
                self.stats.dropped_empty_visual += 1
                return False, "bbox has non-positive width/height"

            # Check bounds if image dimensions known
            if sample.image_width and sample.image_height:
                if x < 0 or y < 0:
                    self.stats.dropped_bbox_out_of_bounds += 1
                    return False, "bbox has negative coordinates"

                if x + w > sample.image_width or y + h > sample.image_height:
                    self.stats.dropped_bbox_out_of_bounds += 1
                    return False, "bbox extends outside image"

        return True, "ok"

    def _validate_polygon(self, data, sample: RawSample) -> Tuple[bool, str]:
        """Validate polygon data.

        Expected format: [[x1, y1], [x2, y2], ...] or [[[x1, y1], ...], ...]
        """
        if not isinstance(data, (list, tuple)):
            self.stats.dropped_polygon_invalid += 1
            return False, "polygon is not a list/tuple"

        if len(data) == 0:
            self.stats.dropped_polygon_invalid += 1
            return False, "polygon list is empty"

        # Normalize: single polygon vs list of polygons
        if isinstance(data[0], (list, tuple)) and len(data[0]) == 2 and isinstance(data[0][0], (int, float)):
            polygons = [data]  # Single polygon
        else:
            polygons = data

        for poly in polygons:
            if not isinstance(poly, (list, tuple)):
                self.stats.dropped_polygon_invalid += 1
                return False, "polygon entry is not a list"

            if len(poly) < 3:
                self.stats.dropped_polygon_invalid += 1
                return False, f"polygon has {len(poly)} vertices, need at least 3"

            for pt in poly:
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    self.stats.dropped_polygon_invalid += 1
                    return False, "polygon vertex is not [x, y]"

                x, y = pt
                if sample.image_width and sample.image_height:
                    if x < 0 or y < 0:
                        self.stats.dropped_polygon_invalid += 1
                        return False, "polygon vertex has negative coordinates"
                    if x > sample.image_width or y > sample.image_height:
                        self.stats.dropped_polygon_invalid += 1
                        return False, "polygon vertex outside image bounds"

        return True, "ok"

    def _validate_svg(self, data, sample: RawSample) -> Tuple[bool, str]:
        """Validate SVG visual grounding data.

        Expected format: SVG string with <path> elements using M/L/Z commands,
        coordinates in 0-1000 scale.
        """
        if not isinstance(data, str):
            self.stats.dropped_empty_visual += 1
            return False, "svg data is not a string"

        data = data.strip()
        if not data:
            self.stats.dropped_empty_visual += 1
            return False, "svg string is empty"

        # Must contain <svg
        if "<svg" not in data:
            self.stats.dropped_empty_visual += 1
            return False, "svg data missing <svg> element"

        # Extract path d= attributes and validate coordinates
        import re as _re
        paths = _re.findall(r'd="([^"]*)"', data)
        if len(paths) == 0:
            # Empty SVG is valid (no problem regions)
            return True, "ok"

        for path_d in paths:
            # Extract all numbers from the path
            nums = [int(n) for n in _re.findall(r'-?\d+', path_d)]
            for n in nums:
                if n < 0 or n > 1000:
                    self.stats.dropped_polygon_invalid += 1
                    return False, f"svg path coordinate {n} outside 0-1000 range"

        return True, "ok"

    def _check_size_alignment(
        self, data_w: int, data_h: int, img_w: int, img_h: int
    ) -> bool:
        """Check if visual data size aligns with image (or is resizable).

        Primary criterion: matching aspect ratio. ImagenWorld masks are
        regularly drawn at a different resolution than the source image
        (e.g. a 640-px-short-side annotation canvas over a 474x266 output),
        but as long as the aspect ratio matches the mask is for the same
        image and can be losslessly resampled. Only truly misaligned masks
        (different scene, cropped, rotated) have mismatched aspect ratios.
        """
        if data_w <= 0 or data_h <= 0 or img_w <= 0 or img_h <= 0:
            return False

        # Exact match — no resampling needed.
        if data_w == img_w and data_h == img_h:
            return True

        if not self.allow_visual_resize:
            return False

        # Safety net: reject absurd size ratios (likely wrong file entirely).
        w_ratio = max(data_w / img_w, img_w / data_w)
        h_ratio = max(data_h / img_h, img_h / data_h)
        if w_ratio > self.max_resize_ratio or h_ratio > self.max_resize_ratio:
            return False

        # Main check: aspect ratio must match within tolerance.
        ar_img = img_w / img_h
        ar_data = data_w / data_h
        ar_diff = abs(ar_img - ar_data) / ar_img
        return ar_diff <= self.aspect_ratio_tolerance


def validate_samples(
    samples: Iterator[RawSample],
    require_image: bool = True,
    allow_visual_resize: bool = True,
) -> Tuple[List[RawSample], ValidationStats]:
    """
    Convenience function to validate samples.

    Args:
        samples: Iterator of RawSample objects
        require_image: Whether to require valid image paths
        allow_visual_resize: Allow resizable visual evidence

    Returns:
        (list of valid samples, validation stats)
    """
    validator = Validator(
        require_image=require_image,
        allow_visual_resize=allow_visual_resize,
    )
    valid_samples = list(validator.validate(samples))
    return valid_samples, validator.stats
