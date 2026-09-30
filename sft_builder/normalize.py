"""
Normalization module for SFT samples.

Handles:
- Score normalization to configurable range (default [0, 1])
- Visual evidence canonicalization to one format (mask or heatmap)
- Bbox to mask/heatmap conversion
"""

import logging
from typing import Iterator, Literal, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .schema import RawSample, SFTSample, Response, VisualReason, Meta


logger = logging.getLogger(__name__)

# Visual format to canonicalize to
CanonicalVisualFormat = Literal["mask", "heatmap"]


class ScoreNormalizer:
    """Normalizes scores to a target range."""

    def __init__(
        self,
        target_min: float = 0.0,
        target_max: float = 1.0,
        source_min: Optional[float] = None,
        source_max: Optional[float] = None,
        auto_detect: bool = True,
    ):
        """
        Initialize score normalizer.

        Args:
            target_min: Target range minimum
            target_max: Target range maximum
            source_min: Known source range minimum
            source_max: Known source range maximum
            auto_detect: Auto-detect source range from scores
        """
        self.target_min = target_min
        self.target_max = target_max
        self.source_min = source_min
        self.source_max = source_max
        self.auto_detect = auto_detect

    def normalize(self, score: float, source_info: Optional[str] = None) -> float:
        """
        Normalize a score to target range.

        Args:
            score: Raw score value
            source_info: Optional source identifier for scale detection

        Returns:
            Normalized score in [target_min, target_max]
        """
        src_min, src_max = self._get_source_range(score, source_info)

        if src_max <= src_min:
            # Can't normalize, return middle of target range
            return (self.target_min + self.target_max) / 2

        # Min-max normalization
        normalized = (score - src_min) / (src_max - src_min)
        scaled = normalized * (self.target_max - self.target_min) + self.target_min

        # Clamp to target range
        return max(self.target_min, min(self.target_max, scaled))

    def _get_source_range(
        self, score: float, source_info: Optional[str]
    ) -> Tuple[float, float]:
        """Determine source range for normalization."""
        if self.source_min is not None and self.source_max is not None:
            return self.source_min, self.source_max

        if not self.auto_detect:
            return 0.0, 1.0

        # Auto-detect based on score value and source
        if source_info:
            source_lower = source_info.lower()

            # RichHF: ingest layer already coerces sub-scores into [0, 1]
            # via _to_unit and synthesizes overall_score the same way, so
            # nothing else to rescale here. Treating it as 1-5 (the old
            # default) clamped every score to 0 because incoming values
            # like 0.7 fell below the source minimum.
            if "richhf" in source_lower:
                return 0.0, 1.0

            # ImagenWorld uses 1-5 scale
            if "imagenworld" in source_lower:
                return 1.0, 5.0

            # COCO-real: perfect score on 1-5 scale
            if "coco" in source_lower:
                return 1.0, 5.0

        # Heuristic: if score > 10, assume 0-100 scale
        if score > 10:
            return 0.0, 100.0

        # Default: assume 0-5 scale (common for ratings)
        return 0.0, 5.0


class VisualNormalizer:
    """Normalizes visual evidence to a canonical format."""

    def __init__(
        self,
        target_format: CanonicalVisualFormat = "mask",
        output_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize visual normalizer.

        Args:
            target_format: Target format for visual evidence ("mask" or "heatmap")
            output_size: Optional fixed output size (width, height)
        """
        self.target_format = target_format
        self.output_size = output_size

    def normalize(
        self,
        vr_type: str,
        vr_data,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[str, Union[str, np.ndarray]]:
        """
        Normalize visual evidence to target format.

        Args:
            vr_type: Input visual reason type
            vr_data: Input visual reason data
            image_size: Image dimensions (width, height) for reference

        Returns:
            (normalized_type, normalized_data)
        """
        # SVG data is a string but not a file path — pass through as-is
        if vr_type == "svg":
            return vr_type, vr_data

        # Load data if it's a file path
        if isinstance(vr_data, str):
            loaded_data, loaded_size = self._load_visual_data(vr_data, vr_type)
            if loaded_data is None:
                return vr_type, vr_data  # Return as-is if can't load
            vr_data = loaded_data
            if image_size is None:
                image_size = loaded_size

        # Convert to target format
        if vr_type == "bbox":
            vr_data = self._bbox_to_target(vr_data, image_size)
            vr_type = self.target_format

        elif vr_type == "mask" and self.target_format == "heatmap":
            vr_data = self._mask_to_heatmap(vr_data)
            vr_type = "heatmap"

        elif vr_type == "heatmap" and self.target_format == "mask":
            vr_data = self._heatmap_to_mask(vr_data)
            vr_type = "mask"

        # Resize if needed
        if self.output_size and isinstance(vr_data, np.ndarray):
            vr_data = self._resize_array(vr_data, self.output_size)

        return vr_type, vr_data

    def _load_visual_data(
        self, path: str, vr_type: str
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Load visual data from a file path."""
        try:
            img = Image.open(path)
            size = img.size  # (width, height)

            if vr_type in ("mask", "heatmap"):
                # Convert to grayscale array
                arr = np.array(img.convert("L")).astype(np.float32) / 255.0
                return arr, size

        except Exception as e:
            logger.warning(f"Failed to load visual data from {path}: {e}")

        return None, None

    def _bbox_to_target(
        self,
        bbox_data,
        image_size: Optional[Tuple[int, int]],
    ) -> np.ndarray:
        """Convert bbox(es) to mask or heatmap."""
        if image_size is None:
            # Default size if unknown
            image_size = (512, 512)

        w, h = image_size
        canvas = np.zeros((h, w), dtype=np.float32)

        # Normalize to list of bboxes
        if isinstance(bbox_data[0], (int, float)):
            bboxes = [bbox_data]
        else:
            bboxes = bbox_data

        for bbox in bboxes:
            x, y, bw, bh = [int(v) for v in bbox]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            canvas[y1:y2, x1:x2] = 1.0

        if self.target_format == "heatmap":
            # Apply gaussian blur for smoother heatmap
            canvas = self._apply_gaussian_blur(canvas)

        return canvas

    def _mask_to_heatmap(self, mask: np.ndarray) -> np.ndarray:
        """Convert binary mask to smooth heatmap."""
        return self._apply_gaussian_blur(mask.astype(np.float32))

    def _heatmap_to_mask(self, heatmap: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Convert heatmap to binary mask."""
        return (heatmap > threshold).astype(np.float32)

    def _apply_gaussian_blur(
        self, arr: np.ndarray, sigma: float = 5.0
    ) -> np.ndarray:
        """Apply gaussian blur to array."""
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(arr, sigma=sigma)
        except ImportError:
            # Fallback: simple box blur
            return arr

    def _resize_array(
        self, arr: np.ndarray, size: Tuple[int, int]
    ) -> np.ndarray:
        """Resize array to target size."""
        try:
            img = Image.fromarray((arr * 255).astype(np.uint8))
            img = img.resize(size, Image.BILINEAR)
            return np.array(img).astype(np.float32) / 255.0
        except Exception:
            return arr


class Normalizer:
    """Combined normalizer for scores and visual evidence."""

    def __init__(
        self,
        score_range: Tuple[float, float] = (0.0, 1.0),
        visual_format: CanonicalVisualFormat = "mask",
        visual_output_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize combined normalizer.

        Args:
            score_range: Target score range (min, max)
            visual_format: Target visual format
            visual_output_size: Optional fixed output size for visual data
        """
        self.score_normalizer = ScoreNormalizer(
            target_min=score_range[0],
            target_max=score_range[1],
        )
        self.visual_normalizer = VisualNormalizer(
            target_format=visual_format,
            output_size=visual_output_size,
        )

    def normalize(self, sample: RawSample) -> SFTSample:
        """
        Normalize a raw sample to final SFT format.

        Args:
            sample: Validated RawSample

        Returns:
            Normalized SFTSample
        """
        # Normalize score
        normalized_score = self.score_normalizer.normalize(
            sample.score, sample.source
        )

        # Normalize visual evidence
        image_size = None
        if sample.image_width and sample.image_height:
            image_size = (sample.image_width, sample.image_height)

        vr_type, vr_data = self.visual_normalizer.normalize(
            sample.visual_reason_type,
            sample.visual_reason_data,
            image_size,
        )

        # Build final sample
        return SFTSample(
            image=sample.image or "",
            instruction=sample.instruction or "",
            response=Response(
                score=normalized_score,
                text_reason=sample.text_reason,
                visual_reason=VisualReason(type=vr_type, data=vr_data),
            ),
            meta=Meta(
                source=sample.source,
                id=sample.id,
                orig=sample.orig,
            ),
        )

    def normalize_batch(
        self, samples: Iterator[RawSample]
    ) -> Iterator[SFTSample]:
        """
        Normalize a batch of samples.

        Args:
            samples: Iterator of RawSample objects

        Yields:
            Normalized SFTSample objects
        """
        for sample in samples:
            try:
                yield self.normalize(sample)
            except Exception as e:
                logger.warning(f"Failed to normalize sample {sample.id}: {e}")
                continue
