"""
RichHF-18K dataset ingestor.

RichHF provides:
- aesthetics_score, artifact_score, misalignment_score, overall_score
- artifact_map: heatmap for artifact regions
- misalignment_map: heatmap for misalignment regions
- prompt_misalignment_label: token-level misalignment labels

Supports two formats:
1. HuggingFace format (Exploration/richhf_18k_with_images) - preferred
2. Legacy TFRecord format (google-research-datasets/richhf-18k)
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Any
import json

import numpy as np

from .base import BaseIngestor
from ..schema import RawSample


logger = logging.getLogger(__name__)


class RichHFIngestor(BaseIngestor):
    """Ingestor for RichHF-18K dataset.

    RichHF-18K contains real-world generated images with fine-grained human
    feedback including aesthetics, artifact, misalignment and overall scores,
    plus optional heatmaps for artifact/misalignment regions.
    """

    def __init__(
        self,
        data_root: Path,
        max_samples: Optional[int] = None,
        split: str = "train",
        prompt_mapping_path: Optional[Path] = None,
        require_visual: bool = False,
        image_save_dir: Optional[Path] = None,
    ):
        """
        Initialize RichHF ingestor.

        Args:
            data_root: Root directory containing the dataset
            max_samples: Maximum samples to ingest
            split: Dataset split (train, dev, test)
            prompt_mapping_path: Path to filename->prompt JSON mapping (for legacy format)
            require_visual: If True, skip samples without heatmap data (legacy behaviour).
                            If False (default), include text-only samples too.
            image_save_dir: Directory to save PIL images to disk (for HF format).
                            Defaults to data_root / "images".
        """
        super().__init__(data_root, max_samples)
        self.split = split
        self.prompt_mapping_path = prompt_mapping_path
        self.require_visual = require_visual
        self.image_save_dir = Path(image_save_dir) if image_save_dir else (data_root / "images")
        self._prompt_mapping: Optional[Dict[str, str]] = None

    @property
    def source_name(self) -> str:
        return "RichHF-18K"

    def iterate(self) -> Iterator[RawSample]:
        """Iterate over RichHF samples."""
        # Try HuggingFace format first (preferred)
        if self._has_huggingface_format():
            logger.info("Using HuggingFace format for RichHF-18K")
            yield from self._iterate_huggingface()
        # Fall back to legacy TFRecord format
        elif self._has_tfrecord_format():
            logger.info("Using legacy TFRecord format for RichHF-18K")
            yield from self._iterate_tfrecord()
        else:
            logger.warning(f"RichHF-18K not found at {self.data_root}")
            logger.info("Download with: python download_data.py --datasets richhf")

    def _has_huggingface_format(self) -> bool:
        """Check if HuggingFace format is available."""
        return any(self.data_root.glob("**/*.parquet"))

    def _has_tfrecord_format(self) -> bool:
        """Check if legacy TFRecord format is available."""
        return (self.data_root / f"{self.split}.tfrecord").exists()

    def _iterate_huggingface(self) -> Iterator[RawSample]:
        """Iterate using HuggingFace datasets format."""
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed. Install with: pip install datasets")
            return

        try:
            # Load from local directory
            ds = load_dataset(
                str(self.data_root),
                split=self.split,
            )
        except Exception as e:
            logger.warning(f"Failed to load RichHF split '{self.split}' from local: {e}")
            # Try without explicit split (some datasets have only one split)
            try:
                ds = load_dataset(str(self.data_root))
                # Get first available split
                split_name = list(ds.keys())[0]
                logger.info(f"Using split '{split_name}' from local RichHF data")
                ds = ds[split_name]
            except Exception:
                # Try loading directly from HuggingFace Hub
                try:
                    ds = load_dataset(
                        "Exploration/richhf_18k_with_images",
                        split=self.split,
                    )
                except Exception as e2:
                    logger.error(f"Failed to load from HuggingFace Hub: {e2}")
                    return

        logger.info(f"Loaded {len(ds)} samples from RichHF-18K")

        for i, ex in enumerate(ds):
            if self._should_stop():
                break

            sample = self._process_hf_example(ex, i)
            if sample is not None:
                yield sample
                self._increment()

    @staticmethod
    def _to_unit(val: Any) -> Optional[float]:
        """Coerce a RichHF sub-score into the [0, 1] range.

        The HuggingFace mirror (Exploration/richhf_18k_with_images) stores
        sub-scores in 0-1, while the original TFRecord release uses the
        paper's 1-5 scale. We auto-detect which scale the value is on so
        downstream code can treat everything uniformly.
        """
        if val is None:
            return None
        try:
            v = float(val)
        except (TypeError, ValueError):
            return None
        if v <= 1.0:
            return max(0.0, v)
        if v <= 5.0:
            return (v - 1.0) / 4.0
        return min(1.0, v / 5.0)

    def _save_map_sidecar(self, arr: Any, filename: Any, tag: str) -> Optional[str]:
        """Save a continuous heatmap as an 8-bit grayscale PNG sidecar and
        return its path, or None when the map is missing or all-zero.

        The normalizer binarizes ``visual_reason`` into a single mask, which
        destroys both the channel identity (artifact vs misalignment) and the
        intensity needed for severity levels — so the raw maps must leave the
        ingestor through a side channel. RichHF annotates BOTH channels on
        every sample, so a missing sidecar downstream means "channel clean",
        not "channel unknown". 1/255 intensity resolution is ample for the
        builder's problem/severity thresholds.
        """
        if arr is None or getattr(arr, "size", 0) == 0 or arr.ndim != 2:
            return None
        if float(np.max(arr)) <= 0.0:
            return None
        from PIL import Image
        stem = Path(str(filename))
        if stem.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            stem = stem.with_suffix("")
        out = self.data_root / "sidecar_maps" / f"{stem}_{tag}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            try:
                a = np.clip(arr.astype(np.float32), 0.0, 1.0)
                Image.fromarray((a * 255).astype(np.uint8), mode="L").save(out)
            except Exception as e:
                logger.warning(f"Failed to save {tag} sidecar for {filename}: {e}")
                return None
        return str(out)

    @classmethod
    def _synthesize_overall(
        cls,
        aesthetics: Any,
        artifact: Any,
        misalignment: Any,
        raw_overall: Any,
    ) -> Tuple[float, bool]:
        """Return (overall_in_unit, was_synthesized).

        RichHF's overall_score is missing on the HuggingFace mirror (returns 0).
        When that happens, fall back to averaging the sub-scores so eval still
        sees variance in the GT distribution. All sub-scores follow the
        "higher = better" convention (high artifact score = few artifacts).
        """
        overall_unit = cls._to_unit(raw_overall)
        if overall_unit is not None and overall_unit > 0.0:
            return overall_unit, False

        parts = [
            cls._to_unit(v)
            for v in (aesthetics, artifact, misalignment)
        ]
        parts = [p for p in parts if p is not None]
        if not parts:
            return 0.5, True
        return sum(parts) / len(parts), True

    def _process_hf_example(self, ex: Dict[str, Any], idx: int) -> Optional[RawSample]:
        """Process a single HuggingFace example."""
        # Extract sub-scores. The HF mirror reports them in 0-1, the legacy
        # TFRecord release uses 1-5 — _to_unit normalises both to 0-1.
        aesthetics_raw = ex.get("aesthetics_score", 0)
        artifact_raw = ex.get("artifact_score", 0)
        misalignment_raw = ex.get("misalignment_score", 0)
        overall_raw = ex.get("overall_score", 0)

        aesthetics = self._to_unit(aesthetics_raw) or 0.0
        artifact = self._to_unit(artifact_raw) or 0.0
        misalignment = self._to_unit(misalignment_raw) or 0.0
        overall, overall_synth = self._synthesize_overall(
            aesthetics_raw, artifact_raw, misalignment_raw, overall_raw,
        )

        # Get prompt
        prompt = ex.get("prompt", ex.get("caption", ""))

        # Get image
        image = ex.get("image")
        image_path = None
        image_width, image_height = None, None

        if image is not None:
            # PIL Image object from HuggingFace — save to disk
            if hasattr(image, "size"):
                image_width, image_height = image.size
                filename = ex.get("filename", ex.get("image_id", f"sample_{idx}"))
                # Strip any existing image extension so we don't end up with
                # "foo.png.png" when the HF 'filename' field already carries it.
                stem = Path(str(filename))
                if stem.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                    stem = stem.with_suffix("")
                save_path = self.image_save_dir / f"{stem}.png"
                # Ensure the full parent chain exists (filename may contain a
                # split subdir like "train/<uuid>").
                save_path.parent.mkdir(parents=True, exist_ok=True)
                if not save_path.exists():
                    try:
                        image.save(str(save_path))
                    except Exception as e:
                        logger.warning(f"Failed to save image {save_path}: {e}")
                image_path = str(save_path)
            # Could be a path string
            elif isinstance(image, str):
                image_path = image

        # Get heatmaps
        artifact_map = ex.get("artifact_map")
        misalignment_map = ex.get("misalignment_map")

        # Convert to numpy if needed
        if artifact_map is not None and not isinstance(artifact_map, np.ndarray):
            artifact_map = np.array(artifact_map)
        if misalignment_map is not None and not isinstance(misalignment_map, np.ndarray):
            misalignment_map = np.array(misalignment_map)

        # Choose the most informative heatmap
        visual_reason_data = None
        visual_reason_type = None

        if artifact_map is not None and artifact_map.size > 0 and np.max(artifact_map) > 0.1:
            visual_reason_data = artifact_map
            visual_reason_type = "heatmap"
        elif misalignment_map is not None and misalignment_map.size > 0 and np.max(misalignment_map) > 0.1:
            visual_reason_data = misalignment_map
            visual_reason_type = "heatmap"

        # Skip if no visual evidence and require_visual is set
        if visual_reason_data is None and self.require_visual:
            return None

        # Persist BOTH raw maps as PNG sidecars (channel + intensity survive
        # even though visual_reason above keeps only one binarized map).
        filename = ex.get("filename", ex.get("image_id", f"sample_{idx}"))
        artifact_png = self._save_map_sidecar(artifact_map, filename, "artifact")
        misalign_png = self._save_map_sidecar(misalignment_map, filename, "misalign")

        # Build text reason. Sub-scores are now in 0-1 range, so a "low quality"
        # threshold of 0.5 mirrors the old "<3 on a 1-5 scale" check.
        text_parts = []
        if artifact < 0.5:
            text_parts.append(f"Notable artifacts present (score: {artifact:.2f})")
        if misalignment < 0.5:
            text_parts.append(f"Text-image misalignment detected (score: {misalignment:.2f})")
        if aesthetics < 0.5:
            text_parts.append(f"Below average aesthetics (score: {aesthetics:.2f})")

        if not text_parts:
            if overall < 0.7:
                text_parts.append(f"Moderate overall quality (score: {overall:.2f})")
            else:
                text_parts.append("Good quality across dimensions")

        text_reason = "; ".join(text_parts)

        orig_meta = {
            "filename": filename,
            "aesthetics_score": aesthetics,
            "artifact_score": artifact,
            "misalignment_score": misalignment,
            "overall_score": overall,
            "overall_synthesized": overall_synth,
        }
        if artifact_png:
            orig_meta["artifact_map_png"] = artifact_png
        if misalign_png:
            orig_meta["misalign_map_png"] = misalign_png

        # `score` is in [0, 1]; the normalizer's RichHF range is also (0, 1)
        # so this passes through unchanged into the final SFTSample.
        return RawSample(
            image=image_path,
            instruction=prompt if prompt else None,
            score=overall,
            text_reason=text_reason,
            visual_reason_type=visual_reason_type,
            visual_reason_data=visual_reason_data,
            source="RichHF-18K",
            id=f"richhf_{idx}_{filename}",
            orig=orig_meta,
            image_width=image_width,
            image_height=image_height,
        )

    def _iterate_tfrecord(self) -> Iterator[RawSample]:
        """Iterate using legacy TFRecord format."""
        tfrecord_path = self.data_root / f"{self.split}.tfrecord"

        try:
            import tensorflow as tf
        except ImportError:
            logger.warning("TensorFlow not installed. Cannot load legacy RichHF-18K format.")
            logger.info("Install with: pip install tensorflow")
            logger.info("Or download the HuggingFace version: python download_data.py --datasets richhf")
            return

        feature_description = {
            "filename": tf.io.FixedLenFeature([], tf.string),
            "aesthetics_score": tf.io.FixedLenFeature([], tf.float32),
            "artifact_score": tf.io.FixedLenFeature([], tf.float32),
            "misalignment_score": tf.io.FixedLenFeature([], tf.float32),
            "overall_score": tf.io.FixedLenFeature([], tf.float32),
            "artifact_map": tf.io.VarLenFeature(tf.float32),
            "misalignment_map": tf.io.VarLenFeature(tf.float32),
        }

        def parse_example(example_proto):
            return tf.io.parse_single_example(example_proto, feature_description)

        prompt_mapping = self._load_prompt_mapping()

        try:
            raw_ds = tf.data.TFRecordDataset(str(tfrecord_path)).map(parse_example)
            score_ranges = self._compute_score_ranges(raw_ds)
        except Exception as e:
            logger.error(f"Failed to read RichHF TFRecord: {e}")
            logger.info("Try downloading the HuggingFace version: python download_data.py --datasets richhf")
            return

        try:
            raw_ds = tf.data.TFRecordDataset(str(tfrecord_path)).map(parse_example)

            for i, ex in enumerate(raw_ds):
                if self._should_stop():
                    break

                sample = self._process_tfrecord_example(ex, i, prompt_mapping, score_ranges)
                if sample is not None:
                    yield sample
                    self._increment()
        except Exception as e:
            logger.error(f"Error reading RichHF records: {e}")
            return

    def _load_prompt_mapping(self) -> Dict[str, str]:
        """Load filename to prompt mapping."""
        if self._prompt_mapping is not None:
            return self._prompt_mapping

        if self.prompt_mapping_path and self.prompt_mapping_path.exists():
            try:
                with open(self.prompt_mapping_path, "r") as f:
                    self._prompt_mapping = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load prompt mapping: {e}")
                self._prompt_mapping = {}
        else:
            self._prompt_mapping = {}

        return self._prompt_mapping

    def _compute_score_ranges(self, ds) -> Dict[str, Tuple[float, float]]:
        """Compute min/max for each score type."""
        ranges = {
            "aesthetics": [float("inf"), float("-inf")],
            "artifact": [float("inf"), float("-inf")],
            "misalignment": [float("inf"), float("-inf")],
            "overall": [float("inf"), float("-inf")],
        }

        for ex in ds.take(5000):
            for key in ["aesthetics", "artifact", "misalignment", "overall"]:
                score_key = f"{key}_score"
                val = float(ex[score_key].numpy())
                ranges[key][0] = min(ranges[key][0], val)
                ranges[key][1] = max(ranges[key][1], val)

        return {k: tuple(v) for k, v in ranges.items()}

    def _process_tfrecord_example(
        self,
        ex,
        idx: int,
        prompt_mapping: Dict[str, str],
        score_ranges: Dict[str, Tuple[float, float]],
    ) -> Optional[RawSample]:
        """Process a single TFRecord example."""
        import tensorflow as tf

        ex_dict = {k: v.numpy() for k, v in ex.items()}
        filename = ex_dict["filename"].decode("utf-8")

        aesthetics_raw = float(ex_dict["aesthetics_score"])
        artifact_raw = float(ex_dict["artifact_score"])
        misalignment_raw = float(ex_dict["misalignment_score"])
        overall_raw = float(ex_dict["overall_score"])

        aesthetics = self._to_unit(aesthetics_raw) or 0.0
        artifact = self._to_unit(artifact_raw) or 0.0
        misalignment = self._to_unit(misalignment_raw) or 0.0
        overall, overall_synth = self._synthesize_overall(
            aesthetics_raw, artifact_raw, misalignment_raw, overall_raw,
        )

        artifact_map = tf.sparse.to_dense(ex["artifact_map"]).numpy()
        misalignment_map = tf.sparse.to_dense(ex["misalignment_map"]).numpy()

        visual_reason_data = None
        visual_reason_type = None

        if artifact_map.size > 0 and np.max(artifact_map) > 0.1:
            visual_reason_data = artifact_map
            visual_reason_type = "heatmap"
        elif misalignment_map.size > 0 and np.max(misalignment_map) > 0.1:
            visual_reason_data = misalignment_map
            visual_reason_type = "heatmap"

        if visual_reason_data is None and self.require_visual:
            return None

        prompt = prompt_mapping.get(filename, "")

        text_parts = []
        if artifact < 0.5:
            text_parts.append(f"Notable artifacts present (score: {artifact:.2f})")
        if misalignment < 0.5:
            text_parts.append(f"Text-image misalignment detected (score: {misalignment:.2f})")
        if aesthetics < 0.5:
            text_parts.append(f"Below average aesthetics (score: {aesthetics:.2f})")

        if not text_parts:
            if overall < 0.7:
                text_parts.append(f"Moderate overall quality (score: {overall:.2f})")
            else:
                text_parts.append("Good quality across dimensions")

        text_reason = "; ".join(text_parts)

        return RawSample(
            image=None,
            instruction=prompt if prompt else None,
            score=overall,
            text_reason=text_reason,
            visual_reason_type=visual_reason_type,
            visual_reason_data=visual_reason_data,
            source="RichHF-18K",
            id=f"richhf_{idx}_{filename}",
            orig={
                "filename": filename,
                "aesthetics_score": aesthetics,
                "artifact_score": artifact,
                "misalignment_score": misalignment,
                "overall_score": overall,
                "overall_synthesized": overall_synth,
                "score_ranges": score_ranges,
            },
            image_width=None,
            image_height=None,
        )
