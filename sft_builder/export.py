"""
Export module for SFT samples.

Handles:
- JSONL export with proper serialization
- Visual data handling (save to file or embed)
- Metadata preservation
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np

from .schema import SFTSample


logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


class Exporter:
    """Exports SFT samples to various formats."""

    def __init__(
        self,
        output_path: Path,
        visual_data_dir: Optional[Path] = None,
        embed_visual_data: bool = False,
        include_orig_meta: bool = True,
    ):
        """
        Initialize exporter.

        Args:
            output_path: Path to output JSONL file
            visual_data_dir: Directory to save visual data files (optional)
            embed_visual_data: Whether to embed visual data as arrays in JSON
            include_orig_meta: Whether to include original metadata
        """
        self.output_path = Path(output_path)
        self.visual_data_dir = Path(visual_data_dir) if visual_data_dir else None
        self.embed_visual_data = embed_visual_data
        self.include_orig_meta = include_orig_meta

        # Create directories
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.visual_data_dir:
            self.visual_data_dir.mkdir(parents=True, exist_ok=True)

        self._count = 0

    def export(self, samples: Iterator[SFTSample]) -> int:
        """
        Export samples to JSONL file.

        Args:
            samples: Iterator of SFTSample objects

        Returns:
            Number of samples exported
        """
        self._count = 0

        with open(self.output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                try:
                    record = self._sample_to_record(sample)
                    line = json.dumps(record, cls=NumpyEncoder, ensure_ascii=False)
                    f.write(line + "\n")
                    self._count += 1

                    if self._count % 100 == 0:
                        logger.info(f"Exported {self._count} samples...")

                except Exception as e:
                    logger.warning(f"Failed to export sample {sample.meta.id}: {e}")
                    continue

        logger.info(f"Exported {self._count} samples to {self.output_path}")
        return self._count

    def _sample_to_record(self, sample: SFTSample) -> Dict[str, Any]:
        """Convert SFTSample to exportable dict."""
        # Handle visual reason data
        visual_data = sample.response.visual_reason.data
        visual_type = sample.response.visual_reason.type

        if isinstance(visual_data, np.ndarray):
            if self.visual_data_dir and not self.embed_visual_data:
                # Save to file
                visual_path = self._save_visual_data(
                    visual_data, sample.meta.id, visual_type
                )
                visual_data = str(visual_path)
            elif self.embed_visual_data:
                # Embed as list (already handled by NumpyEncoder)
                pass
            else:
                # Default: convert to list
                visual_data = visual_data.tolist()

        # Build record
        record = {
            "image": sample.image,
            "instruction": sample.instruction,
            "response": {
                "score": sample.response.score,
                "text_reason": sample.response.text_reason,
                "visual_reason": {
                    "type": visual_type,
                    "data": visual_data,
                },
            },
            "meta": {
                "source": sample.meta.source,
                "id": sample.meta.id,
            },
        }

        if self.include_orig_meta and sample.meta.orig:
            # Filter out large nested objects from orig
            filtered_orig = self._filter_orig_meta(sample.meta.orig)
            if filtered_orig:
                record["meta"]["orig"] = filtered_orig

        return record

    def _save_visual_data(
        self, data: np.ndarray, sample_id: str, vr_type: str
    ) -> Path:
        """Save visual data to file."""
        # Sanitize filename
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in sample_id)

        if vr_type == "mask":
            filename = f"{safe_id}_mask.png"
            path = self.visual_data_dir / filename

            # Save as PNG
            from PIL import Image
            img = Image.fromarray((data * 255).astype(np.uint8))
            img.save(path)

        elif vr_type == "heatmap":
            filename = f"{safe_id}_heatmap.npy"
            path = self.visual_data_dir / filename
            np.save(path, data)

        else:
            filename = f"{safe_id}_visual.npy"
            path = self.visual_data_dir / filename
            np.save(path, data)

        return path

    def _filter_orig_meta(self, orig: Dict[str, Any]) -> Dict[str, Any]:
        """Filter original metadata to keep only essential fields."""
        # Skip large objects and nested dicts with more than 5 items
        filtered = {}
        max_items = 10
        max_str_len = 500

        for key, value in orig.items():
            if key in ("raw_scores", "task", "condition", "model", "annotator"):
                # Keep these fields
                if isinstance(value, str) and len(value) > max_str_len:
                    value = value[:max_str_len] + "..."
                filtered[key] = value
            elif isinstance(value, dict) and len(value) <= max_items:
                # Keep small dicts
                filtered[key] = value
            elif isinstance(value, (int, float, bool)):
                filtered[key] = value
            elif isinstance(value, str) and len(value) <= max_str_len:
                filtered[key] = value
            # Skip large objects

        return filtered


def export_samples(
    samples: Iterator[SFTSample],
    output_path: Path,
    visual_data_dir: Optional[Path] = None,
) -> int:
    """
    Convenience function to export samples.

    Args:
        samples: Iterator of SFTSample objects
        output_path: Path to output JSONL file
        visual_data_dir: Optional directory for visual data files

    Returns:
        Number of samples exported
    """
    exporter = Exporter(
        output_path=output_path,
        visual_data_dir=visual_data_dir,
    )
    return exporter.export(samples)


def load_samples(input_path: Path) -> Iterator[SFTSample]:
    """
    Load SFT samples from JSONL file.

    Args:
        input_path: Path to JSONL file

    Yields:
        SFTSample objects
    """
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                yield SFTSample.from_dict(record)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse line: {e}")
                continue
