"""
ImagenWorld dataset ingestor.

ImagenWorld structure:
    train/{TASK}/{TASK}_{TOPIC}_{ID}/
        input/
            metadata.json  (prompt + context)
        outputs/
            {model_name}/
                out.png
                {annotator}/
                    evaluation.json
                    error_mask.png (optional)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from PIL import Image

from .base import BaseIngestor
from ..schema import RawSample


logger = logging.getLogger(__name__)

# Task types
IMAGENWORLD_TASKS = ["TIG", "TIE", "SRIG", "SRIE", "MRIG", "MRIE"]

# Model preference (higher priority first)
MODEL_PREFERENCE = [
    "gpt-image-1",
    "gemini",
    "omnigen2",
    "bagel",
    "sdxl",
    "flux1kreadev",
    "infinity",
    "uno",
    "januspro",
    "qwenimage",
]

# Annotator preference
ANNOTATOR_PREFERENCE = ["annotator1", "annotator2", "annotator3"]


class ImagenWorldIngestor(BaseIngestor):
    """Ingestor for ImagenWorld-annotated-set."""

    def __init__(
        self,
        data_root: Path,
        max_samples: Optional[int] = None,
        max_per_task: Optional[int] = None,
        require_error_mask: bool = True,
        tasks: Optional[List[str]] = None,
    ):
        """
        Initialize ImagenWorld ingestor.

        Args:
            data_root: Root of ImagenWorld (containing train/)
            max_samples: Global max samples
            max_per_task: Max samples per task type
            require_error_mask: Only include samples with error masks (default True per spec)
            tasks: List of tasks to include (default: all)
        """
        super().__init__(data_root, max_samples)
        self.max_per_task = max_per_task
        self.require_error_mask = require_error_mask
        self.tasks = tasks or IMAGENWORLD_TASKS

    @property
    def source_name(self) -> str:
        return "ImagenWorld"

    def iterate(self) -> Iterator[RawSample]:
        """Iterate over ImagenWorld samples."""
        train_root = self.data_root / "train"
        if not train_root.exists():
            logger.warning(f"ImagenWorld train dir not found: {train_root}")
            return

        for task in self.tasks:
            if self._should_stop():
                break

            task_count = 0
            task_root = train_root / task
            cond_dirs = self._list_condition_dirs(task_root)

            for cond in cond_dirs:
                if self._should_stop():
                    break
                if self.max_per_task and task_count >= self.max_per_task:
                    break

                samples = list(self._process_condition(cond, task))
                for sample in samples:
                    if self._should_stop():
                        break
                    if self.max_per_task and task_count >= self.max_per_task:
                        break

                    yield sample
                    self._increment()
                    task_count += 1

    def _list_condition_dirs(self, task_dir: Path) -> List[Path]:
        """List condition directories, handling nested structure."""
        if not task_dir.exists():
            return []

        subdirs = [p for p in task_dir.iterdir() if p.is_dir()]

        # Handle extra nested layer (train/TIG/TIG/TIG_A_xxx)
        if len(subdirs) == 1 and subdirs[0].name == task_dir.name:
            task_dir = subdirs[0]
            subdirs = [p for p in task_dir.iterdir() if p.is_dir()]

        return sorted(subdirs)

    def _process_condition(self, cond: Path, task: str) -> Iterator[RawSample]:
        """Process a single condition directory."""
        # Read metadata
        meta_path = cond / "input" / "metadata.json"
        if not meta_path.exists():
            return

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.debug(f"Failed to read metadata: {meta_path}: {e}")
            return

        prompt = self._extract_prompt(meta)
        if not prompt:
            return

        # Find all evaluations with error masks
        outputs_root = cond / "outputs"
        if not outputs_root.exists():
            return

        for result in self._find_evaluations(outputs_root):
            if result is None:
                continue

            img_path, eval_path, error_mask_path, model_name, annotator = result

            # Skip if no error mask and we require it
            if self.require_error_mask and error_mask_path is None:
                continue

            # Parse evaluation
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    ev = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            score, text_reason, raw_scores = self._parse_evaluation(ev, meta)

            if score is None:
                continue

            # Get image dimensions
            img_width, img_height = None, None
            if img_path.exists():
                try:
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                except Exception:
                    pass

            # Build sample ID
            sample_id = f"{cond.name}_{model_name}_{annotator}"

            yield RawSample(
                image=str(img_path) if img_path.exists() else None,
                instruction=prompt,
                score=score,
                text_reason=text_reason,
                visual_reason_type="mask" if error_mask_path else None,
                visual_reason_data=str(error_mask_path) if error_mask_path else None,
                source=f"ImagenWorld-{task}",
                id=sample_id,
                orig={
                    "task": task,
                    "condition": cond.name,
                    "model": model_name,
                    "annotator": annotator,
                    "raw_scores": raw_scores,
                    "metadata": meta,
                },
                image_width=img_width,
                image_height=img_height,
            )

    def _find_evaluations(
        self, outputs_root: Path
    ) -> Iterator[Optional[Tuple[Path, Path, Optional[Path], str, str]]]:
        """
        Find all evaluations, ranked by model/annotator preference.

        Yields:
            (image_path, eval_path, error_mask_path or None, model_name, annotator)
        """
        candidates = []

        for model_dir in outputs_root.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            for ann_dir in model_dir.iterdir():
                if not ann_dir.is_dir():
                    continue
                annotator = ann_dir.name
                if not annotator.startswith("annotator"):
                    continue

                eval_path = ann_dir / "evaluation.json"
                if not eval_path.exists():
                    continue

                # Find image
                img_path = model_dir / "out.png"
                if not img_path.exists():
                    img_path = ann_dir / "out.png"

                # Check for error mask
                error_mask_path = ann_dir / "error_mask.png"
                if not error_mask_path.exists():
                    error_mask_path = None

                candidates.append(
                    (model_name, annotator, img_path, eval_path, error_mask_path)
                )

        # Sort by preference
        def rank_candidate(c):
            m, a = c[0], c[1]
            try:
                m_rank = MODEL_PREFERENCE.index(m)
            except ValueError:
                m_rank = len(MODEL_PREFERENCE)
            try:
                a_rank = ANNOTATOR_PREFERENCE.index(a)
            except ValueError:
                a_rank = len(ANNOTATOR_PREFERENCE)
            # Prefer samples with error masks
            has_mask = 0 if c[4] else 1
            return (has_mask, m_rank, a_rank)

        for c in sorted(candidates, key=rank_candidate):
            yield (c[2], c[3], c[4], c[0], c[1])

    def _extract_prompt(self, meta: Dict[str, Any]) -> str:
        """Extract prompt from metadata."""
        candidate_keys = ["prompt", "instruction", "text", "input_text", "caption"]

        for key in candidate_keys:
            if key in meta and isinstance(meta[key], str) and meta[key].strip():
                return meta[key].strip()

        return ""

    def _parse_evaluation(
        self, ev: Dict[str, Any], meta: Dict[str, Any]
    ) -> Tuple[Optional[float], Optional[str], Dict[str, Any]]:
        """
        Parse evaluation.json to extract score and text_reason.

        Returns:
            (overall_score, text_reason, raw_scores_dict)
        """
        raw_scores = {}
        ratings = {}
        issues = []

        # Extract from annotation result
        annotation = ev.get("annotation", {})
        results = annotation.get("result", [])

        for item in results:
            item_type = item.get("type")
            from_name = item.get("from_name", "")
            value = item.get("value", {})

            if item_type == "rating":
                rating = value.get("rating")
                if rating is not None:
                    ratings[from_name] = rating
                    raw_scores[from_name] = rating

            elif item_type == "choices":
                choices = value.get("choices", [])
                if choices and from_name == "object_issues":
                    # Object issues point to problematic objects
                    issues.extend(choices)
                elif choices and from_name == "segmentation_issues":
                    # Segment IDs with issues
                    raw_scores["segmentation_issues"] = choices

        if not ratings:
            return None, None, {}

        # Compute overall score (average of ratings, scale 1-5)
        overall = sum(ratings.values()) / len(ratings)

        # Build text_reason from issues and low-scoring dimensions
        text_parts = []

        # Check for low scores
        low_threshold = 3
        for dim, score in ratings.items():
            if score <= low_threshold:
                dim_readable = dim.replace("_", " ")
                text_parts.append(f"Low {dim_readable} (score: {score}/5)")

        # Add object issues
        if issues:
            # Filter out generic "None of the objects" type answers
            real_issues = [
                i for i in issues
                if "none" not in i.lower() and "no issues" not in i.lower()
            ]
            if real_issues:
                objects_str = ", ".join(real_issues[:5])  # Limit to 5
                text_parts.append(f"Issues with: {objects_str}")

        # If no issues found but score is not perfect, note the areas
        if not text_parts and overall < 5.0:
            # Find lowest scoring dimension
            lowest_dim = min(ratings, key=ratings.get)
            lowest_score = ratings[lowest_dim]
            dim_readable = lowest_dim.replace("_", " ")
            text_parts.append(
                f"Weakest aspect: {dim_readable} ({lowest_score}/5)"
            )

        # Build final text_reason
        if text_parts:
            text_reason = "; ".join(text_parts)
        else:
            # Perfect score - still need a reason
            text_reason = "Excellent quality across all dimensions"

        return overall, text_reason, raw_scores
