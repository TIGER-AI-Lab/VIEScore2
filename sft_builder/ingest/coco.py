"""
COCO dataset ingestor for real-world "no error" samples.

Uses COCO images (real photographs) as perfect-quality samples with
score = 5.0 (max on 1-5 scale) and empty visual grounding (no problem
regions).  COCO captions serve as the instruction/prompt.

Supports loading from:
1. Local directory with saved images + captions JSON
2. HuggingFace Hub (auto-downloads and caches)
"""

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .base import BaseIngestor
from ..schema import RawSample


logger = logging.getLogger(__name__)


# Fallback COCO-80 category names in the 0-indexed order used by
# detection-datasets/coco (the primary loader below). This is only consulted
# when the live dataset doesn't expose a ClassLabel feature we can read.
_COCO80_NAMES_0INDEXED = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


# Caption templates used when COCO has no human-written caption — written to
# loosely match the prompt distribution of ImagenWorld (instruction-style) and
# RichHF (descriptive-style) so the model doesn't see "A photo of X" 5000 times.
# {subj} is filled with a comma-separated list of category names.
_COCO_CAPTION_TEMPLATES = [
    "A photograph showing {subj} in a real-world scene.",
    "Render a realistic image featuring {subj}.",
    "A candid shot capturing {subj} together.",
    "An everyday scene with {subj}.",
    "Generate a natural picture of {subj}.",
    "{subj_cap} arranged in a realistic setting.",
    "A high-resolution photo depicting {subj}.",
    "Real-life photo: {subj}.",
    "An image of {subj} taken in natural light.",
    "Photo capturing {subj} in their typical environment.",
    "A documentary-style shot of {subj}.",
    "{subj_cap} photographed from a natural perspective.",
    "A snapshot featuring {subj}.",
    "Picture of {subj} in a typical context.",
    "A clean, well-lit photo of {subj}.",
]

import re as _re

_TEMPLATE_PATTERNS = [
    _re.compile(
        "^"
        + _re.escape(t).replace(r"\{subj\}", "(.+)").replace(r"\{subj_cap\}", "(.+)")
        + "$"
    )
    for t in _COCO_CAPTION_TEMPLATES
]


def is_template_caption(caption: str) -> bool:
    """True iff ``caption`` was produced by one of the synthetic templates
    above. Used by the strict-caption gate and by corpus hygiene audits
    (P0-5: official builds must contain ZERO template captions)."""
    return any(p.match(caption or "") for p in _TEMPLATE_PATTERNS)


class COCOIngestor(BaseIngestor):
    """Ingestor for COCO real-world images (no-error samples)."""

    def __init__(
        self,
        data_root: Path,
        max_samples: Optional[int] = 5000,
        split: str = "train",
        seed: int = 42,
        image_save_dir: Optional[Path] = None,
        strict_captions: bool = True,
    ):
        """
        Initialize COCO ingestor.

        Args:
            data_root: Root directory for saving COCO data
            max_samples: Number of samples to select (default 5000)
            split: Dataset split to use
            seed: Random seed for reproducible sampling
            image_save_dir: Directory to save images. Defaults to data_root / "images".
            strict_captions: When True (default), samples WITHOUT an official
                human-written caption are SKIPPED (and counted) instead of
                receiving a synthetic template caption. The paper claims COCO
                uses official human captions, so official builds must run
                strict; template captions are opt-in for ablations only (P0-5).
        """
        super().__init__(data_root, max_samples)
        self.split = split
        self.seed = seed
        self.image_save_dir = Path(image_save_dir) if image_save_dir else (data_root / "images")
        self.strict_captions = strict_captions
        self._template_skipped = 0

    @property
    def source_name(self) -> str:
        return "COCO-real"

    def iterate(self) -> Iterator[RawSample]:
        """Iterate over COCO samples as perfect-quality examples."""
        # Try local cache first
        manifest = self.data_root / "manifest.jsonl"
        if manifest.exists():
            logger.info("Loading COCO samples from local manifest")
            yield from self._iterate_local(manifest)
            return

        # Load from HuggingFace
        logger.info("Loading COCO from HuggingFace (this may take a while on first run)...")
        yield from self._iterate_huggingface()

    def _iterate_local(self, manifest: Path) -> Iterator[RawSample]:
        """Iterate from previously saved local manifest."""
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                if self._should_stop():
                    break
                rec = json.loads(line)
                img_path = rec["image_path"]
                if not Path(img_path).exists():
                    continue

                # A cached manifest may predate the strict-caption gate and
                # carry materialized template captions — filter them here too.
                caption_is_template = is_template_caption(rec["caption"])
                if self.strict_captions and caption_is_template:
                    self._template_skipped += 1
                    continue

                sample = self._build_sample(
                    image_path=img_path,
                    caption=rec["caption"],
                    idx=rec["idx"],
                    image_id=rec.get("image_id", rec["idx"]),
                    caption_source="template" if caption_is_template else "human",
                )
                yield sample
                self._increment()
        if self._template_skipped:
            logger.warning(
                f"COCO strict captions: skipped {self._template_skipped} "
                f"manifest rows with template captions"
            )

    def _iterate_huggingface(self) -> Iterator[RawSample]:
        """Load COCO from HuggingFace, save images, and yield samples."""
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed. Install with: pip install datasets")
            return

        try:
            ds = load_dataset(
                "detection-datasets/coco",
                split="train",
                trust_remote_code=True,
            )
        except Exception:
            try:
                ds = load_dataset(
                    "HuggingFace-M4/COCO",
                    split="train",
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.error(f"Failed to load COCO dataset: {e}")
                return

        logger.info(f"COCO dataset loaded: {len(ds)} samples")

        # Resolve category ID → name once per dataset so every caption fallback
        # can use readable names instead of raw integers.
        category_names = self._resolve_category_names(ds)

        # Sample a random subset
        total = len(ds)
        n = min(self.max_samples or total, total)
        rng = random.Random(self.seed)
        indices = rng.sample(range(total), n)

        self.image_save_dir.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.data_root / "manifest.jsonl"

        manifest_f = open(manifest_path, "w", encoding="utf-8")

        try:
            for i, idx in enumerate(indices):
                if self._should_stop():
                    break

                ex = ds[idx]
                image = ex.get("image")
                if image is None:
                    continue

                # Save image to disk
                image_id = ex.get("image_id", idx)
                caption, caption_source = self._extract_caption(
                    ex, category_names, image_id=image_id)
                if not caption:
                    continue
                save_path = self.image_save_dir / f"coco_{image_id}.jpg"
                if not save_path.exists():
                    try:
                        if hasattr(image, "save"):
                            image.save(str(save_path))
                        else:
                            continue
                    except Exception as e:
                        logger.warning(f"Failed to save COCO image {image_id}: {e}")
                        continue

                image_path = str(save_path)

                # Save to manifest for fast re-loading
                manifest_rec = {
                    "idx": i,
                    "image_id": image_id,
                    "image_path": image_path,
                    "caption": caption,
                    "caption_source": caption_source,
                }
                manifest_f.write(json.dumps(manifest_rec, ensure_ascii=False) + "\n")

                sample = self._build_sample(
                    image_path=image_path,
                    caption=caption,
                    idx=i,
                    image_id=image_id,
                    caption_source=caption_source,
                )
                yield sample
                self._increment()

                if (i + 1) % 500 == 0:
                    logger.info(f"Processed {i + 1}/{n} COCO samples")
        finally:
            manifest_f.close()

        logger.info(f"COCO: saved {self._count} samples, manifest at {manifest_path}")
        if self._template_skipped:
            logger.warning(
                f"COCO strict captions: skipped {self._template_skipped} "
                f"samples lacking an official human caption"
            )

    @staticmethod
    def _resolve_category_names(ds: Any) -> List[str]:
        """Return a list of category names indexed by category id.

        Tries to pull a ClassLabel name list off the dataset's feature schema
        (as used by detection-datasets/coco). Falls back to the 80-class COCO
        list when the schema is unavailable.
        """
        try:
            features = getattr(ds, "features", None)
            if features is not None and "objects" in features:
                objects_feat = features["objects"]
                # datasets.Sequence wraps a dict of sub-features
                sub = getattr(objects_feat, "feature", objects_feat)
                if isinstance(sub, dict) and "category" in sub:
                    cat_feat = sub["category"]
                    names = getattr(cat_feat, "names", None)
                    if names:
                        return list(names)
        except Exception as e:
            logger.debug(f"Could not read category names from features: {e}")
        return list(_COCO80_NAMES_0INDEXED)

    def _extract_caption(
        self,
        ex: Dict[str, Any],
        category_names: Optional[List[str]] = None,
        image_id: Any = None,
    ) -> tuple:
        """Extract a caption from a COCO example.

        Returns ``(caption, source)`` where source is "human" or "template";
        ``("", "none")`` means the sample must be skipped. Under strict
        captions (the default) the template path is DISABLED: no official
        human caption → skip + count, never synthesize one (P0-5)."""
        # Different COCO HF datasets use different field names
        for key in ("caption", "captions", "sentences", "text"):
            val = ex.get(key)
            if val:
                if isinstance(val, list):
                    # Pick the longest caption
                    val = [v for v in val if isinstance(v, str) and v.strip()]
                    if val:
                        return max(val, key=len), "human"
                elif isinstance(val, str) and val.strip():
                    return val.strip(), "human"

        if self.strict_captions:
            self._template_skipped += 1
            return "", "none"

        # Template fallback (opt-in, strict_captions=False only). Some COCO
        # datasets store per-object categories instead of a caption. Map ids
        # through category_names (ClassLabel or the static COCO-80 fallback).
        # Duplicates are collapsed while preserving order.
        objects = ex.get("objects", {})
        resolved: List[str] = []
        if isinstance(objects, dict):
            cats = objects.get("category", [])
            if cats:
                names = category_names or _COCO80_NAMES_0INDEXED
                seen: set = set()
                for c in cats:
                    label: Optional[str] = None
                    if isinstance(c, str):
                        label = c.strip() or None
                    elif isinstance(c, int) and 0 <= c < len(names):
                        label = names[c]
                    if label and label not in seen:
                        resolved.append(label)
                        seen.add(label)
                    if len(resolved) >= 10:
                        break

        if not resolved:
            return "", "none"

        return self._render_template_caption(resolved, image_id), "template"

    @staticmethod
    def _render_template_caption(labels: List[str], image_id: Any) -> str:
        """Pick a varied caption template deterministically per image_id.

        Hashing image_id (rather than using the global RNG) keeps the same
        photo phrased the same way across rebuilds, while spreading the 15
        templates roughly uniformly across the dataset so the model doesn't
        memorize a single "A photo of X" pattern.
        """
        if len(labels) == 1:
            subj = labels[0]
        elif len(labels) == 2:
            subj = f"{labels[0]} and {labels[1]}"
        else:
            subj = ", ".join(labels[:-1]) + f", and {labels[-1]}"

        seed_src = f"coco-{image_id}".encode("utf-8")
        seed = int(hashlib.md5(seed_src).hexdigest(), 16)
        template = _COCO_CAPTION_TEMPLATES[seed % len(_COCO_CAPTION_TEMPLATES)]
        return template.format(subj=subj, subj_cap=subj[0].upper() + subj[1:])

    def _build_sample(
        self,
        image_path: str,
        caption: str,
        idx: int,
        image_id: Any,
        caption_source: str = "human",
    ) -> RawSample:
        """Build a perfect-quality RawSample from a COCO image."""
        return RawSample(
            image=image_path,
            instruction=caption,
            score=5.0,  # Perfect score (1-5 scale, will normalize to 1.0)
            text_reason="Real photograph with no generation artifacts or quality issues",
            visual_reason_type="svg",
            visual_reason_data='<svg viewBox="0 0 1000 1000"></svg>',  # Empty — no problem regions
            source="COCO-real",
            id=f"coco_{image_id}",
            orig={
                "dataset": "COCO",
                "image_id": image_id,
                "caption": caption,
                "caption_source": caption_source,
            },
            image_width=None,
            image_height=None,
        )
