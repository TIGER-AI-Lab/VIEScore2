"""
Pipeline orchestration for SFT dataset building.

Flow: ingest -> validate -> normalize -> export
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from .schema import RawSample, SFTSample
from .ingest import ImagenWorldIngestor, RichHFIngestor, COCOIngestor
from .validate import Validator, ValidationStats
from .normalize import Normalizer, CanonicalVisualFormat
from .export import Exporter


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the SFT pipeline."""

    # Data paths
    data_root: Path = Path("data")
    output_path: Path = Path("sft_samples/sft_unified.jsonl")
    visual_data_dir: Optional[Path] = None

    # Source selection
    sources: List[str] = field(default_factory=lambda: ["imagenworld"])

    # ImagenWorld options
    imagenworld_tasks: Optional[List[str]] = None
    imagenworld_max_per_task: Optional[int] = None
    imagenworld_require_error_mask: bool = True  # Per spec: keep only has_error_mask == true

    # RichHF options
    richhf_split: str = "train"
    richhf_prompt_mapping: Optional[Path] = None
    richhf_require_visual: bool = False  # Include text-only samples by default

    # COCO options
    coco_max_samples: int = 5000
    coco_seed: int = 42

    # Global limits
    max_samples_per_source: Optional[int] = None
    max_total_samples: Optional[int] = None

    # Validation options
    require_image: bool = True
    allow_visual_resize: bool = True

    # Normalization options
    score_range: Tuple[float, float] = (0.0, 1.0)
    visual_format: CanonicalVisualFormat = "mask"

    # Export options
    embed_visual_data: bool = False
    include_orig_meta: bool = True


@dataclass
class PipelineStats:
    """Statistics from pipeline execution."""

    sources_processed: List[str] = field(default_factory=list)
    samples_ingested: Dict[str, int] = field(default_factory=dict)
    samples_validated: int = 0
    samples_exported: int = 0
    validation_stats: Optional[ValidationStats] = None

    def summary(self) -> str:
        """Return summary string."""
        lines = [
            "=" * 50,
            "Pipeline Summary",
            "=" * 50,
            f"Sources processed: {', '.join(self.sources_processed)}",
            f"Samples ingested by source:",
        ]
        for src, count in self.samples_ingested.items():
            lines.append(f"  - {src}: {count}")
        lines.append(f"Total ingested: {sum(self.samples_ingested.values())}")
        lines.append(f"Samples after validation: {self.samples_validated}")
        lines.append(f"Samples exported: {self.samples_exported}")

        if self.validation_stats:
            lines.append("")
            lines.append(self.validation_stats.summary())

        lines.append("=" * 50)
        return "\n".join(lines)


class Pipeline:
    """SFT dataset building pipeline."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.stats = PipelineStats()

        # Initialize components
        self.validator = Validator(
            require_image=config.require_image,
            allow_visual_resize=config.allow_visual_resize,
        )
        self.normalizer = Normalizer(
            score_range=config.score_range,
            visual_format=config.visual_format,
        )
        self.exporter = Exporter(
            output_path=config.output_path,
            visual_data_dir=config.visual_data_dir,
            embed_visual_data=config.embed_visual_data,
            include_orig_meta=config.include_orig_meta,
        )

    def run(self) -> PipelineStats:
        """
        Run the pipeline.

        Returns:
            Pipeline statistics
        """
        logger.info("Starting SFT pipeline...")

        # Step 1: Ingest from all sources
        logger.info("Step 1: Ingesting data...")
        raw_samples = self._ingest_all()

        # Step 2: Validate
        logger.info("Step 2: Validating samples...")
        valid_samples = list(self.validator.validate(raw_samples))
        self.stats.samples_validated = len(valid_samples)
        self.stats.validation_stats = self.validator.stats
        logger.info(f"Validated: {len(valid_samples)} samples passed")

        # Step 3: Normalize
        logger.info("Step 3: Normalizing samples...")
        normalized_samples = self.normalizer.normalize_batch(iter(valid_samples))

        # Apply total limit if configured
        if self.config.max_total_samples:
            normalized_samples = self._limit_samples(
                normalized_samples, self.config.max_total_samples
            )

        # Step 4: Export
        logger.info("Step 4: Exporting samples...")
        count = self.exporter.export(normalized_samples)
        self.stats.samples_exported = count

        logger.info(self.stats.summary())
        return self.stats

    def _ingest_all(self) -> Iterator[RawSample]:
        """Ingest samples from all configured sources."""
        for source in self.config.sources:
            source_lower = source.lower()

            if source_lower == "imagenworld":
                yield from self._ingest_imagenworld()

            elif source_lower == "richhf":
                yield from self._ingest_richhf()

            elif source_lower == "coco":
                yield from self._ingest_coco()

            else:
                logger.warning(f"Unknown source: {source}")

    def _ingest_imagenworld(self) -> Iterator[RawSample]:
        """Ingest from ImagenWorld dataset."""
        imagenworld_root = self.config.data_root / "ImagenWorld-annotated-set"

        if not imagenworld_root.exists():
            logger.warning(f"ImagenWorld not found at {imagenworld_root}")
            return

        ingestor = ImagenWorldIngestor(
            data_root=imagenworld_root,
            max_samples=self.config.max_samples_per_source,
            max_per_task=self.config.imagenworld_max_per_task,
            require_error_mask=self.config.imagenworld_require_error_mask,
            tasks=self.config.imagenworld_tasks,
        )

        count = 0
        for sample in ingestor.iterate():
            yield sample
            count += 1

        self.stats.sources_processed.append("ImagenWorld")
        self.stats.samples_ingested["ImagenWorld"] = count
        logger.info(f"ImagenWorld: ingested {count} samples")

    def _ingest_richhf(self) -> Iterator[RawSample]:
        """Ingest from RichHF-18K dataset."""
        richhf_root = self.config.data_root / "richhf-18k"

        if not richhf_root.exists():
            logger.warning(f"RichHF not found at {richhf_root}")
            return

        ingestor = RichHFIngestor(
            data_root=richhf_root,
            max_samples=self.config.max_samples_per_source,
            split=self.config.richhf_split,
            prompt_mapping_path=self.config.richhf_prompt_mapping,
            require_visual=self.config.richhf_require_visual,
            image_save_dir=richhf_root / "images",
        )

        count = 0
        for sample in ingestor.iterate():
            yield sample
            count += 1

        self.stats.sources_processed.append("RichHF-18K")
        self.stats.samples_ingested["RichHF-18K"] = count
        logger.info(f"RichHF-18K: ingested {count} samples")

    def _ingest_coco(self) -> Iterator[RawSample]:
        """Ingest real-world no-error samples from COCO."""
        coco_root = self.config.data_root / "coco-real"

        ingestor = COCOIngestor(
            data_root=coco_root,
            max_samples=self.config.coco_max_samples,
            seed=self.config.coco_seed,
            image_save_dir=coco_root / "images",
        )

        count = 0
        for sample in ingestor.iterate():
            yield sample
            count += 1

        self.stats.sources_processed.append("COCO-real")
        self.stats.samples_ingested["COCO-real"] = count
        logger.info(f"COCO-real: ingested {count} samples")

    def _limit_samples(
        self, samples: Iterator[SFTSample], limit: int
    ) -> Iterator[SFTSample]:
        """Limit total number of samples."""
        count = 0
        for sample in samples:
            if count >= limit:
                break
            yield sample
            count += 1


def build_sft(
    sources: List[str],
    output_path: Path,
    data_root: Path = Path("data"),
    score_range: Tuple[float, float] = (0.0, 1.0),
    visual_format: CanonicalVisualFormat = "mask",
    max_samples: Optional[int] = None,
    require_error_mask: bool = True,
) -> PipelineStats:
    """
    Convenience function to build SFT dataset.

    Args:
        sources: List of source names (e.g., ["imagenworld", "richhf"])
        output_path: Path to output JSONL file
        data_root: Root directory for datasets
        score_range: Target score range
        visual_format: Target visual format
        max_samples: Maximum total samples
        require_error_mask: ImagenWorld: only keep samples with error masks

    Returns:
        Pipeline statistics
    """
    config = PipelineConfig(
        data_root=data_root,
        output_path=output_path,
        sources=sources,
        score_range=score_range,
        visual_format=visual_format,
        max_total_samples=max_samples,
        imagenworld_require_error_mask=require_error_mask,
    )

    pipeline = Pipeline(config)
    return pipeline.run()
