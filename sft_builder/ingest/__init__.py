"""
Data ingestion modules for various datasets.
"""

from .base import BaseIngestor
from .imagenworld import ImagenWorldIngestor
from .richhf import RichHFIngestor
from .coco import COCOIngestor

__all__ = ["BaseIngestor", "ImagenWorldIngestor", "RichHFIngestor", "COCOIngestor"]
