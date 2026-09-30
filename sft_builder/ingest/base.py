"""
Base class for data ingestors.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

from ..schema import RawSample


class BaseIngestor(ABC):
    """Abstract base class for dataset ingestors."""

    def __init__(self, data_root: Path, max_samples: Optional[int] = None):
        """
        Initialize ingestor.

        Args:
            data_root: Root directory containing the dataset
            max_samples: Maximum number of samples to ingest (None = all)
        """
        self.data_root = Path(data_root)
        self.max_samples = max_samples
        self._count = 0

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the source name for this dataset."""
        pass

    @abstractmethod
    def iterate(self) -> Iterator[RawSample]:
        """
        Iterate over raw samples from the dataset.

        Yields:
            RawSample objects (may have invalid/None fields)
        """
        pass

    def _should_stop(self) -> bool:
        """Check if we've reached max_samples."""
        if self.max_samples is None:
            return False
        return self._count >= self.max_samples

    def _increment(self) -> None:
        """Increment the sample counter."""
        self._count += 1
