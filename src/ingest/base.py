"""
Collector interface. Every source (Reddit, Twitter, LinkedIn, etc.) implements
this. The orchestrator only knows about Collector, never about source-specific
details.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterator

from ..schema import Mention


class Collector(ABC):
    """Source-agnostic ingestion contract."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        ...

    @abstractmethod
    def collect(self, since: datetime, until: datetime) -> Iterator[Mention]:
        """Yield mentions in the [since, until) window. Idempotent."""
        ...
