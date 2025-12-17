from dataclasses import dataclass
from typing import Optional

@dataclass
class QueryContext:
    """Context passed to agents for per-query configuration."""
    min_similarity: Optional[float] = None

