"""Tell the model when earlier turns came from a different model (private, not re-exported at top level)."""

from pydantic_ai_harness.experimental._warn import warn_experimental
from pydantic_ai_harness.experimental.cross_model_label._capability import (
    CrossModelFormatter,
    CrossModelHistory,
    CrossModelHistoryLabel,
    FamilyResolver,
    Granularity,
    model_family,
    normalize_model_name,
)

warn_experimental('cross_model_label')

__all__ = [
    'CrossModelFormatter',
    'CrossModelHistory',
    'CrossModelHistoryLabel',
    'FamilyResolver',
    'Granularity',
    'model_family',
    'normalize_model_name',
]
