"""Disclose the run's remaining usage budget to the model (private, not re-exported at top level)."""

from pydantic_ai_harness.experimental._warn import warn_experimental
from pydantic_ai_harness.experimental.budget_disclosure._capability import (
    BudgetDimension,
    BudgetDisclosure,
    BudgetFormatter,
)

warn_experimental('budget_disclosure')

__all__ = [
    'BudgetDimension',
    'BudgetDisclosure',
    'BudgetFormatter',
]
