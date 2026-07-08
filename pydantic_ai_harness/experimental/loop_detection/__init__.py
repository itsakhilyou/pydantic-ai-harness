"""Detect and interrupt repeated-action loops (private, not re-exported at top level)."""

from pydantic_ai_harness.experimental._warn import warn_experimental
from pydantic_ai_harness.experimental.loop_detection._capability import (
    LoopDetected,
    LoopDetectedError,
    LoopDetection,
    LoopTier,
    OnLoop,
)

warn_experimental('loop_detection')

__all__ = [
    'LoopDetected',
    'LoopDetectedError',
    'LoopDetection',
    'LoopTier',
    'OnLoop',
]
