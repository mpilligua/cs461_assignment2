"""Factories for all built-in TTA methods."""

from .base import TTAMethod  # noqa: F401
from .unadapted import Unadapted  # noqa: F401
from .norm import TestTimeNorm  # noqa: F401
from .submission import Submission  # noqa: F401

__all__ = ["TTAMethod", "Unadapted", "TestTimeNorm", "Submission"]
