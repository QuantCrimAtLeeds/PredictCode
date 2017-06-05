"""
enum
~~~~

Some simple utilities on top of the standard library `enum` package.
"""

from enum import Enum
import enum as _enum

class IntEnum(_enum.IntEnum):
    """As :class:`enum.IntEnum` but with extra functionality."""
    @classmethod
    def fromvalue(cls, value):
        """Find the (first) enum type with this value."""
        for x in cls:
            if x.value == value:
                return x
        raise ValueError()
