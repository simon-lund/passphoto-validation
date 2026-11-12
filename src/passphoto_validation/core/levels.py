"""Where a measurement falls against a check's warn and fail levels: the one place that decides it.

Every check ends the same way, with a number, a warn level and a fail level. `grade` says which level the
number reached and `judge` also records the finding, so no check compares against its levels by hand.

The two levels are judged on their own: the fail level holds whether or not the warn level is on. A level
that is off is never reached and never printed. Off is +inf or -inf (what `false` in the thresholds file
becomes, see core/thresholds.py) or None.
"""
import math
from typing import Callable, Optional, Tuple

from .models import ValidationContext, ValidationIssue

Level = Optional[float]
Finding = Tuple[str, int]  # (message, severity)


def is_on(level: Level) -> bool:
    """Whether a level is set: not None and not ±inf."""
    return level is not None and math.isfinite(level)


def first_on(*levels: Level) -> Level:
    """The first of `levels` that is on, or None: the level a message should name."""
    return next((level for level in levels if is_on(level)), None)


def grade(value: float, *, warn_above: Level = None, fail_above: Level = None, warn_below: Level = None,
          fail_below: Level = None, warn_at: Level = None, fail_at: Level = None) -> str:
    """'fail', 'warn' or 'pass' for `value` against the levels given.

    `…_above` is reached by a higher value, `…_below` by a lower one, `…_at` by a count from that number on.
    """
    def reached(above: Level, below: Level, at: Level) -> bool:
        return ((above is not None and value > above) or (below is not None and value < below)
                or (at is not None and value >= at))

    if reached(fail_above, fail_below, fail_at):
        return 'fail'
    if reached(warn_above, warn_below, warn_at):
        return 'warn'
    return 'pass'


def judge(context: ValidationContext, value: float, *, fail: Finding, warn: Finding, **levels: Level) -> str:
    """Grade `value` (see `grade`) and record the finding: `fail` and `warn` are (message, severity)."""
    level = grade(value, **levels)
    if level != 'pass':
        message, severity = fail if level == 'fail' else warn
        context.issues.append(ValidationIssue(
            category='error' if level == 'fail' else 'warning', message=message, severity=severity))
    return level


def span(low: Level, high: Level, fmt: Callable[[float], str] = str, unit: str = '') -> str:
    """The allowed range in words: '70-80%' when both ends are on, 'at least 70%' or 'at most 80%' with
    one, '' with neither."""
    if is_on(low) and is_on(high):
        return f'{fmt(low)}-{fmt(high)}{unit}'
    if is_on(low):
        return f'at least {fmt(low)}{unit}'
    if is_on(high):
        return f'at most {fmt(high)}{unit}'
    return ''
