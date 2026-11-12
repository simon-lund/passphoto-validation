"""A check, and the sequence of checks a photo runs through.

A check is one measurement of a photo, graded against a warn level and a fail level. It is written the way
a layer is written in PyTorch: a class declares its parameters, an instance holds its own values, `measure`
does the work, and a `Sequence` runs the instances in order (like `torch.nn.Sequential`):

    class Contrast(Check):
        name = 'contrast'
        warn_below: float = 40.0        # a parameter with its default
        fail_below: float = 30.0

        def measure(self, context):
            contrast = ...
            context.measurements['contrast'] = contrast
            self.judge(context, contrast, fail=('Contrast is far too low.', 7), warn=('Low contrast.', 5))

    checks = Sequence(Resolution(), Contrast(), OfiqPose('ofiq_head_yaw', axis='yaw'), ...)
    checks.load({'contrast': {'warn_below': 45}})       # values by check name, like a state dict
    results = checks.run(context)

What a check is made of:

- `name`: how reports, rules and thresholds files refer to it. A contract: add names, never rename. A class
  gives the name its one instance has; a class used several times (OFIQ's measures) takes it per instance.
- parameters: the annotated class attributes, in the order the thresholds file lists them. One without a
  default has to be given a value (core/thresholds.py holds the calibrated numbers). Levels are named
  `warn_…` and `fail_…`; a text parameter (an axis, a measure name) says what the instance is and is not a
  setting.
- `stops_after`: checks whose failure leaves this one nothing to measure. It is then not run and ends as
  skipped with "Not judged: <the other's reason>".
- `reads`: checks whose result this one looks at (`context.results`). They have to come earlier in the
  sequence; the sequence refuses an order where they do not.
- `measure(context)`: reads the shared measurements of the photo, writes its numbers to
  `context.measurements`, grades them with `self.judge`, and raises `CheckSkipped` when it cannot judge
  this photo.

The order of a sequence is the order of the report, and nothing else decides it: there is no discovery and
no registration by import. A check from outside this package is a subclass put into the sequence.

Two ways to do without a check, with different results. `enabled=False` (or `enabled = false` in the
thresholds file) keeps it in the sequence and does not run it: a rule only it covers ends as not_checked,
which never blocks validity. A check left out of the sequence altogether is simply missing: a rule only it
covers ends as skipped, and a photo with a skipped rule is never valid (core/rules.py).
"""
import copy
import logging
from typing import Any, ClassVar, Dict, Iterator, List, Mapping, Optional, Set, Tuple, get_origin

from .levels import Finding, Level, judge
from .models import CheckResult, CheckSkipped, ValidationContext

logger = logging.getLogger(__name__)

# The level names `Check.judge` reads from the check itself (see core/levels.py for what each means)
LEVEL_NAMES = ('warn_above', 'fail_above', 'warn_below', 'fail_below', 'warn_at', 'fail_at')
_REQUIRED = object()


class Check:
    """One measurement of a photo. Subclass it, declare the parameters, write `measure` (see the module)."""

    name: ClassVar[Optional[str]] = None
    stops_after: ClassVar[Tuple[str, ...]] = ()
    reads: ClassVar[Tuple[str, ...]] = ()
    # Every declared parameter with its default, in declaration order (_REQUIRED where there is none)
    _declared: ClassVar[Dict[str, Any]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        declared = {}
        for klass in reversed(cls.__mro__):
            for key, annotation in vars(klass).get('__annotations__', {}).items():
                if not key.startswith('_') and get_origin(annotation) is not ClassVar:
                    declared[key] = getattr(cls, key, _REQUIRED)
        cls._declared = declared

    def __init__(self, name: Optional[str] = None, *, stops_after: Optional[Tuple[str, ...]] = None,
                 reads: Optional[Tuple[str, ...]] = None, enabled: bool = True, **values):
        """
        :param name: The check's name, for a class that has none of its own or is used more than once.
        :param stops_after: Replaces the class's `stops_after` for this instance.
        :param reads: Replaces the class's `reads` for this instance.
        :param enabled: False leaves the check out of the run; a rule only it covers is then not_checked.
        :param values: Parameter values that differ from the class's defaults.
        """
        if name is not None:
            self.name = name
        if not self.name:
            raise TypeError(f'{type(self).__name__} has no name of its own: give one, {type(self).__name__}("name", ...)')
        if stops_after is not None:
            self.stops_after = tuple(stops_after)
        if reads is not None:
            self.reads = tuple(reads)
        self.enabled = enabled
        self.set(**values)

    # --- Parameters

    def set(self, **values) -> 'Check':
        """Give parameters new values, in place. A name the class does not declare is refused."""
        unknown = [key for key in values if key not in self._declared]
        if unknown:
            takes = ', '.join(self._declared) or 'no parameters'
            raise TypeError(f'{self.name} has no parameter {", ".join(unknown)}. It takes: {takes}')
        for key, value in values.items():
            setattr(self, key, value)
        return self

    def replace(self, **values) -> 'Check':
        """A copy of the check with some parameters changed; `enabled` can be changed the same way."""
        other = copy.copy(self)
        if 'enabled' in values:
            other.enabled = values.pop('enabled')
        return other.set(**values)

    def values(self) -> Dict[str, Any]:
        """Every parameter that has a value, in declaration order."""
        return {key: getattr(self, key) for key in self._declared if hasattr(self, key)}

    def missing(self) -> List[str]:
        """The parameters without a default that have not been given a value."""
        return [key for key in self._declared if not hasattr(self, key)]

    def levels(self) -> Dict[str, Level]:
        """The check's levels that carry the standard names (`LEVEL_NAMES`), for `grade` and `judge`."""
        return {key: getattr(self, key) for key in LEVEL_NAMES if key in self._declared}

    # --- The work

    def measure(self, context: ValidationContext) -> None:
        """Measure the photo and grade the result. Raise CheckSkipped when this photo cannot be judged."""
        raise NotImplementedError

    def judge(self, context: ValidationContext, value: float, *, fail: Finding, warn: Finding, **levels: Level) -> str:
        """Grade `value` against this check's levels and record the finding (core/levels.py).

        Without `levels` the check's own standard ones are used (`warn_above`, `fail_below`, …); a check
        whose levels carry other names passes them: `self.judge(..., warn_above=self.warn_share_above)`.
        A check with neither would pass every photo, so that is refused.
        """
        levels = levels or self.levels()
        if not levels:
            raise TypeError(f'{self.name} has no level to grade against: declare one ({", ".join(LEVEL_NAMES)}) '
                            f'or pass the levels to judge()')
        return judge(context, value, fail=fail, warn=warn, **levels)

    def __call__(self, context: ValidationContext) -> CheckResult:
        """Run the check on a photo and say how it ended, with the reasons and measurements it added."""
        issues_before = len(context.issues)
        measurements_before = dict(context.measurements)

        try:
            self.measure(context)
        except CheckSkipped as skip:
            return CheckResult(name=self.name, status='skipped', reasons=[str(skip)])
        except Exception as exc:
            logger.exception('Check %s crashed', self.name)
            return CheckResult(name=self.name, status='error', reasons=[f'{type(exc).__name__}: {exc}'])

        new_issues = context.issues[issues_before:]
        for issue in new_issues:
            issue.check = self.name

        if any(issue.category == 'error' for issue in new_issues):
            status = 'fail'
        elif any(issue.category == 'warning' for issue in new_issues):
            status = 'warn'
        else:
            status = 'pass'

        measurements = {
            key: value for key, value in context.measurements.items()
            if measurements_before.get(key, object()) != value
        }
        return CheckResult(
            name=self.name,
            status=status,
            reasons=[issue.message for issue in new_issues if issue.category != 'info'],
            measurements=measurements,
        )

    def __repr__(self) -> str:
        values = ', '.join(f'{key}={value!r}' for key, value in self.values().items())
        off = '' if self.enabled else ', enabled=False'
        return f'{type(self).__name__}({self.name!r}{", " if values else ""}{values}{off})'


def not_judged(root: CheckResult) -> str:
    """The reason of a check left out because `root` failed."""
    return f'Not judged: {root.reasons[0] if root.reasons else root.name + " failed."}'


class Sequence:
    """The checks a photo runs through, in the order they are reported."""

    def __init__(self, *checks: Check):
        self._checks: List[Check] = list(checks)
        names = [check.name for check in self._checks]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(f'Two checks are called {", ".join(twice)}; a name is used once in a sequence')
        position = {name: index for index, name in enumerate(names)}
        for check in self._checks:
            late = [other for other in check.reads if position.get(other, -1) > position[check.name]]
            if late:
                raise ValueError(f'{check.name} reads the result of {", ".join(late)}, which has to come before it')
        # Both kinds of dependency say "that one first"; two checks that wait for each other could never run
        before = {check.name: [other for other in (*check.stops_after, *check.reads) if other in position]
                  for check in self._checks}

        def waits(name: str, path: Tuple[str, ...]):
            if name in path:
                raise ValueError(f'These checks wait for each other: {" -> ".join((*path[path.index(name):], name))}')
            for other in before[name]:
                waits(other, (*path, name))

        for name in names:
            waits(name, ())

    def __iter__(self) -> Iterator[Check]:
        return iter(self._checks)

    def __len__(self) -> int:
        return len(self._checks)

    def __contains__(self, name: str) -> bool:
        return any(check.name == name for check in self._checks)

    def __getitem__(self, name: str) -> Check:
        for check in self._checks:
            if check.name == name:
                return check
        raise KeyError(f'No check called {name}. The checks are: {", ".join(self.names())}')

    def __add__(self, more) -> 'Sequence':
        """This sequence with more checks after it: `default_checks() + [MyCheck()]`."""
        return Sequence(*self._checks, *more)

    def __repr__(self) -> str:
        return 'Sequence(\n' + ''.join(f'  {check!r},\n' for check in self._checks) + ')'

    def names(self) -> List[str]:
        return [check.name for check in self._checks]

    def enabled(self) -> List[Check]:
        """The checks that run."""
        return [check for check in self._checks if check.enabled]

    def disabled(self) -> Set[str]:
        """The names of the checks that are turned off."""
        return {check.name for check in self._checks if not check.enabled}

    def copy(self) -> 'Sequence':
        """A sequence of copies, so that changing its values leaves this one alone."""
        return Sequence(*(check.replace() for check in self._checks))

    def load(self, values: Mapping[str, Mapping[str, Any]]) -> 'Sequence':
        """Give the checks parameter values by check name, in place (like a state dict): `{check: {parameter:
        value}}`. A check or parameter that does not exist is refused. Returns the sequence."""
        for name, parameters in values.items():
            self[name].set(**parameters)
        return self

    def complete(self) -> 'Sequence':
        """The sequence, once every parameter of every check has a value; otherwise an error that names them."""
        missing = [f'{check.name}: {", ".join(check.missing())}' for check in self._checks if check.missing()]
        if missing:
            raise ValueError('Parameters without a value (core/thresholds.py holds the numbers): ' + '; '.join(missing))
        return self

    def dependencies(self) -> Dict[str, Tuple[str, ...]]:
        """Which checks a failed check stops: {the check that failed: the checks not run after it}."""
        stops: Dict[str, List[str]] = {}
        for check in self._checks:
            for root in check.stops_after:
                stops.setdefault(root, []).append(check.name)
        return {root: tuple(names) for root, names in stops.items()}

    def run(self, context: ValidationContext) -> List[CheckResult]:
        """Run every enabled check and return the results in sequence order.

        A check another one waits for (`stops_after`) is measured before the first check that waits for it,
        so that check can be left out when the first has failed; the results, the issues and the measurements
        still come out in sequence order, as if nothing had moved. A check that is skipped, warns, crashes or
        is turned off stops nothing. Each result is also put into `context.results`, where a later check can
        read it (`reads`); a check measured early has the checks it reads measured before it.
        """
        checks = {check.name: check for check in self.enabled()}
        results: Dict[str, CheckResult] = {}

        def run(name: str) -> CheckResult:
            if name not in results:
                check = checks[name]
                for other in check.reads:
                    if other in checks:
                        run(other)
                failed = next((root for root in (run(other) for other in check.stops_after if other in checks)
                               if root.status == 'fail'), None)
                if failed is not None:
                    results[name] = CheckResult(name=name, status='skipped', reasons=[not_judged(failed)])
                else:
                    results[name] = check(context)
                context.results[name] = results[name]
            return results[name]

        for name in checks:
            run(name)
        ordered = [results[name] for name in checks]

        # A check measured early put its issues and measurements before its turn; restore the sequence order
        position = {name: i for i, name in enumerate(checks)}
        context.issues.sort(key=lambda issue: position.get(issue.check, -1))
        measurements = {}
        for result in ordered:
            for key in result.measurements:
                measurements.setdefault(key, context.measurements[key])
        measurements.update({key: value for key, value in context.measurements.items() if key not in measurements})
        context.measurements = measurements
        return ordered
