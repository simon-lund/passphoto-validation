"""A check is a class, and a sequence of checks is the order of the report (core/check.py)."""
import numpy as np
import pytest

from passphoto_validation.core import PassportPhotoValidator, ValidatorConfig
from passphoto_validation.core.check import Check, Sequence
from passphoto_validation.core.checks import default_checks
from passphoto_validation.core.models import CheckSkipped, ValidationContext
from passphoto_validation.core.thresholds import THRESHOLDS
from passphoto_validation.core.validator import CHECK_DEPENDENCIES


class Brightness(Check):
    """A check the way one is written: a name, the parameters with their defaults, the measurement."""
    name = 'brightness'

    warn_below: float = 100
    fail_below: float
    region: str = 'whole photo'

    def measure(self, context):
        value = float(context.image_np.mean())
        context.measurements['brightness'] = value
        self.judge(context, value, fail=(f'Too dark ({value:.0f}).', 7), warn=(f'Rather dark ({value:.0f}).', 3))


class Follows(Check):
    """A check that waits for another and reads its result."""
    stops_after = ('brightness',)
    reads = ('brightness',)

    def measure(self, context):
        context.measurements[self.name] = context.results['brightness'].status


class Cannot(Check):
    name = 'cannot'

    def measure(self, context):
        raise CheckSkipped('Nothing to measure.')


def _context(level=120):
    return ValidationContext(image_np=np.full((90, 70, 3), level, np.uint8))


def test_a_class_declares_its_parameters_and_an_instance_holds_its_values():
    check = Brightness(fail_below=60)
    assert check.name == 'brightness'
    assert check.values() == dict(warn_below=100, fail_below=60, region='whole photo')  # in declaration order
    assert check.levels() == dict(warn_below=100, fail_below=60)
    assert repr(check) == "Brightness('brightness', warn_below=100, fail_below=60, region='whole photo')"
    # Another instance, another name, other values; the class and the first instance stay as they were
    other = Brightness('brightness_of_the_face', warn_below=90, fail_below=50, enabled=False)
    assert (other.name, other.warn_below, other.enabled) == ('brightness_of_the_face', 90, False)
    assert (check.warn_below, Brightness.warn_below, check.enabled) == (100, 100, True)
    changed = check.replace(warn_below=80, enabled=False)
    assert (changed.warn_below, changed.enabled, check.warn_below, check.enabled) == (80, False, 100, True)


def test_a_parameter_the_class_does_not_declare_is_refused():
    with pytest.raises(TypeError, match='brightness has no parameter warn_abov. It takes: warn_below, fail_below, region'):
        Brightness(warn_abov=5)
    with pytest.raises(TypeError, match='Follows has no name of its own'):
        Follows()
    assert Brightness().missing() == ['fail_below']  # no default and no value yet


def test_a_check_ends_as_pass_warn_fail_skipped_or_error_with_what_it_added():
    result = Brightness(fail_below=60)(_context(80))
    assert (result.name, result.status, result.reasons) == ('brightness', 'warn', ['Rather dark (80).'])
    assert result.measurements == {'brightness': 80.0}
    assert Brightness(fail_below=60)(_context(40)).status == 'fail'
    assert Brightness(fail_below=60)(_context(200)).status == 'pass'
    skipped = Cannot()(_context())
    assert (skipped.status, skipped.reasons) == ('skipped', ['Nothing to measure.'])
    crashed = Brightness()(_context())  # fail_below has no value: a bug, not a property of the photo
    assert crashed.status == 'error' and 'AttributeError' in crashed.reasons[0]


def test_the_sequence_is_the_order_of_the_results_and_takes_values_by_name():
    checks = Sequence(Brightness(), Follows('after'), Cannot()).load({'brightness': dict(fail_below=60, warn_below=90)})
    assert checks.names() == ['brightness', 'after', 'cannot'] and len(checks) == 3
    assert (checks['brightness'].fail_below, checks['brightness'].warn_below) == (60, 90)
    assert 'brightness' in checks and 'contrast' not in checks
    with pytest.raises(KeyError, match='No check called contrast'):
        checks.load({'contrast': dict(warn_below=1)})
    with pytest.raises(TypeError, match='has no parameter fail_above'):
        checks.load({'brightness': dict(fail_above=1)})
    with pytest.raises(ValueError, match='Two checks are called brightness'):
        Sequence(Brightness(), Brightness())
    with pytest.raises(ValueError, match='Parameters without a value .* brightness: fail_below'):
        Sequence(Brightness()).complete()


def test_a_check_is_not_run_once_one_it_waits_for_has_failed():
    # `after` comes first in the sequence, but brightness is measured before it
    checks = Sequence(Follows('after', reads=()), Brightness(fail_below=60))
    assert checks.dependencies() == {'brightness': ('after',)}
    context = _context(40)
    results = checks.run(context)
    assert [(r.name, r.status) for r in results] == [('after', 'skipped'), ('brightness', 'fail')]
    assert results[0].reasons == ['Not judged: Too dark (40).']
    assert context.results['after'] is results[0]
    # A warning stops nothing, and neither does a check that is turned off
    assert [r.status for r in checks.run(_context(80))] == ['pass', 'warn']
    off = Sequence(Follows('after', reads=()), Brightness(fail_below=60, enabled=False))
    assert [(r.name, r.status) for r in off.run(_context(40))] == [('after', 'error')]  # nothing to read: its own bug


def test_a_check_that_reads_another_has_to_come_after_it():
    with pytest.raises(ValueError, match='after reads the result of brightness, which has to come before it'):
        Sequence(Follows('after'), Brightness(fail_below=60))
    context = _context(200)
    results = Sequence(Brightness(fail_below=60), Follows('after')).run(context)
    assert [r.status for r in results] == ['pass', 'pass'] and context.measurements['after'] == 'pass'


def test_the_default_sequence_is_complete_and_its_numbers_come_from_the_table():
    checks = default_checks()
    assert len(checks) == 39 and not [check.name for check in checks if check.missing()]
    for name, entry in THRESHOLDS.items():
        assert all(getattr(checks[name], key) == value for key, value in entry.kwargs.items()), name
    # Two classes serve OFIQ's fourteen measures, each instance under its own name
    assert type(checks['ofiq_head_roll']) is type(checks['ofiq_head_yaw'])
    assert (checks['ofiq_head_roll'].axis, checks['ofiq_head_yaw'].axis) == ('roll', 'yaw')
    # What stops what is declared by the checks, and reads the same as before they were classes
    assert set(CHECK_DEPENDENCIES) == {'ofiq_eyes_visible'}
    assert set(CHECK_DEPENDENCIES['ofiq_eyes_visible']) == {
        'ofiq_eyes_open', 'gaze', 'red_eyes', 'glasses_tint', 'glasses_glare', 'siglip_glasses_glare', 'siglip_gaze',
        'colour_cast'}
    assert checks['skin_smoothing'].reads == ('face_sharpness',)
    # Building it twice gives separate instances: changing one validator's values never reaches another
    assert default_checks()['contrast'] is not checks['contrast']


def test_a_check_from_outside_joins_the_sequence():
    extra = Brightness('brightness_overall', fail_below=60)
    validator = PassportPhotoValidator(ValidatorConfig(ofiq_config_dir=None, thresholds_file=None),
                                       checks=default_checks() + [extra])
    assert validator.check_names()[-1] == 'brightness_overall' and len(validator.checks) == 40
    assert validator.parameters()['brightness_overall'] == dict(enabled=True, warn_below=100, fail_below=60)
    assert validator.checks['brightness_overall'] is not extra  # the validator works on its own copy


def test_a_check_without_a_value_for_a_parameter_stops_the_validator_at_the_start():
    # Without this the check would crash on every photo instead
    with pytest.raises(ValueError, match='Parameters without a value .* brightness: fail_below'):
        PassportPhotoValidator(ValidatorConfig(ofiq_config_dir=None, thresholds_file=None),
                               checks=default_checks() + [Brightness()])


def test_a_check_without_levels_cannot_grade():
    class Unlevelled(Check):
        name = 'unlevelled'
        strength: float = 1.0

        def measure(self, context):
            self.judge(context, 5.0, fail=('Too much.', 5), warn=('A little much.', 2))

    # Graded against nothing, every photo would pass; the check ends as crashed, which no photo survives
    result = Unlevelled()(_context())
    assert result.status == 'error' and 'unlevelled has no level to grade against' in result.reasons[0]
    # Levels under other names are passed to judge
    class Named(Unlevelled):
        name = 'named'
        warn_strength_above: float = 3.0

        def measure(self, context):
            self.judge(context, 5.0, warn_above=self.warn_strength_above, fail=('Too much.', 5), warn=('A little much.', 2))

    assert Named()(_context()).status == 'warn'


def test_a_check_measured_early_has_what_it_reads_measured_first():
    # `first` waits for `second`, which reads brightness: running `second` early must not skip its reading
    checks = Sequence(Follows('first', stops_after=('second',), reads=()), Brightness(fail_below=60),
                      Follows('second', stops_after=()))
    context = _context(200)
    assert [(r.name, r.status) for r in checks.run(context)] == [('first', 'pass'), ('brightness', 'pass'), ('second', 'pass')]
    assert context.measurements['second'] == 'pass'


def test_checks_that_wait_for_each_other_are_refused():
    with pytest.raises(ValueError, match='These checks wait for each other: a -> b -> a'):
        Sequence(Follows('a', stops_after=('b',), reads=()), Follows('b', stops_after=('a',), reads=()))
