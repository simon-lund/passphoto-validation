"""The thresholds table is what the validator runs with, and the values are pinned."""
from edutap.image_api.core import PassportPhotoValidator
from edutap.image_api.core.thresholds import THRESHOLDS, thresholds, warn_fail


def _registered():
    validator = PassportPhotoValidator()
    return {name: kwargs for name, _, kwargs in validator._checks()}


def test_every_table_entry_is_what_its_check_runs_with():
    registered = _registered()
    for name, entry in THRESHOLDS.items():
        assert name in registered, name
        # The check may take more keyword arguments (an axis, a measure name), never other threshold values
        assert all(registered[name].get(key) == value for key, value in entry.kwargs.items()), name
        assert entry.note


def test_every_threshold_the_validator_passes_is_in_the_table():
    threshold_keys = ('warn_', 'fail_', 'min_')
    for name, kwargs in _registered().items():
        passed = {key for key in kwargs if key.startswith(threshold_keys)}
        if name == 'resolution':  # its minimums come from ValidatorConfig
            continue
        assert passed <= set(thresholds(name)) if name in THRESHOLDS else not passed, name


def test_warn_fail_reads_either_direction():
    assert warn_fail('face_sharpness') == (0.14, 0.11)
    assert warn_fail('background_gradient') == (18, 30)
    assert warn_fail('background_brightness') == (120, 80)
    assert warn_fail('print_defects') == (None, 3)
    assert warn_fail('skin_smoothing') == (0.15, None)
    assert warn_fail('red_eyes') == (None, 0.3)


def test_thresholds_returns_a_copy():
    kwargs = thresholds('gaze')
    kwargs['warn_above'] = 1
    assert thresholds('gaze')['warn_above'] == 0.08
