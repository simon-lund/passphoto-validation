"""The thresholds table is what the validator runs with, and the values are pinned; a thresholds file overrides them."""
import math
import pathlib
import re
import tomllib

import pytest
from PIL import Image
from typer.testing import CliRunner

from passphoto_validation.cli import app
from passphoto_validation.core import PassportPhotoValidator, ValidatorConfig
from passphoto_validation.core.checks import default_checks
from passphoto_validation.core.utils import convert_to_bgr
from passphoto_validation.core.thresholds import THRESHOLDS, ThresholdsFileError, parameters, thresholds, to_toml, warn_fail

IMAGES = pathlib.Path(__file__).parent / 'test-images'


def _registered():
    # What each check of a validator runs with, by name: every parameter with its value
    validator = PassportPhotoValidator()
    return {check.name: check.values() for check in validator._checks()}


def test_every_table_entry_is_what_its_check_runs_with():
    registered = _registered()
    for name, entry in THRESHOLDS.items():
        assert name in registered, name
        # The check may have more parameters (an axis, a measure name), never other threshold values
        assert all(registered[name].get(key) == value for key, value in entry.kwargs.items()), name
        assert entry.note


def test_every_threshold_the_validator_runs_with_is_in_the_table():
    # A check's values differ from the defaults of its class only where the table says so: the numbers have
    # one home. A parameter without a default has to be in the table
    threshold_keys = ('warn_', 'fail_', 'min_', 'chroma_', 'rim_')
    for check in default_checks():
        defaults = type(check)._declared
        own = {key for key, value in check.values().items() if key.startswith(threshold_keys) and defaults[key] != value}
        assert own <= set(thresholds(check.name)) if check.name in THRESHOLDS else not own, check.name


def test_the_table_is_pinned():
    # Changing a threshold changes verdicts: do it on purpose, here and in the notebook
    assert {name: entry.kwargs for name, entry in THRESHOLDS.items()} == {
        'face_sharpness': dict(warn_below=0.14, fail_below=0.11),
        'compression': dict(warn_above=1.6, fail_above=2.5),
        'exposure': dict(warn_below=75, fail_below=60),
        'contrast': dict(warn_below=40, fail_below=30),
        'colour_cast': dict(warn_above=12, fail_above=20),
        'background_brightness': dict(warn_below=120, fail_below=80),
        'background_pattern': dict(warn_above=10, fail_above=20),
        'background_gradient': dict(warn_above=18, fail_above=30),
        'background_shadow': dict(warn_above=5, fail_above=9, rim_confidence=0.85, rim_texture=3.0),
        'print_defects': dict(warn_at=1, fail_at=3),
        'skin_smoothing': dict(warn_below=0.15, warn_gradient_ratio_below=0.115),
        'skin_colour_patches': dict(warn_share_above=0.015, fail_share_above=0.025, chroma_distance=30),
        'gaze': dict(warn_above=0.08, fail_above=0.15),
        'red_eyes': dict(fail_above=0.3),
        'glasses_tint': dict(min_glasses_share=0.03, warn_below=0.55, fail_below=0.45),
        'glasses_glare': dict(min_glasses_share=0.03, warn_above=0.01, fail_above=0.04),
        'siglip_glasses_glare': dict(min_glasses_share=0.03, warn_above=-1.12),
        'siglip_gaze': dict(warn_above=-0.33),
        'ofiq_head_yaw': dict(warn_above=5, fail_above=10),
        'ofiq_head_pitch': dict(warn_above=8, fail_above=12),
        'ofiq_head_roll': dict(warn_above=4, fail_above=8),
        'ofiq_expression': dict(warn_below=85, fail_below=50),
        'ofiq_mouth_closed': dict(warn_below=93, fail_below=90),
        'ofiq_eyes_open': dict(warn_below=95, fail_below=70),
        'ofiq_eyes_visible': dict(warn_below=60, fail_below=25),
        'ofiq_face_occlusion': dict(warn_below=90, fail_below=70),
        'ofiq_mouth_occlusion': dict(warn_below=90, fail_below=70),
        'ofiq_head_covering': dict(warn_below=90, fail_below=50),
        'ofiq_background': dict(warn_below=40, fail_below=25),
        'ofiq_illumination': dict(warn_below=72, fail_below=60),
        'ofiq_overexposure': dict(warn_below=90, fail_below=50),
        'ofiq_natural_colour': dict(warn_below=80, fail_below=60),
    }


def test_warn_fail_reads_either_direction():
    assert warn_fail('face_sharpness') == (0.14, 0.11)
    assert warn_fail('background_gradient') == (18, 30)
    assert warn_fail('background_brightness') == (120, 80)
    assert warn_fail('print_defects') == (1, 3)
    assert warn_fail('skin_smoothing') == (0.15, None)
    assert warn_fail('skin_colour_patches') == (0.015, 0.025)
    assert warn_fail('red_eyes') == (None, 0.3)
    # Every entry with a warn or fail key is readable this way
    for name, entry in THRESHOLDS.items():
        warn, fail = warn_fail(name)
        assert (warn is not None) == any(key.startswith(('warn_', 'min_')) for key in entry.kwargs), name
        assert (fail is not None) == any(key.startswith('fail_') for key in entry.kwargs), name


def test_thresholds_returns_a_copy():
    kwargs = thresholds('gaze')
    kwargs['warn_above'] = 1
    assert thresholds('gaze')['warn_above'] == 0.08


# --- The thresholds file

def _checks(thresholds_file=None):
    config = ValidatorConfig(ofiq_config_dir=None, thresholds_file=thresholds_file)
    return PassportPhotoValidator.registered_checks(config)


def _set(checks):
    # The values a sequence differs by from the defaults of its classes, by check name: what was passed as
    # keyword arguments before the checks were classes. Text parameters (an axis) are left out
    return {check.name: {key: value for key, value in parameters(check).items() if type(check)._declared[key] != value}
            for check in checks}


def test_a_thresholds_file_changes_what_the_checks_run_with(tmp_path):
    file = tmp_path / 'thresholds.toml'
    file.write_text('[colour_cast]\nfail_above = 15\n\n[background_uniformity]\nfail_above = 150\n\n'
                    '[face_centered]\nvertical_range = [0.4, 0.7]\n')
    kwargs = _set(_checks(str(file)))
    assert kwargs['colour_cast'] == dict(warn_above=12, fail_above=15)
    assert kwargs['background_uniformity'] == dict(fail_above=150)
    assert kwargs['face_centered']['vertical_range'] == (0.4, 0.7)
    # Everything else keeps the table's values
    assert kwargs['compression'] == thresholds('compression')


def test_a_thresholds_file_is_checked_before_any_photo(tmp_path):
    cases = (
        ('[colour_caste]\nfail_above = 15\n', 'no check called colour_caste'),
        ('[colour_cast]\nfail_abov = 15\n', '[colour_cast] has no parameter fail_abov'),
        ('[colour_cast]\nfail_above = "high"\n', 'must be a number'),
        ('[colour_cast]\nfail_above = true\n', 'must be a number'),
        ('[ofiq_head_yaw]\naxis = "roll"\n', 'has no parameter axis'),
        ('colour_cast = 15\n', 'must be a table'),
        ('[red_eyes]\nenabled = 1\n', 'enabled must be true or false'),
    )
    for text, message in cases:
        file = tmp_path / 'thresholds.toml'
        file.write_text(text)
        with pytest.raises(ThresholdsFileError, match=re.escape(message)):
            _checks(str(file))
    with pytest.raises(ThresholdsFileError, match='no such file'):
        _checks(str(tmp_path / 'missing.toml'))


def test_the_exported_thresholds_file_round_trips(tmp_path):
    checks = _checks()
    text = to_toml(checks)
    data = tomllib.loads(text)
    assert set(data) == {check.name for check in checks if parameters(check)}
    for check in checks:
        for key, value in parameters(check).items():  # every value a check runs with is in the file, names aside
            if isinstance(value, float) and math.isinf(value):
                assert data[check.name][key] is False, (check.name, key)  # a level that is off
            elif value is not None and not (isinstance(value, tuple) and math.inf in value):
                assert data[check.name][key] == (list(value) if isinstance(value, tuple) else value), (check.name, key)
    # Feeding the file back changes nothing
    file = tmp_path / 'thresholds.toml'
    file.write_text(text)
    resolved = [(check.name, parameters(check)) for check in _checks(str(file))]
    assert resolved == [(check.name, parameters(check)) for check in checks]


def test_the_cli_prints_the_thresholds_file():
    result = CliRunner().invoke(app, ['thresholds'])
    assert result.exit_code == 0, result.output
    data = tomllib.loads(result.output)
    assert data['colour_cast'] == dict(enabled=True, warn_above=12, fail_above=20, min_whites_share=0.02, band_share=0.03)
    assert data['contrast'] == dict(enabled=True, warn_below=40, fail_below=30)
    assert data['red_eyes']['warn_above'] is False  # a level that is off


def test_false_turns_a_level_off_and_the_export_says_so(tmp_path):
    file = tmp_path / 'thresholds.toml'
    file.write_text('[red_eyes]\nfail_above = false\n\n[exposure]\nwarn_below = false\n')
    checks = _checks(str(file))
    kwargs = _set(checks)
    assert kwargs['red_eyes']['fail_above'] == math.inf and kwargs['exposure']['warn_below'] == -math.inf
    data = tomllib.loads(to_toml(checks))
    assert data['red_eyes']['fail_above'] is False and data['exposure']['warn_below'] is False
    assert data['skin_smoothing']['fail_below'] is False  # off by default
    assert all(section['enabled'] is True for section in data.values())


def test_enabled_false_turns_a_check_off_and_its_rule_is_not_checked(tmp_path):
    file = tmp_path / 'thresholds.toml'
    file.write_text('[skin_colour_patches]\nenabled = false\n\n[skin_smoothing]\nenabled = false\n')
    config = ValidatorConfig(ofiq_config_dir=None, thresholds_file=str(file))
    assert PassportPhotoValidator.disabled_checks(config) == {'skin_colour_patches', 'skin_smoothing'}
    validator = PassportPhotoValidator(config)
    assert 'skin_colour_patches' not in validator.check_names()
    assert validator.parameters()['skin_colour_patches']['enabled'] is False
    assert tomllib.loads(to_toml(validator.checks))['skin_smoothing']['enabled'] is False
    assert validator.disabled == {'skin_colour_patches', 'skin_smoothing'}
    image = Image.open(IMAGES / '9.jpg')
    image.thumbnail((600, 600))
    report = validator(convert_to_bgr(image), crop=False)
    assert 'skin_colour_patches' not in {check.name for check in report.checks}
    rule = {rule.id: rule for rule in report.rules}['no_retouching']
    assert rule.status == 'not_checked' and 'turned off' in rule.reasons[0]


def test_a_list_must_have_the_length_and_the_numbers_the_check_unpacks(tmp_path):
    # A list of the wrong length passed the type check and crashed the check on every photo
    cases = (
        ('[face_centered]\nvertical_range = [0.4]\n', 'vertical_range must be a list of 2 numbers'),
        ('[face_centered]\nvertical_range = [0.4, 0.6, 0.8]\n', 'vertical_range must be a list of 2 numbers'),
        ('[face_centered]\nvertical_range = ["low", "high"]\n', 'vertical_range must be a list of 2 numbers'),
        ('[face_centered]\nvertical_range = [true, false]\n', 'vertical_range must be a list of 2 numbers'),
        ('[face_centered]\nvertical_range = 0.5\n', 'vertical_range must be a list of 2 numbers'),
        ('[resolution]\nmin_res = [413]\n', 'min_res must be a list of 2 numbers'),
    )
    for text, message in cases:
        file = tmp_path / 'thresholds.toml'
        file.write_text(text)
        with pytest.raises(ThresholdsFileError, match=re.escape(message)):
            _checks(str(file))


def test_a_renamed_parameter_still_answers_to_its_old_name(tmp_path):
    # Thresholds files exported before the warn/fail naming use these names
    file = tmp_path / 'thresholds.toml'
    file.write_text('[background_brightness]\nmin_brightness = 100\n\n[background_uniformity]\nvariance_threshold = 150\n\n'
                    '[background_consistency]\ncov_threshold = 0.4\n')
    checks = _checks(str(file))
    kwargs = _set(checks)
    assert kwargs['background_brightness'] == dict(warn_below=100, fail_below=80)
    assert kwargs['background_uniformity'] == dict(fail_above=150)
    assert kwargs['background_consistency'] == dict(fail_above=0.4)
    # The export writes the current names only
    data = tomllib.loads(to_toml(checks))
    assert data['background_brightness']['warn_below'] == 100 and 'min_brightness' not in data['background_brightness']
    assert 'variance_threshold' not in data['background_uniformity'] and 'cov_threshold' not in data['background_consistency']
    # Old and new name together is a contradiction, whichever comes first
    for text in ('[background_brightness]\nmin_brightness = 100\nwarn_below = 110\n',
                 '[background_brightness]\nwarn_below = 110\nmin_brightness = 100\n'):
        file.write_text(text)
        with pytest.raises(ThresholdsFileError, match='sets both min_brightness and warn_below'):
            _checks(str(file))


def test_false_turns_off_the_levels_with_older_names_too(tmp_path):
    file = tmp_path / 'thresholds.toml'
    file.write_text('[resolution]\nmin_res = false\nrecommended_res = false\n\n[photo_format]\ntolerance = false\n\n'
                    '[face_centered]\nvertical_range = false\n')
    checks = _checks(str(file))
    kwargs = _set(checks)
    assert kwargs['resolution'] == dict(min_res=(0, 0), recommended_res=(0, 0))
    assert kwargs['photo_format'] == dict(tolerance=math.inf)
    assert kwargs['face_centered'] == dict(vertical_range=(-math.inf, math.inf))
    # The export says false for them, and that file loads back to the same values
    text = to_toml(checks)
    data = tomllib.loads(text)
    assert data['resolution']['min_res'] is False and data['resolution']['recommended_res'] is False
    assert data['photo_format']['tolerance'] is False and data['photo_format']['warn_tolerance'] is False
    assert data['face_centered']['vertical_range'] is False and data['face_centered']['fail_vertical_range'] is False
    file.write_text(text)
    assert [(check.name, parameters(check)) for check in _checks(str(file))] == \
        [(check.name, parameters(check)) for check in checks]


def test_a_range_open_at_one_end_round_trips(tmp_path):
    file = tmp_path / 'thresholds.toml'
    file.write_text('[face_centered]\nvertical_range = [0.45, inf]\n')
    checks = _checks(str(file))
    assert _set(checks)['face_centered'] == dict(vertical_range=(0.45, math.inf))
    text = to_toml(checks)
    assert 'vertical_range = [0.45, inf]' in text
    file.write_text(text)
    assert _checks(str(file))['face_centered'].vertical_range == (0.45, math.inf)


def test_the_parameters_the_api_lists_are_json(tmp_path):
    import json
    file = tmp_path / 'thresholds.toml'
    file.write_text('[face_centered]\nvertical_range = [0.45, inf]\n\n[resolution]\nrecommended_res = false\n')
    validator = PassportPhotoValidator(ValidatorConfig(ofiq_config_dir=None, thresholds_file=str(file)))
    listed = json.loads(json.dumps(validator.parameters(), allow_nan=False))  # JSON has no infinity
    assert listed['face_centered']['vertical_range'] == [0.45, None]
    assert listed['face_centered']['fail_vertical_range'] is False
    assert listed['resolution'] == dict(enabled=True, min_res=[413, 531], recommended_res=False)
    assert listed['photo_format']['warn_tolerance'] is False and listed['red_eyes']['warn_above'] is False
