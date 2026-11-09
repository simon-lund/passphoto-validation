"""The thresholds table is what the validator runs with, and the values are pinned."""
from passphoto_validation.core import PassportPhotoValidator
from passphoto_validation.core.thresholds import THRESHOLDS, thresholds, warn_fail


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
    threshold_keys = ('warn_', 'fail_', 'min_', 'chroma_', 'rim_')
    for name, kwargs in _registered().items():
        passed = {key for key in kwargs if key.startswith(threshold_keys)}
        if name == 'resolution':  # its minimums come from ValidatorConfig
            continue
        assert passed <= set(thresholds(name)) if name in THRESHOLDS else not passed, name


def test_the_table_is_pinned():
    # Changing a threshold changes verdicts: do it on purpose, here and in the notebook
    assert {name: entry.kwargs for name, entry in THRESHOLDS.items()} == {
        'face_sharpness': dict(warn_below=0.14, fail_below=0.11),
        'compression': dict(warn_above=1.6, fail_above=2.5),
        'exposure': dict(warn_below=75, fail_below=60),
        'background_brightness': dict(min_brightness=120, fail_below=80),
        'background_pattern': dict(warn_above=10, fail_above=20),
        'background_gradient': dict(warn_above=18, fail_above=30),
        'background_shadow': dict(warn_above=5, fail_above=9, rim_confidence=0.85, rim_texture=3.0),
        'print_defects': dict(fail_at=3),
        'skin_smoothing': dict(warn_below=0.15, warn_gradient_ratio_below=0.115),
        'skin_colour_patches': dict(warn_share_above=0.02, chroma_distance=30),
        'gaze': dict(warn_above=0.08, fail_above=0.15),
        'red_eyes': dict(fail_above=0.3),
        'glasses_tint': dict(min_glasses_share=0.03, warn_below=0.55, fail_below=0.45),
        'glasses_glare': dict(min_glasses_share=0.03, warn_above=0.01, fail_above=0.04),
        'siglip_glasses_glare': dict(min_glasses_share=0.03, warn_above=-1.12),
        'siglip_gaze': dict(warn_above=-0.33),
        'ofiq_head_yaw': dict(warn_above=5, fail_above=10),
        'ofiq_head_pitch': dict(warn_above=8, fail_above=12),
        'ofiq_head_roll': dict(warn_above=6, fail_above=8),
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
    assert warn_fail('print_defects') == (None, 3)
    assert warn_fail('skin_smoothing') == (0.15, None)
    assert warn_fail('skin_colour_patches') == (0.02, None)
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
