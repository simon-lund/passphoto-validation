"""Every check's thresholds in one place, with where each number came from.

The sequence of checks loads this table on top of the defaults of the check classes
(`core/checks/__init__.py`: `default_checks`), the evaluation notebook draws the same numbers as the warn
and fail lines of its panels (`warn_fail`), and PASSPORT_VALIDATOR.md points here. To change a threshold: edit
it here, run the tests (`tests/test_thresholds.py` pins the table, so update its snapshot on purpose) and
re-run `notebooks/evaluation.ipynb` to see the effect.

The names are the parameters of the check classes. `warn_above` / `fail_above` mean higher is worse,
`warn_below` / `fail_below` lower is worse, `fail_at` is a count. A check not listed here runs with the
defaults of its class (`photo_format`, `face_size`, `face_centered`, `background_uniformity`,
`background_consistency`, `glare`); `resolution` takes its minimums from `ValidatorConfig`, where they can
be set per deployment.

A deployment can override any parameter of any check without editing this file: a TOML thresholds file
(`ValidatorConfig.thresholds_file`, `passport-validator --thresholds`), one table per check, is read by
`load_overrides` and merged by `apply_overrides`, which refuses unknown checks, unknown parameters and
wrong types (a list of the wrong length too). `to_toml` writes every parameter with its current value, to
start such a file from. The file is applied last, after this table and `ValidatorConfig` (so its
`[resolution]` wins over `MIN_RES_*`).

The convention of that file: every check has a warn level and a fail level (`warn_…`, `fail_…`); `false`
turns a level off (in the code an off level is +inf or -inf, which no measurement reaches; core/levels.py
grades every check); `enabled = false` turns a check off: it is not run, and a rule only it covered ends
as not_checked, which never blocks validity. A few levels carry older names or are a (low, high) pair
(`OTHER_LEVELS`); `false` turns those off the same way. A parameter that was renamed keeps answering to
its old name (`RENAMED`), so a thresholds file written earlier still loads.
"""
import math
import textwrap
import tomllib
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, Optional, Set, Tuple

from .check import Check, Sequence


@dataclass(frozen=True)
class Thresholds:
    """The threshold keyword arguments one check runs with, and where the numbers came from."""
    kwargs: Dict[str, Any]
    note: str


THRESHOLDS: Dict[str, Thresholds] = {
    # --- Our own checks on pixels and geometry (core/checks/). Calibrated on the London Set (notebooks/evaluation.ipynb)
    'face_sharpness': Thresholds(
        dict(warn_below=0.14, fail_below=0.11),
        'Laplacian over gradient energy on the face at 256 px; blur of sigma 2 and 4 px fails, sharp crops stay above'),
    'compression': Thresholds(
        dict(warn_above=1.6, fail_above=2.5),
        'Brightness jumps at 8-px JPEG block edges over those inside; JPEG quality 4 fails'),
    'contrast': Thresholds(
        dict(warn_below=40, fail_below=30),
        'Standard deviation of the grey levels: compliant crops never below 42, the low-contrast variant never '
        'above 25, the darkened one 29; 95% of real candid photos above 41 (1 of 223 fails, 4% warn). Fails since '
        'the review of 2026-11-10: a grey photo is not a valid photo'),
    'exposure': Thresholds(
        dict(warn_below=75, fail_below=60),
        'L* of the brightest 1% (the background), so dark skin is not read as underexposed; the darkened variant fails'),
    'colour_cast': Thresholds(
        dict(warn_above=12, fail_above=20),
        'CIELAB chroma the background and the whites of the eyes share in hue. Every London crop that should '
        'pass measures 0 (the backdrop is neutral) and every colour-cast variant 32 or more, so the London Set '
        'sets no threshold; the numbers come from 223 real candid photos judged by eye: above 20 the tint is '
        'plain (11 photos, 8 of them failed by OFIQ too, the other 3 warm indoor light), 12-20 is warm light '
        '(19 photos warn, 15 of which OFIQ passes). A cast on the face alone reads 0 (that is OFIQ\'s case), so '
        'does one that cancels a coloured background; dark lenses are read as eye whites (ofiq_eyes_visible '
        'fails those photos); eyes too closed to show their whites skip the check'),
    'background_brightness': Thresholds(
        dict(warn_below=120, fail_below=80),
        'Fails on dark backgrounds, warns on mid-tones: every Open Images background labelled dark is below 80, '
        'none labelled light; the London Set\'s plain backgrounds are all above 190'),
    'background_pattern': Thresholds(
        dict(warn_above=10, fail_above=20),
        'The London Set\'s plain backgrounds stay below 7 at 1063 px height (below 14 at 800 px); the faint UN '
        'flag in Open Images measures 34 (background labels: data/openimages/eval/vlm_background_labels.jsonl)'),
    # The four youngest checks: calibrated on the London Set's compliant crops and synthetic defects built from them
    # (scripts/build_eval_set.py: bg_gradient, bg_shadow, ink_stain, smoothed)
    'background_gradient': Thresholds(
        dict(warn_above=18, fail_above=30),
        'L* change across a plane fitted to the background; compliant crops reach 16, the gradient variant starts at 41'),
    'background_shadow': Thresholds(
        dict(warn_above=5, fail_above=9, rim_confidence=0.85, rim_texture=3.0),
        'L* the band next to the person is darker than that plane; compliant crops reach 9 (one photo; 98% stay '
        'under 5), the shadow variant starts at 3 (90% above 5): the rest of the shadow is swallowed into the '
        'person mask. The rim of the mask (loose 0.5 against tight 0.85) read on smooth backdrop-like pixels '
        '(lightness deviation under 3) measures 28 and more on every shadow variant and at most 4.7 on compliant, '
        'glasses and rotated crops; a texture cap of 5 would let good crops reach 7.3'),
    'print_defects': Thresholds(
        dict(warn_at=1, fail_at=3),
        'Marks on the background; any mark warns, three fail. Compliant crops show up to 2 (stray hair), the ink '
        'variant 3 or more'),
    'skin_smoothing': Thresholds(
        dict(warn_below=0.15, warn_gradient_ratio_below=0.115),
        'Cheek and forehead texture over eye texture at 192 px face width; filtered skin stays under 0.14, '
        'natural skin above 0.19. Skin texture over the median gradient of the face, for a filter that smooths '
        'the eyes too: compliant, glasses, cast, shadow, dark and low-contrast crops stay above 0.125, the '
        'whole-face smoothed variant under 0.11 (smoothed_all). Warns only by default: smooth skin can be real; '
        'the fail levels are off (fail_below, fail_gradient_ratio_below)'),
    'skin_colour_patches': Thresholds(
        dict(warn_share_above=0.015, fail_share_above=0.025, chroma_distance=30),
        'Patches of the lower face more than 30 CIELAB chroma units from the skin, at least as saturated: 0 on '
        'every London crop (plain, glasses, sunglasses, cast, shadow), 3% or more of the face oval with the '
        'sticker variant (face_stickers); of 223 real candid photos 4 fail (a clown nose, goggles, face paint) '
        'and 1 more warns. Fails since the review of 2026-11-10: a filter\'s stickers make the photo invalid'),
    # Eyes
    'gaze': Thresholds(
        dict(warn_above=0.08, fail_above=0.15),
        'Iris offset as a share of the eye width, both eyes averaged; real averted gaze on Open Images: 59% caught, 18% false'),
    'red_eyes': Thresholds(
        dict(fail_above=0.3),
        'Red share of the pupil relative to the skin; the red-eye variant fails, a warm colour cast passes'),
    'glasses_tint': Thresholds(
        dict(min_glasses_share=0.03, warn_below=0.55, fail_below=0.45),
        'Lightness behind the lens over the cheek below it, when the lenses cover at least 3% of the eye-distance '
        'area; sunglasses fail, real tinted glasses 92% caught with ofiq_eyes_visible'),
    'glasses_glare': Thresholds(
        dict(min_glasses_share=0.03, warn_above=0.01, fail_above=0.04),
        'Share of blown-out white pixels inside the lenses; drawn glare 96% caught, real reflections 10%'),
    # Second opinions, warn only. SigLIP SO400M: the score 80% of fine photos stay under, on the calibrate half of
    # the Open Images set (notebooks/real_photos.ipynb)
    'siglip_glasses_glare': Thresholds(
        dict(min_glasses_share=0.03, warn_above=-1.12),
        'SigLIP\'s "glasses with glare" against "clean lenses"; 37% of real reflections caught, 15% false'),
    'siglip_gaze': Thresholds(
        dict(warn_above=-0.33),
        'SigLIP\'s "eyes looking away" against "into the camera"; 80% of real averted gaze caught, 28% false'),

    # --- OFIQ (ISO/IEC 29794-5). Pose in degrees; quality measures on OFIQ's 0-100 scale, higher is better.
    # Calibrated on the London Set; the eye thresholds also on real photos with glasses, hats and closed eyes from
    # Open Images (scripts/score_labels.py)
    'ofiq_head_yaw': Thresholds(dict(warn_above=5, fail_above=10), '3/4 views: 96-100% caught, 2% false warns'),
    'ofiq_head_pitch': Thresholds(dict(warn_above=8, fail_above=12), 'No variant tests it'),
    'ofiq_head_roll': Thresholds(
        dict(warn_above=4, fail_above=8),
        '12 degrees fails 98%, 1% false fails. Warns from 4 since the review of 2026-11-10 (a visible tilt should '
        'warn): half of the 4-degree variant warns, and 6% of compliant crops, whose real heads tilt up to 4.7'),
    'ofiq_expression': Thresholds(
        dict(warn_below=85, fail_below=50), 'Smiling: 65% caught, 17% warned; faint closed-mouth smiles pass'),
    'ofiq_mouth_closed': Thresholds(
        dict(warn_below=93, fail_below=90),
        'The London Set\'s closed mouths never score below 94; 7 in 10 real open mouths score below 90'),
    'ofiq_eyes_open': Thresholds(
        dict(warn_below=95, fail_below=70), 'Real closed eyes: 56% caught (40% fail), 10% false; 80/50 caught only 8%'),
    'ofiq_eyes_visible': Thresholds(
        dict(warn_below=60, fail_below=25),
        'Sunglasses 100%, real tinted glasses 92% with glasses_tint; 90/50 flagged 61% of clear glasses'),
    'ofiq_face_occlusion': Thresholds(dict(warn_below=90, fail_below=70), 'No labelled set; untested'),
    'ofiq_mouth_occlusion': Thresholds(dict(warn_below=90, fail_below=70), 'Untested'),
    'ofiq_head_covering': Thresholds(dict(warn_below=90, fail_below=50), 'Real hats: 90% caught, 8% false'),
    'ofiq_background': Thresholds(
        dict(warn_below=40, fail_below=25), 'Pattern 100%; objects 2% (the other background checks catch those); 1% false fails'),
    'ofiq_illumination': Thresholds(
        dict(warn_below=72, fail_below=60), 'Side shadow on the face: 98%; a shadow from above is not seen'),
    'ofiq_overexposure': Thresholds(dict(warn_below=90, fail_below=50), 'Brightened: 90% caught, 8% warned'),
    'ofiq_natural_colour': Thresholds(
        dict(warn_below=80, fail_below=60),
        'Every London crop that should pass scores 100; the colour-cast variant scores 39 in the middle, 52-60 on the '
        'ones a fail threshold of 50 only warned on, and 100 on one in ten (OFIQ sees nothing there; colour_cast '
        'catches those). At 60: 88-90% caught; on real candid photos 10% fail instead of 9%'),
}

_WARN_KEYS = ('warn_above', 'warn_below', 'warn_share_above', 'warn_at')
_FAIL_KEYS = ('fail_above', 'fail_below', 'fail_share_above', 'fail_at')


def thresholds(name: str) -> Dict[str, Any]:
    """The threshold keyword arguments for the check `name`, as a fresh dict."""
    return dict(THRESHOLDS[name].kwargs)


def warn_fail(name: str) -> Tuple[Optional[float], Optional[float]]:
    """The (warn, fail) values of the check `name`, whatever their direction, for drawing them as lines.

    None where the check has no such threshold (`skin_smoothing` never fails, `red_eyes` never warns).
    A level turned off (±inf) reads as None too.
    """
    kwargs = THRESHOLDS[name].kwargs
    warn = next((kwargs[key] for key in _WARN_KEYS if key in kwargs), None)
    fail = next((kwargs[key] for key in _FAIL_KEYS if key in kwargs), None)
    return (None if is_off(warn) else warn), (None if is_off(fail) else fail)


# --- The thresholds file: every parameter of every check, overridable per deployment

LEVEL_PREFIXES = ('warn_', 'fail_')

# Levels the prefix does not show (they were named before the convention) or whose off value is not a single
# ±inf (a width and height, a low and high end), by check, with the value that turns each off. The names stay,
# since thresholds files use them; `false` works on them as on any level.
OTHER_LEVELS: Dict[str, Dict[str, Any]] = {
    'resolution': {'min_res': (0, 0), 'recommended_res': (0, 0)},  # no photo is smaller than 0x0: fail, warn
    'photo_format': {'tolerance': math.inf},  # its fail level
    'face_centered': {'vertical_range': (-math.inf, math.inf), 'fail_vertical_range': (-math.inf, math.inf)},
}

# Parameters that were renamed, by check: the old name -> the name it has now. A thresholds file may use either
RENAMED: Dict[str, Dict[str, str]] = {
    'background_brightness': {'min_brightness': 'warn_below'},
    'background_uniformity': {'variance_threshold': 'fail_above'},
    'background_consistency': {'cov_threshold': 'fail_above'},
}


class ThresholdsFileError(ValueError):
    """A thresholds file names a check or a parameter the validator does not have, or gives a wrong type."""


def is_level(key: str, check: Optional[str] = None) -> bool:
    """Whether a parameter is a warn or fail level, the kind `false` turns off. With `check`, the levels
    that check has under other names count too (`OTHER_LEVELS`)."""
    return key.startswith(LEVEL_PREFIXES) or key in OTHER_LEVELS.get(check, {})


def off_value(key: str, check: Optional[str] = None) -> Any:
    """The value that turns a level off: nothing is above +inf, nothing below -inf."""
    if key in OTHER_LEVELS.get(check, {}):
        return OTHER_LEVELS[check][key]
    return -math.inf if key.endswith('_below') else math.inf


def is_off(value: Any) -> bool:
    """Whether a level is turned off (±inf)."""
    return isinstance(value, float) and math.isinf(value)


def level_is_off(check: str, key: str, value: Any) -> bool:
    """Whether `value` turns the level `key` of `check` off: ±inf, or the off value of a level in `OTHER_LEVELS`."""
    if not is_level(key, check):
        return False
    if isinstance(value, (tuple, list)):
        return tuple(value) == off_value(key, check)
    return is_off(value)


def load_overrides(path: str) -> Dict[str, Dict[str, Any]]:
    """Read a TOML thresholds file: one table per check, whose keys are the check's parameters."""
    try:
        with open(path, 'rb') as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise ThresholdsFileError(f'{path}: no such file')
    except tomllib.TOMLDecodeError as exc:
        raise ThresholdsFileError(f'{path}: not valid TOML ({exc})')
    for name, values in data.items():
        if not isinstance(values, dict):
            raise ThresholdsFileError(f'{path}: [{name}] must be a table of parameters, one per line, not a value')
    return data


def parameters(check: Check) -> Dict[str, Any]:
    """The parameters of `check` a thresholds file may set, with the values it runs with, in the order the
    class declares them. A text parameter (an axis, a measure name) says what the check is and is no setting."""
    return {key: value for key, value in check.values().items() if not isinstance(value, str)}


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _kind(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return 'true or false'
    if isinstance(value, Real):
        return 'number'
    if isinstance(value, (tuple, list)):
        return 'list of numbers'
    return None


def _renamed(name: str, values: Dict[str, Any], source: str) -> Dict[str, Any]:
    """The table of one check with old parameter names (`RENAMED`) replaced by the current ones."""
    out = {}
    for key, value in values.items():
        new = RENAMED.get(name, {}).get(key, key)
        if new in out or (new != key and new in values):
            old = next(o for o, n in RENAMED[name].items() if n == new)
            raise ThresholdsFileError(f'{source}: [{name}] sets both {old} and {new}, the old and the new name '
                                      f'of one parameter. Keep {new}')
        out[new] = value
    return out


def disabled_in(overrides: Dict[str, Dict[str, Any]], source: str = 'thresholds file') -> Set[str]:
    """The checks a thresholds file turns off (`enabled = false`)."""
    off = set()
    for name, values in overrides.items():
        if 'enabled' in values:
            if not isinstance(values['enabled'], bool):
                raise ThresholdsFileError(f'{source}: [{name}] enabled must be true or false, not {values["enabled"]!r}')
            if not values['enabled']:
                off.add(name)
    return off


def apply_overrides(checks: Sequence, overrides: Dict[str, Dict[str, Any]], source: str = 'thresholds file') -> Sequence:
    """A sequence of the same checks with `overrides` merged into their parameters; `checks` is left alone.

    Every check name, parameter name and value type is checked first, so a typo fails loudly when the
    validator starts rather than silently keeping the default. `false` on a warn or fail level turns it off,
    `enabled = false` turns the check off; the sequence keeps every check.
    """
    unknown = [name for name in overrides if name not in checks]
    if unknown:
        raise ThresholdsFileError(f'{source}: no check called {", ".join(unknown)}. The checks are: {", ".join(checks.names())}')
    merged = []
    for check in checks:
        name = check.name
        new_kwargs = {}
        allowed = parameters(check)
        for key, value in _renamed(name, overrides.get(name, {}), source).items():
            if key == 'enabled':
                if not isinstance(value, bool):
                    raise ThresholdsFileError(f'{source}: [{name}] enabled must be true or false, not {value!r}')
                new_kwargs['enabled'] = value
                continue
            if key not in allowed:
                takes = ', '.join(allowed) if allowed else 'no parameters'
                raise ThresholdsFileError(f'{source}: [{name}] has no parameter {key}. It takes: enabled, {takes}')
            if is_level(key, name) and value is False:
                new_kwargs[key] = off_value(key, name)
                continue
            current = allowed[key]
            kind = _kind(current) if current is not None else _kind(value)
            if isinstance(current, (tuple, list)):
                # A list is unpacked by the check, so its length and what it holds are checked here too
                kind = f'list of {len(current)} numbers'
                right = isinstance(value, list) and len(value) == len(current) and all(_is_number(v) for v in value)
            else:
                right = kind is not None and kind == _kind(value)
            if not right:
                wanted = kind or 'number'
                if is_level(key, name):
                    wanted += ', or false to turn the level off'
                raise ThresholdsFileError(f'{source}: [{name}] {key} must be a {wanted}, not {value!r}')
            new_kwargs[key] = tuple(value) if isinstance(current, tuple) else value
        merged.append(check.replace(**new_kwargs))
    return Sequence(*merged)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (tuple, list)):
        return '[' + ', '.join(_toml_value(v) for v in value) + ']'
    if is_off(value):
        return 'inf' if value > 0 else '-inf'  # TOML's own spelling; an end of a range that is open
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def to_toml(checks: Sequence, disabled: Set[str] = frozenset()) -> str:
    """Every parameter of every check with the value it runs with, as TOML: the file to edit and pass back.
    A check that is turned off (or named in `disabled`) is written with `enabled = false`."""
    lines = [
        '# Every parameter of every check, with the value the validator runs with.',
        '# Keep the lines to change and delete the rest, then point the validator at the file:',
        '#   uv run passport-validator validate photo.jpg --thresholds thresholds.toml',
        '#   THRESHOLDS_FILE=thresholds.toml uv run passport-validator-api',
        '# A section is a check, its keys are the parameters of that check (core/thresholds.py says where',
        '# the calibrated numbers came from; the others are the defaults of the check classes).',
        '# warn_... and fail_... are the levels: false turns one off (also min_res and recommended_res of',
        '# resolution, tolerance of photo_format and vertical_range of face_centered, which are levels too).',
        '# enabled = false turns the whole check off; a rule only that check covered is then not_checked,',
        '# which never blocks validity.',
        '',
    ]
    for check in checks:
        name = check.name
        found = parameters(check)
        lines.append(f'[{name}]')
        if name in THRESHOLDS:
            lines.extend(textwrap.wrap(THRESHOLDS[name].note, 100, initial_indent='# ', subsequent_indent='# '))
        lines.append(f'enabled = {"false" if name in disabled or not check.enabled else "true"}')
        for key, value in found.items():
            if value is None:
                lines.append(f'# {key}: unset')
            elif level_is_off(name, key, value):
                lines.append(f'{key} = false')
            else:
                lines.append(f'{key} = {_toml_value(value)}')
        lines.append('')
    return '\n'.join(lines)
