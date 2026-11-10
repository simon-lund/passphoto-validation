"""Every check's thresholds in one place, with where each number came from.

The validator reads this table when it registers its checks (`PassportPhotoValidator._checks`), the
evaluation notebook draws the same numbers as the warn and fail lines of its panels (`warn_fail`), and
PASSPORT_VALIDATOR.md points here. To change a threshold: edit it here, run the tests (`tests/test_thresholds.py`
pins the table, so update its snapshot on purpose) and re-run `notebooks/evaluation.ipynb` to see the effect.

The keyword names are those of the check functions. `warn_above` / `fail_above` mean higher is worse,
`warn_below` / `fail_below` lower is worse, `fail_at` is a count. A check not listed here runs with the
defaults of its function signature (`photo_format`, `face_size`, `face_centered`, `contrast`,
`background_uniformity`, `background_consistency`, `glare`); `resolution` takes its minimums from
`ValidatorConfig`, where they can be set per deployment.

A deployment can override any parameter of any check without editing this file: a TOML thresholds file
(`ValidatorConfig.thresholds_file`, `passport-validator --thresholds`), one table per check, is read by
`load_overrides` and merged by `apply_overrides`, which refuses unknown checks, unknown parameters and
wrong types. `to_toml` writes every parameter with its current value, to start such a file from. The file
is applied last, after this table and `ValidatorConfig` (so its `[resolution]` wins over `MIN_RES_*`).

The convention of that file: every check has a warn level and a fail level (`warn_…`, `fail_…`); `false`
turns a level off (in the code an off level is +inf or -inf, which no measurement reaches, so the checks
need no special case); `enabled = false` turns a check off: it is not run, and a rule only it covered
ends as not_checked, which never blocks validity. A check whose measure has no sensible second level
(`photo_format`) keeps the one it has.
"""
import inspect
import math
import textwrap
import tomllib
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


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
        'flag in Open Images measures 34 (background labels: openimages-faces/eval/vlm_background_labels.jsonl)'),
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

Check = Tuple[str, Callable, Dict[str, Any]]  # (name, function, keyword arguments), as the validator registers them
LEVEL_PREFIXES = ('warn_', 'fail_')


class ThresholdsFileError(ValueError):
    """A thresholds file names a check or a parameter the validator does not have, or gives a wrong type."""


def is_level(key: str) -> bool:
    """Whether a parameter is a warn or fail level, the kind `false` turns off."""
    return key.startswith(LEVEL_PREFIXES)


def off_value(key: str) -> float:
    """The value that turns a level off: nothing is above +inf, nothing below -inf."""
    return -math.inf if key.endswith('_below') else math.inf


def is_off(value: Any) -> bool:
    """Whether a level is turned off (±inf)."""
    return isinstance(value, float) and math.isinf(value)


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


def parameters(check: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """The parameters of `check` a thresholds file may set, with the values it runs with: the keyword arguments
    the validator passes, then the function's own defaults. Names (an axis, a measure name) and debug flags are
    not parameters."""
    found = {}
    for name, parameter in inspect.signature(check).parameters.items():
        if name == 'context' or name == 'debug':
            continue
        value = kwargs.get(name, parameter.default)
        if value is inspect.Parameter.empty or isinstance(value, str):
            continue
        found[name] = value
    return found


def _kind(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return 'true or false'
    if isinstance(value, Real):
        return 'number'
    if isinstance(value, (tuple, list)):
        return 'list of numbers'
    return None


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


def apply_overrides(checks: List[Check], overrides: Dict[str, Dict[str, Any]], source: str = 'thresholds file') -> List[Check]:
    """The check list with `overrides` merged into each check's keyword arguments.

    Every check name, parameter name and value type is checked first, so a typo fails loudly when the
    validator starts rather than silently keeping the default. `false` on a warn or fail level turns it off;
    `enabled` is read by `disabled_in`, the list keeps every check.
    """
    by_name = {name: (check, kwargs) for name, check, kwargs in checks}
    unknown = [name for name in overrides if name not in by_name]
    if unknown:
        raise ThresholdsFileError(f'{source}: no check called {", ".join(unknown)}. The checks are: {", ".join(by_name)}')
    merged = []
    for name, check, kwargs in checks:
        new_kwargs = dict(kwargs)
        allowed = parameters(check, kwargs)
        for key, value in overrides.get(name, {}).items():
            if key == 'enabled':
                if not isinstance(value, bool):
                    raise ThresholdsFileError(f'{source}: [{name}] enabled must be true or false, not {value!r}')
                continue
            if key not in allowed:
                takes = ', '.join(allowed) if allowed else 'no parameters'
                raise ThresholdsFileError(f'{source}: [{name}] has no parameter {key}. It takes: enabled, {takes}')
            if is_level(key) and value is False:
                new_kwargs[key] = off_value(key)
                continue
            current = allowed[key]
            kind = _kind(current) if current is not None else _kind(value)
            if kind is None or kind != _kind(value):
                wanted = kind or 'number'
                if is_level(key):
                    wanted += ', or false to turn the level off'
                raise ThresholdsFileError(f'{source}: [{name}] {key} must be a {wanted}, not {value!r}')
            new_kwargs[key] = tuple(value) if isinstance(current, tuple) else value
        merged.append((name, check, new_kwargs))
    return merged


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if is_off(value):
        return 'false'
    if isinstance(value, (tuple, list)):
        return '[' + ', '.join(_toml_value(v) for v in value) + ']'
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def to_toml(checks: List[Check], disabled: Set[str] = frozenset()) -> str:
    """Every parameter of every check with the value it runs with, as TOML: the file to edit and pass back."""
    lines = [
        '# Every parameter of every check, with the value the validator runs with.',
        '# Keep the lines to change and delete the rest, then point the validator at the file:',
        '#   uv run passport-validator validate photo.jpg --thresholds thresholds.toml',
        '#   THRESHOLDS_FILE=thresholds.toml uv run passport-validator-api',
        '# A section is a check, its keys are the parameters of that check (core/thresholds.py says where',
        '# the calibrated numbers came from; the others are the defaults of the check functions).',
        '# warn_... and fail_... are the levels: false turns one off. enabled = false turns the whole check off;',
        '# a rule only that check covered is then not_checked, which never blocks validity.',
        '',
    ]
    for name, check, kwargs in checks:
        found = parameters(check, kwargs)
        lines.append(f'[{name}]')
        if name in THRESHOLDS:
            lines.extend(textwrap.wrap(THRESHOLDS[name].note, 100, initial_indent='# ', subsequent_indent='# '))
        lines.append(f'enabled = {"false" if name in disabled else "true"}')
        for key, value in found.items():
            if value is None:
                lines.append(f'# {key}: unset')
            else:
                lines.append(f'{key} = {_toml_value(value)}')
        lines.append('')
    return '\n'.join(lines)
