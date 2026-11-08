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
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


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
    'exposure': Thresholds(
        dict(warn_below=75, fail_below=60),
        'L* of the brightest 1% (the background), so dark skin is not read as underexposed; the darkened variant fails'),
    'background_brightness': Thresholds(
        dict(min_brightness=120, fail_below=80),
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
        dict(warn_above=5, fail_above=9),
        'L* the band next to the person is darker than that plane; compliant crops reach 9 (one photo; 98% stay '
        'under 5), the shadow variant starts at 3 (90% above 5). The margin is thin'),
    'print_defects': Thresholds(
        dict(fail_at=3),
        'Marks on the background; any mark warns, three fail. Compliant crops show up to 2 (stray hair), the ink '
        'variant 3 or more'),
    'skin_smoothing': Thresholds(
        dict(warn_below=0.15, warn_gradient_ratio_below=0.115),
        'Cheek and forehead texture over eye texture at 192 px face width; filtered skin stays under 0.14, '
        'natural skin above 0.19. Skin texture over the median gradient of the face, for a filter that smooths '
        'the eyes too: compliant, glasses, cast, shadow, dark and low-contrast crops stay above 0.125, the '
        'whole-face smoothed variant under 0.11 (smoothed_all). Warns only'),
    'skin_colour_patches': Thresholds(
        dict(warn_share_above=0.02, chroma_distance=30),
        'Patches of the lower face more than 30 CIELAB chroma units from the skin, at least as saturated: 0 on '
        'every London crop (plain, glasses, sunglasses, cast, shadow), 3-4% of the face oval with the sticker '
        'variant (face_stickers); 2% of real candid photos warn (mirrored goggles, a clown nose, a glass or fruit '
        'at the mouth). Warns only'),
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
    'ofiq_head_roll': Thresholds(dict(warn_above=6, fail_above=8), '12 degrees fails 98%; 4 degrees passes; 1% false fails'),
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
    'ofiq_natural_colour': Thresholds(dict(warn_below=80, fail_below=50), 'Colour cast: 81% caught, 10% warned'),
}

_WARN_KEYS = ('warn_above', 'warn_below', 'min_brightness')
_FAIL_KEYS = ('fail_above', 'fail_below', 'fail_at')


def thresholds(name: str) -> Dict[str, Any]:
    """The threshold keyword arguments for the check `name`, as a fresh dict."""
    return dict(THRESHOLDS[name].kwargs)


def warn_fail(name: str) -> Tuple[Optional[float], Optional[float]]:
    """The (warn, fail) values of the check `name`, whatever their direction, for drawing them as lines.

    None where the check has no such threshold (`skin_smoothing` never fails, `red_eyes` never warns).
    `background_brightness` warns below `min_brightness`.
    """
    kwargs = THRESHOLDS[name].kwargs
    warn = next((kwargs[key] for key in _WARN_KEYS if key in kwargs), None)
    fail = next((kwargs[key] for key in _FAIL_KEYS if key in kwargs), None)
    return warn, fail
