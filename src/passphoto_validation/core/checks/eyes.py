"""Eye checks OFIQ doesn't cover: gaze, red eyes, tinted glasses and reflections on glasses."""
import math
from typing import Tuple

import cv2
import numpy as np

from ..levels import judge
from ..models import CheckSkipped, ValidationContext
from ..ofiq import LANDMARK_LEFT_EYE, LANDMARK_RIGHT_EYE, PARSING_LABELS
from ..utils import mesh_points

# MediaPipe face mesh indices with refine_landmarks=True: outer corner, inner corner, upper lid, lower lid,
# iris centre, and the four iris edge points
MP_RIGHT_EYE = dict(outer=33, inner=133, upper=159, lower=145, iris=468, ring=(469, 470, 471, 472))
MP_LEFT_EYE = dict(outer=263, inner=362, upper=386, lower=374, iris=473, ring=(474, 475, 476, 477))


def _gaze_offset(points: np.ndarray, eye: dict) -> Tuple[float, float]:
    """Iris centre relative to the eye centre, as a share of the eye width (x right, y down)."""
    outer, inner = points[eye['outer']], points[eye['inner']]
    width = np.linalg.norm(inner - outer)
    if not width:
        raise CheckSkipped('Could not measure the eyes.')
    centre_x = (outer[0] + inner[0]) / 2
    centre_y = (points[eye['upper']][1] + points[eye['lower']][1]) / 2
    iris = points[eye['iris']]
    return (iris[0] - centre_x) / width, (iris[1] - centre_y) / width


def check_gaze(context: ValidationContext, warn_above: float, fail_above: float):
    """Check that the eyes look into the camera, from where the irises sit within the eyes.

    Looking aside moves both irises the same way, so the offsets of both eyes are averaged; head pose is
    checked separately.

    :param warn_above: Mean horizontal or vertical iris offset (share of eye width) above which it warns.
    :param fail_above: Offset above which it fails.
    """
    points = mesh_points(context, iris=True)
    right = _gaze_offset(points, MP_RIGHT_EYE)
    left = _gaze_offset(points, MP_LEFT_EYE)
    horizontal = (right[0] + left[0]) / 2
    vertical = (right[1] + left[1]) / 2
    context.measurements['gaze_horizontal'] = float(horizontal)
    context.measurements['gaze_vertical'] = float(vertical)

    offset = max(abs(horizontal), abs(vertical))
    judge(context, offset, warn_above=warn_above, fail_above=fail_above,
          fail=('Eyes are not looking into the camera.', 8),
          warn=('Eyes may not be looking straight into the camera.', 5))


def _disk(context: ValidationContext, centre, radius: float) -> np.ndarray:
    mask = np.zeros((context.height, context.width), np.uint8)
    cv2.circle(mask, (int(round(centre[0])), int(round(centre[1]))), max(1, int(round(radius))), 1, -1)
    return mask.astype(bool)


def check_red_eyes(context: ValidationContext, fail_above: float, warn_above: float = math.inf,
                   pupil_share: float = 0.5, redder_than_skin: float = 2.0):
    """Check for red eyes (flash reflected by the retina): pupil pixels much redder than the skin below the eye.

    Redness is red over the larger of green and blue. Comparing with the skin keeps a warm colour cast, and
    eyelid skin over a half-closed eye, from counting as red eyes.

    :param fail_above: Share of red pupil pixels in the redder eye above which it fails.
    :param warn_above: Share above which it warns; off by default.
    :param pupil_share: Pupil radius as a share of the iris radius.
    :param redder_than_skin: How many times as red as the skin below the eye a pupil pixel must be.
    """
    points = mesh_points(context, iris=True)
    b, g, r = (context.image_np[..., i].astype(np.float32) for i in range(3))
    redness = r / np.maximum(1, np.maximum(g, b))

    shares = []
    for eye in (MP_RIGHT_EYE, MP_LEFT_EYE):
        centre = points[eye['iris']]
        width = np.linalg.norm(points[eye['inner']] - points[eye['outer']])
        pupil = _disk(context, centre, pupil_share * np.mean([np.linalg.norm(points[i] - centre) for i in eye['ring']]))
        skin = _disk(context, (centre[0], points[eye['lower']][1] + 0.25 * width), 0.12 * width)
        if pupil.any() and skin.any():
            red = (r > 50) & (redness > redder_than_skin * np.median(redness[skin]))
            shares.append(float(red[pupil].mean()))
    if not shares:
        raise CheckSkipped('Could not find the pupils.')

    context.measurements['red_eye_share'] = max(shares)
    judge(context, max(shares), warn_above=warn_above, fail_above=fail_above,
          fail=('Red eyes detected. Retake the photo without red-eye.', 8), warn=('The pupils may be red.', 5))


def _glasses(context: ValidationContext):
    """The parsing map, the eye landmarks and whether glasses are worn (from OFIQ's face parsing)."""
    if context.ofiq is None:
        raise CheckSkipped(context.ofiq_unavailable or 'OFIQ did not run on this photo.')
    parsing, landmarks = context.ofiq.parsing, context.ofiq.landmarks
    if parsing is None or landmarks is None:
        raise CheckSkipped('OFIQ found no face landmarks.')

    eyes = [landmarks[list(LANDMARK_RIGHT_EYE)], landmarks[list(LANDMARK_LEFT_EYE)]]
    eye_distance = float(np.linalg.norm(eyes[0].mean(axis=0) - eyes[1].mean(axis=0)))
    if not eye_distance:
        raise CheckSkipped('Could not measure the eyes.')
    glasses_share = float((parsing == PARSING_LABELS['glasses']).sum() / eye_distance ** 2)
    context.measurements['glasses_share'] = glasses_share
    return parsing, eyes, eye_distance, glasses_share


def _lens_area(context: ValidationContext, parsing: np.ndarray, eye: np.ndarray, eye_distance: float) -> np.ndarray:
    """Boolean mask of the lens area around one eye (glasses or skin pixels in a box around the eye)."""
    cx, cy = eye.mean(axis=0)
    rows, cols = np.ogrid[:context.height, :context.width]
    box = (np.abs(cols - cx) <= 0.32 * eye_distance) & (np.abs(rows - cy) <= 0.2 * eye_distance)
    return box & np.isin(parsing, [PARSING_LABELS['glasses'], PARSING_LABELS['skin']])


def _under_eye_and_cheek(context: ValidationContext, parsing: np.ndarray, eye: np.ndarray, eye_distance: float):
    """Boolean masks of the skin just below one eye (behind the lens) and of the cheek further down (outside it).

    The band below the eye avoids the frame, the iris and the lashes, which are dark whatever the lens.
    """
    cx, cy = eye.mean(axis=0)
    rows, cols = np.ogrid[:context.height, :context.width]
    columns = np.abs(cols - cx) <= 0.18 * eye_distance
    under_eye = columns & (rows >= cy + 0.1 * eye_distance) & (rows <= cy + 0.2 * eye_distance)
    cheek = columns & (rows >= cy + 0.45 * eye_distance) & (rows <= cy + 0.7 * eye_distance) & (parsing == PARSING_LABELS['skin'])
    return under_eye, cheek


def check_glasses_tint(context: ValidationContext, min_glasses_share: float, warn_below: float, fail_below: float):
    """Check that glasses, if worn, are not tinted: the skin just below the eyes, behind the lenses, must be
    about as light as the cheeks below the lenses.

    The ratio is 0.6-1.0 without tint (under-eye skin is often a little darker) and about 0.2-0.3 behind
    dark lenses. Opaque lenses also lower OFIQ's EyesVisible, which covers the same rule.

    :param min_glasses_share: Glasses pixels per squared eye distance above which glasses count as worn.
    :param warn_below: Under-eye to cheek lightness ratio below which it warns.
    :param fail_below: Ratio below which it fails.
    """
    parsing, eyes, eye_distance, glasses_share = _glasses(context)
    if glasses_share < min_glasses_share:
        context.measurements['glasses_worn'] = False
        return
    context.measurements['glasses_worn'] = True

    lightness = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    ratios = []
    for eye in eyes:
        under_eye, cheek = _under_eye_and_cheek(context, parsing, eye, eye_distance)
        if under_eye.sum() >= 20 and cheek.sum() >= 20:
            ratios.append(float(np.median(lightness[under_eye]) / max(1.0, np.median(lightness[cheek]))))
    if not ratios:
        raise CheckSkipped('Could not measure the lenses.')

    ratio = min(ratios)
    context.measurements['lens_brightness_ratio'] = ratio
    judge(context, ratio, warn_below=warn_below, fail_below=fail_below,
          fail=('Glasses appear tinted. Tinted glasses and sunglasses are not allowed.', 8),
          warn=('Glasses may be tinted.', 5))


def check_glasses_glare(context: ValidationContext, min_glasses_share: float, warn_above: float, fail_above: float):
    """Check for reflections on glasses: near-white, colourless pixels within the lenses.

    Passes when no glasses are worn.

    :param min_glasses_share: Glasses pixels per squared eye distance above which glasses count as worn.
    :param warn_above: Share of glare pixels in the lens area above which it warns.
    :param fail_above: Share above which it fails.
    """
    parsing, eyes, eye_distance, glasses_share = _glasses(context)
    if glasses_share < min_glasses_share:
        context.measurements['glasses_worn'] = False
        return
    context.measurements['glasses_worn'] = True

    hsv = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2HSV)
    specular = (hsv[..., 2] >= 235) & (hsv[..., 1] <= 40)
    lens = np.zeros(parsing.shape, bool)
    for eye in eyes:
        lens |= _lens_area(context, parsing, eye, eye_distance)
    if lens.sum() < 20:
        raise CheckSkipped('Could not measure the lenses.')

    share = float(specular[lens].mean())
    context.measurements['lens_glare_share'] = share
    judge(context, share, warn_above=warn_above, fail_above=fail_above,
          fail=('Reflections on the glasses. Tilt the glasses or change the lighting.', 8),
          warn=('Possible reflections on the glasses.', 5))


def _siglip_score(context: ValidationContext, question: str, measurement: str) -> float:
    if context.siglip is None:
        raise CheckSkipped(context.siglip_unavailable or 'SigLIP did not run on this photo.')
    score = context.siglip[question]
    context.measurements[measurement] = score
    return score


def check_glasses_glare_siglip(context: ValidationContext, min_glasses_share: float, warn_above: float, fail_above: float = math.inf):
    """Second opinion on reflections on glasses: SigLIP's zero-shot score on the eye region (see core/siglip.py).

    It sees soft sheens and mirrored windows that check_glasses_glare misses, but less surely, so it only
    warns. Passes when no glasses are worn.

    :param min_glasses_share: Glasses pixels per squared eye distance above which glasses count as worn.
    :param warn_above: Score (logit of reflection prompts minus clean-lens prompts) above which it warns.
    :param fail_above: Score above which it fails; off by default.
    """
    score = _siglip_score(context, 'glare', 'siglip_glare_score')
    if _glasses(context)[3] < min_glasses_share:
        return
    finding = ('There may be reflections on the glasses.', 5)
    judge(context, score, warn_above=warn_above, fail_above=fail_above, fail=finding, warn=finding)


def check_gaze_siglip(context: ValidationContext, warn_above: float, fail_above: float = math.inf):
    """Second opinion on gaze: SigLIP's zero-shot score on the eye region (see core/siglip.py).

    It judges the whole eye region instead of the iris position, and ranked averted gazes on real photos
    better than check_gaze (AUC 0.83 against 0.76), but it only warns.

    :param warn_above: Score (logit of looking-away prompts minus looking-at-camera prompts) above which it warns.
    :param fail_above: Score above which it fails; off by default.
    """
    score = _siglip_score(context, 'gaze_aside', 'siglip_gaze_score')
    finding = ('The eyes may not be looking into the camera.', 5)
    judge(context, score, warn_above=warn_above, fail_above=fail_above, fail=finding, warn=finding)
