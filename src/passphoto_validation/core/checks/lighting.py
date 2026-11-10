"""Exposure, contrast and colour cast checks.

Overexposure and shadows on the face come from OFIQ (checks/ofiq.py). Underexposure is judged here
instead: OFIQ's UnderExposurePrevention counts dark face pixels, so it scores well-lit dark skin as
underexposed (on the London Set it was below 100 only for Black subjects). Colour casts are judged by
OFIQ's natural colour on the skin and here by a tint the background and the whites of the eyes share.
"""
import cv2
import numpy as np

from ..models import CheckSkipped, ValidationIssue, ValidationContext
from ..utils import mesh_points
from .eyes import MP_LEFT_EYE, MP_RIGHT_EYE

# MediaPipe face mesh indices around each eye (the lid contours, 16 points each)
RIGHT_EYE_CONTOUR = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
LEFT_EYE_CONTOUR = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)


def check_exposure_level(context: ValidationContext, warn_below: float, fail_below: float):
    """Check that the photo is not underexposed, from its brightest parts rather than from the skin.

    An ID photo needs a light background, so in a correctly exposed photo the brightest 1% of the pixels
    are light whatever the skin tone (L* 83-100 on the London Set; darkened copies stay below 40).

    :param warn_below: 99th percentile of the lightness L* (0-100) below which it warns.
    :param fail_below: Lightness below which it fails.
    """
    lightness = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * (100 / 255)
    highlights = float(np.percentile(lightness, 99))
    context.measurements['highlight_lightness'] = highlights

    if highlights < fail_below:
        context.issues.append(ValidationIssue(
            category='error', message='Photo is too dark (underexposed). Increase the lighting.', severity=8))
    elif highlights < warn_below:
        context.issues.append(ValidationIssue(
            category='warning', message='Photo may be too dark. Check the lighting.', severity=5))


def check_contrast(context: ValidationContext, warn_below: float = 40.0, fail_below: float = 30.0):
    """Check overall contrast: the standard deviation of the grey levels.

    :param warn_below: Standard deviation (0-255 scale) below which the photo looks washed out.
    :param fail_below: Standard deviation below which it fails (the low-contrast and darkened London variants stay under 30).
    """
    gray = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2GRAY)

    contrast = gray.std()
    context.measurements['contrast'] = float(contrast)

    if contrast < fail_below:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'Contrast is far too low (grey-level spread {contrast:.0f}). The photo looks grey.',
            severity=7
        ))
    elif contrast < warn_below:
        context.issues.append(ValidationIssue(
            category='warning',
            message=f'Low contrast (grey-level spread {contrast:.0f}). Image may appear washed out.',
            severity=5
        ))


def eye_whites(context: ValidationContext, points: np.ndarray) -> np.ndarray:
    """Mask of the whites of the eyes: inside each lid contour, outside the iris, the lighter half (the
    lids and lashes are darker than the sclera)."""
    lightness = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2LAB)[..., 0]
    whites = np.zeros((context.height, context.width), bool)
    for contour, eye in ((RIGHT_EYE_CONTOUR, MP_RIGHT_EYE), (LEFT_EYE_CONTOUR, MP_LEFT_EYE)):
        region = np.zeros((context.height, context.width), np.uint8)
        cv2.fillPoly(region, [points[list(contour)].round().astype(np.int32)], 1)
        centre = points[eye['iris']]
        radius = 1.15 * np.mean([np.linalg.norm(points[i] - centre) for i in eye['ring']])
        cv2.circle(region, (int(round(centre[0])), int(round(centre[1]))), max(1, int(round(radius))), 0, -1)
        region = region.astype(bool)
        if region.any():
            whites |= region & (lightness >= np.percentile(lightness[region], 50))
    return whites


def check_colour_cast(context: ValidationContext, warn_above: float, fail_above: float,
                      min_whites_share: float = 0.02, band_share: float = 0.03):
    """Check for a colour cast: a tint the background and the whites of the eyes share.

    OFIQ's natural colour judges the skin, and passes a cast when the tinted skin still lies in the natural
    range. Two things in an ID photo have no colour of their own: the backdrop and the whites of the eyes. Each
    is averaged in CIELAB (a*, b*), one band away from the person for the background. The measure is the chroma
    the two share: the shorter colour vector projected on the longer, and 0 when they point different ways,
    so a coloured backdrop behind natural eyes does not count, nor does a cast that only reaches the face.

    :param warn_above: Shared chroma above which it warns.
    :param fail_above: Shared chroma above which it fails.
    :param min_whites_share: Whites pixels per squared eye width below which the eyes are too closed to judge.
    :param band_share: Width of the band next to the person left out of the background, as a share of the
        short side.
    """
    points = mesh_points(context, iris=True)
    mask = context.person_mask_binary
    if mask is None:
        raise CheckSkipped('No segmentation mask present.')
    band = max(3, int(round(min(context.width, context.height) * band_share)))
    distance = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    background = (mask == 0) & (distance > band)
    if background.sum() < 100:
        raise CheckSkipped('No background visible.')

    whites = eye_whites(context, points)
    eye_width = np.mean([np.linalg.norm(points[eye['inner']] - points[eye['outer']]) for eye in (MP_RIGHT_EYE, MP_LEFT_EYE)])
    whites_share = float(whites.sum() / max(1.0, eye_width ** 2))
    context.measurements['eye_whites_share'] = whites_share
    if whites_share < min_whites_share:
        raise CheckSkipped('The whites of the eyes are not visible enough to judge the colour.')

    lab = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    background_colour = np.array([a[background].mean(), b[background].mean()])
    whites_colour = np.array([a[whites].mean(), b[whites].mean()])
    context.measurements['background_colour_a'] = float(background_colour[0])
    context.measurements['background_colour_b'] = float(background_colour[1])
    context.measurements['eye_whites_colour_a'] = float(whites_colour[0])
    context.measurements['eye_whites_colour_b'] = float(whites_colour[1])

    chroma = np.linalg.norm(background_colour), np.linalg.norm(whites_colour)
    agreement = float(background_colour @ whites_colour / (chroma[0] * chroma[1])) if min(chroma) > 1e-6 else 0.0
    cast = max(agreement, 0.0) * float(min(chroma))
    context.measurements['colour_cast'] = cast

    if cast > fail_above:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'The photo has a colour cast: the background and the whites of the eyes are tinted the '
                    f'same way (shared chroma {cast:.0f}). Colours must be natural.',
            severity=7))
    elif cast > warn_above:
        context.issues.append(ValidationIssue(
            category='warning',
            message=f'The photo may have a colour cast: the background and the whites of the eyes share a '
                    f'tint (shared chroma {cast:.0f}).',
            severity=4))
