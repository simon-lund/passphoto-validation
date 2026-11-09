"""Exposure and contrast checks.

Overexposure and shadows on the face come from OFIQ (checks/ofiq.py). Underexposure is judged here
instead: OFIQ's UnderExposurePrevention counts dark face pixels, so it scores well-lit dark skin as
underexposed (on the London Set it was below 100 only for Black subjects).
"""
import cv2
import numpy as np

from ..models import ValidationIssue, ValidationContext


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


def check_contrast(context: ValidationContext):
    """Check overall contrast.

    :param context: The validation context containing the image and results.
    """
    gray = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2GRAY)

    contrast = gray.std()
    context.measurements['contrast'] = float(contrast)

    if contrast < 30:
        context.issues.append(ValidationIssue(
            category='warning',
            message='Low contrast. Image may appear washed out.',
            severity=5
        ))
