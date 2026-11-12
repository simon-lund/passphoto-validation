"""Reflections in the upper face (glare). Eyes and expression come from OFIQ (checks/ofiq.py)."""
import cv2
import numpy as np

from ..levels import judge
from ..models import CheckSkipped, ValidationContext

def check_glasses(context: ValidationContext, bright_level: int = 240, warn_above: float = 0.02, fail_above: float = 0.05):
    """Check for glare (bright reflections) in the upper half of the face, e.g. on glasses.

    This does not detect whether glasses are worn; it only counts near-white pixels.

    :param bright_level: Grey level (0-255) from which a pixel counts as a reflection.
    :param warn_above: Share of such pixels in the upper half of the face box above which it warns.
    :param fail_above: Share above which it fails.
    """
    bbox = context.face_data.get('bbox')
    if not bbox:
        raise CheckSkipped('No face found.')

    eye_region = context.image_np[
        max(0, bbox['y']):max(0, bbox['y'] + bbox['height'] // 2),
        max(0, bbox['x']):max(0, bbox['x'] + bbox['width'])
    ]

    if eye_region.size == 0:
        raise CheckSkipped('Face region is outside the photo.')

    gray_eyes = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

    _, bright_spots = cv2.threshold(gray_eyes, bright_level, 255, cv2.THRESH_BINARY)
    bright_pixel_ratio = np.sum(bright_spots > 0) / bright_spots.size

    context.measurements['eye_reflection_ratio'] = float(bright_pixel_ratio)

    judge(context, bright_pixel_ratio, warn_above=warn_above, fail_above=fail_above,
          fail=('Glare or reflections detected on glasses. Adjust angle or remove glasses.', 8),
          warn=('Possible reflections on glasses detected.', 5))
