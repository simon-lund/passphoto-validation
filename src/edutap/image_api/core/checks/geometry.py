"""Geometric validation checks for passport photos."""
from typing import Tuple

import cv2
import numpy as np

from ..models import CheckSkipped, ValidationIssue, ValidationContext

# MediaPipe face mesh indices
NOSE_TIP = 1
CHIN = 152
FOREHEAD = 10  # top of the forehead, roughly the hairline

# How far the crown sits above the forehead landmark, as a share of the forehead-to-chin distance.
# Measured on a bald subject; used when the top of the head is not visible in the photo.
_CROWN_ABOVE_FOREHEAD = 0.32
# Hair above this share is ignored, since voluminous hair does not count towards head height.
_MAX_HAIR_ABOVE_FOREHEAD = 0.45


def _landmarks(context: ValidationContext):
    landmarks = context.face_data.get('landmarks')
    if not landmarks:
        raise CheckSkipped('No facial landmarks found.')
    return landmarks.landmark


def _bbox(context: ValidationContext):
    bbox = context.face_data.get('bbox')
    if not bbox:
        raise CheckSkipped('No face found.')
    return bbox


def _crown_y(context: ValidationContext, forehead_y: float, chin_y: float) -> Tuple[float, str]:
    """Estimate the y coordinate of the top of the head and say how it was found ('mask' or 'estimated')."""
    face_height = chin_y - forehead_y
    estimated = forehead_y - _CROWN_ABOVE_FOREHEAD * face_height

    mask = context.person_mask_binary
    if mask is None:
        return estimated, 'estimated'

    # Look for the first person pixel in a narrow column above the nose
    nose_x = int(np.clip(_landmarks(context)[NOSE_TIP].x * context.width, 0, context.width - 1))
    half_band = max(1, context.width // 50)
    column = mask[:max(1, int(forehead_y)), max(0, nose_x - half_band):nose_x + half_band + 1].any(axis=1)
    if not column.any():
        return estimated, 'estimated'

    top = int(np.argmax(column))
    if top == 0:
        # The head (or a mask leak) runs into the top edge, so the crown is not visible
        return estimated, 'estimated'
    return max(float(top), forehead_y - _MAX_HAIR_ABOVE_FOREHEAD * face_height), 'mask'


def check_face_size(context: ValidationContext):
    """Check that the head (chin to crown) covers 70-80% of the photo height.

    The German rule is 32-36 mm of a 45 mm photo, i.e. 71-80%.

    :param context: The validation context containing the image and results.
    """
    bbox = _bbox(context)
    lm = _landmarks(context)

    chin_y = lm[CHIN].y * context.height
    forehead_y = lm[FOREHEAD].y * context.height
    crown_y, crown_method = _crown_y(context, forehead_y, chin_y)

    head_height_ratio = (chin_y - crown_y) / context.height
    context.measurements['face_height_ratio'] = float(bbox['height'] / context.height)
    context.measurements['head_height_ratio'] = float(head_height_ratio)
    context.measurements['crown_method'] = crown_method
    context.measurements['crown_y'] = float(crown_y)
    context.measurements['chin_y'] = float(chin_y)

    if head_height_ratio < 0.60:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'Head is too small ({head_height_ratio * 100:.1f}% of the photo height). Should be 70-80%.',
            severity=9
        ))
    elif head_height_ratio > 0.85:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'Head is too large ({head_height_ratio * 100:.1f}% of the photo height). Should be 70-80%.',
            severity=9
        ))
    elif head_height_ratio < 0.70 or head_height_ratio > 0.80:
        context.issues.append(ValidationIssue(
            category='warning',
            message=f'Head height ({head_height_ratio * 100:.1f}%) is outside the optimal range (70-80%).',
            severity=5
        ))


def check_face_centered(context: ValidationContext, warn_offset: float = 0.05, fail_offset: float = 0.10,
                        vertical_range: Tuple[float, float] = (0.45, 0.70)):
    """Check that the face is horizontally centered and at a plausible height in the frame.

    With the head at 70-80% of the photo height, the centre of the face box sits at about 55-65% of
    the height (58-62% on the London Set crops).

    :param warn_offset: Horizontal offset of the face centre (share of the photo width) above which it warns.
    :param fail_offset: Offset above which it fails.
    :param vertical_range: Allowed position of the face centre, as a share of the photo height from the top.
    """
    bbox = _bbox(context)

    face_center_x = bbox['x'] + bbox['width'] / 2
    face_center_y = bbox['y'] + bbox['height'] / 2
    horizontal_offset = abs(face_center_x - context.width / 2) / context.width
    vertical_position = face_center_y / context.height
    context.measurements['horizontal_offset'] = float(horizontal_offset)
    context.measurements['vertical_position'] = float(vertical_position)

    if horizontal_offset > fail_offset:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'Face is off centre by {horizontal_offset * 100:.0f}% of the photo width.',
            severity=7
        ))
    elif horizontal_offset > warn_offset:
        context.issues.append(ValidationIssue(
            category='warning',
            message='Face is not quite horizontally centered.',
            severity=5
        ))

    low, high = vertical_range
    if not low <= vertical_position <= high:
        context.issues.append(ValidationIssue(
            category='warning',
            message=f'Face is positioned too {"low" if vertical_position > high else "high"} in the frame.',
            severity=5
        ))


def check_photo_format(context: ValidationContext, aspect: float = 35 / 45, tolerance: float = 0.02):
    """Check that the photo has the 35x45 mm format (width to height 7:9).

    :param aspect: Required width / height.
    :param tolerance: Allowed relative deviation from it.
    """
    ratio = context.width / context.height
    context.measurements['aspect_ratio'] = float(ratio)
    deviation = abs(ratio / aspect - 1)
    if deviation > tolerance:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'Photo is {context.width}x{context.height} px (ratio {ratio:.2f}); it must have the 35x45 mm format (ratio {aspect:.2f}).',
            severity=7
        ))
