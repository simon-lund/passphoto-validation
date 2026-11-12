"""Geometric validation checks for passport photos."""
import math
from typing import Tuple

import numpy as np

from ..check import Check
from ..levels import first_on, grade, is_on, span
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


class FaceSize(Check):
    """Check that the head (chin to crown) covers 70-80% of the photo height.

    The German rule is 32-36 mm of a 45 mm photo, i.e. 71-80%.

    :param warn_below: Head height as a share of the photo height below which it warns.
    :param warn_above: Share above which it warns.
    :param fail_below: Share below which it fails (the head is far too small).
    :param fail_above: Share above which it fails (the head is far too large).
    """

    name = 'face_size'

    warn_below: float = 0.70
    warn_above: float = 0.80
    fail_below: float = 0.60
    fail_above: float = 0.85

    def measure(self, context: ValidationContext):
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

        level = grade(head_height_ratio, warn_below=self.warn_below, warn_above=self.warn_above,
                      fail_below=self.fail_below, fail_above=self.fail_above)
        # The range the message names: the warn levels, and on a side whose warn level is off the fail level
        wanted = span(first_on(self.warn_below, self.fail_below), first_on(self.warn_above, self.fail_above),
                      fmt=lambda share: f'{share * 100:.0f}', unit='%')
        if level == 'fail':
            size = 'large' if is_on(self.fail_above) and head_height_ratio > self.fail_above else 'small'
            context.issues.append(ValidationIssue(
                category='error',
                message=f'Head is too {size} ({head_height_ratio * 100:.1f}% of the photo height). Should be {wanted}.',
                severity=9
            ))
        elif level == 'warn':
            context.issues.append(ValidationIssue(
                category='warning',
                message=f'Head height ({head_height_ratio * 100:.1f}%) is outside the optimal range ({wanted}).',
                severity=5
            ))


class FaceCentered(Check):
    """Check that the face is horizontally centered and at a plausible height in the frame.

    With the head at 70-80% of the photo height, the centre of the face box sits at about 55-65% of
    the height (58-62% on the London Set crops).

    :param warn_offset: Horizontal offset of the face centre (share of the photo width) above which it warns.
    :param fail_offset: Offset above which it fails.
    :param vertical_range: Position of the face centre, as a share of the photo height from the top, outside
        which it warns.
    :param fail_vertical_range: Position outside which it fails; off by default.
    """

    name = 'face_centered'

    warn_offset: float = 0.05
    fail_offset: float = 0.10
    vertical_range: Tuple[float, float] = (0.45, 0.70)
    fail_vertical_range: Tuple[float, float] = (-math.inf, math.inf)

    def measure(self, context: ValidationContext):
        bbox = _bbox(context)

        face_center_x = bbox['x'] + bbox['width'] / 2
        face_center_y = bbox['y'] + bbox['height'] / 2
        horizontal_offset = abs(face_center_x - context.width / 2) / context.width
        vertical_position = face_center_y / context.height
        context.measurements['horizontal_offset'] = float(horizontal_offset)
        context.measurements['vertical_position'] = float(vertical_position)

        self.judge(context, horizontal_offset, warn_above=self.warn_offset, fail_above=self.fail_offset,
                   fail=(f'Face is off centre by {horizontal_offset * 100:.0f}% of the photo width.', 7),
                   warn=('Face is not quite horizontally centered.', 5))

        (low, high), (fail_low, fail_high) = self.vertical_range, self.fail_vertical_range
        level = grade(vertical_position, warn_below=low, warn_above=high, fail_below=fail_low, fail_above=fail_high)
        if level != 'pass':
            # A position past the upper end of the range that was left is too low in the frame (y runs downwards)
            side = 'low' if vertical_position > (fail_high if level == 'fail' else high) else 'high'
            context.issues.append(ValidationIssue(
                category='error' if level == 'fail' else 'warning',
                message=f'Face is positioned {"far " if level == "fail" else ""}too {side} in the frame.',
                severity=7 if level == 'fail' else 5
            ))


class PhotoFormat(Check):
    """Check that the photo has the 35x45 mm format (width to height 7:9).

    :param aspect: Required width / height.
    :param tolerance: Relative deviation from it above which it fails.
    :param warn_tolerance: Deviation above which it warns; off by default.
    """

    name = 'photo_format'

    aspect: float = 35 / 45
    tolerance: float = 0.02
    warn_tolerance: float = math.inf

    def measure(self, context: ValidationContext):
        ratio = context.width / context.height
        context.measurements['aspect_ratio'] = float(ratio)
        deviation = abs(ratio / self.aspect - 1)
        size = f'Photo is {context.width}x{context.height} px (ratio {ratio:.2f})'
        self.judge(context, deviation, warn_above=self.warn_tolerance, fail_above=self.tolerance,
                   fail=(f'{size}; it must have the 35x45 mm format (ratio {self.aspect:.2f}).', 7),
                   warn=(f'{size}; the 35x45 mm format has the ratio {self.aspect:.2f}.', 4))
