"""Plan a 35x45 mm crop that frames the head the way the rules ask.

A crop can fix the framing rules (format, head size, position) and nothing else: everything about the
person and the shot stays as it is. The planner only proposes a box inside the photo, never one that would
need background added (that would be editing the photo), and the validator then checks the cropped photo
like any other before calling it valid.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

ASPECT = 35 / 45
# The checks whose problems a crop can fix
FRAMING_CHECKS = ('photo_format', 'face_size', 'face_centered')

# Head (chin to crown) as a share of the crop height. check_face_size allows 70-80% (32-36 mm of 45);
# the crop stays inside that with a margin, since the head is measured again on the cropped photo.
HEAD_RANGE = (0.72, 0.79)
# Centre of the face box as a share of the crop height from the top. check_face_centered allows 45-70%;
# the London Set crops sit at 58-62%.
FACE_CENTRE = 0.60
FACE_CENTRE_RANGE = (0.47, 0.68)
# How far the face centre may sit off the crop's middle, as a share of the crop width (the check warns above 5%)
MAX_OFFSET = 0.03
# Head sizes tried, from the one the photo already has (or the nearest allowed) outwards
_STEPS = 15


@dataclass
class CropBox:
    """A crop in the photo's pixels."""
    x: int
    y: int
    width: int
    height: int


def plan_crop(width: int, height: int, bbox: Dict, chin_y: float, crown_y: float) -> Tuple[Optional[CropBox], Optional[str]]:
    """Find the 35x45 box inside the photo that frames the head as the rules ask.

    It keeps the head size the photo has when that is allowed (else the nearest allowed one), centres the
    face and puts the face centre at 60% of the height, giving way on each within the margins above to
    stay inside the photo.

    :param width: Photo width in pixels.
    :param height: Photo height in pixels.
    :param bbox: Face box (x, y, width, height), as in the report.
    :param chin_y: Chin height in pixels, as check_face_size measures it.
    :param crown_y: Crown height in pixels (estimated when hair is cut off at the top).
    :return: (box, None), or (None, why no box fits).
    """
    head = chin_y - crown_y
    if head <= 0:
        return None, 'The head height could not be measured.'
    centre_x = bbox['x'] + bbox['width'] / 2
    centre_y = bbox['y'] + bbox['height'] / 2

    low, high = HEAD_RANGE
    preferred = min(max(head / height, low), high)
    ratios = sorted((low + (high - low) * i / (_STEPS - 1) for i in range(_STEPS)), key=lambda r: abs(r - preferred))
    for ratio in [preferred] + ratios:
        crop_height = head / ratio
        crop_width = crop_height * ASPECT
        if crop_height > height or crop_width > width:
            continue
        x = _place(centre_x - crop_width / 2, centre_x - crop_width * (0.5 + MAX_OFFSET),
                   centre_x - crop_width * (0.5 - MAX_OFFSET), width - crop_width)
        y = _place(centre_y - crop_height * FACE_CENTRE, centre_y - crop_height * FACE_CENTRE_RANGE[1],
                   centre_y - crop_height * FACE_CENTRE_RANGE[0], height - crop_height)
        if x is not None and y is not None:
            box_width, box_height = int(round(crop_width)), int(round(crop_height))
            return CropBox(min(int(round(x)), width - box_width), min(int(round(y)), height - box_height),
                           box_width, box_height), None

    return None, _why_not(width, height, head, centre_x, centre_y)


def _place(ideal: float, lowest: float, highest: float, last: float) -> Optional[float]:
    # The start closest to ideal that the rules allow (lowest..highest) and keeps the box inside (0..last)
    low, high = max(lowest, 0.0), min(highest, last)
    if low > high:
        return None
    return min(max(ideal, low), high)


def _why_not(width: int, height: int, head: float, centre_x: float, centre_y: float) -> str:
    # Explain with the smallest allowed crop, the one most likely to fit
    crop_height = head / HEAD_RANGE[1]
    crop_width = crop_height * ASPECT
    if crop_height > height or crop_width > width:
        return 'The head fills too much of the photo to crop it to 35x45 without adding background.'
    if _place(0, centre_x - crop_width * (0.5 + MAX_OFFSET), centre_x - crop_width * (0.5 - MAX_OFFSET),
              width - crop_width) is None:
        side = 'left' if centre_x < width / 2 else 'right'
    else:
        side = 'top' if centre_y < height / 2 else 'bottom'
    return f'The face is too close to the {side} edge to crop the photo to 35x45 without adding background.'
