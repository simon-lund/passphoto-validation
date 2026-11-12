"""Image quality validation checks (sharpness, resolution, compression)."""
from typing import Tuple

import cv2
import numpy as np

from ..levels import judge
from ..models import CheckSkipped, ValidationIssue, ValidationContext


def check_resolution(context: ValidationContext, min_res: Tuple[int, int] = (413, 531), recommended_res: Tuple[int, int] = (827, 1063)):
    """Check if image has sufficient resolution.

    :param context: The validation context containing the image and results.
    :param min_res: The (width, height) below which it fails; (0, 0) turns that off.
    :param recommended_res: The (width, height) below which it warns; (0, 0) turns that off.
    """
    width, height = context.width, context.height
    min_width, min_height = min_res
    recommended_width, recommended_height = recommended_res

    context.measurements['resolution'] = f"{width}x{height}"

    if width < min_width or height < min_height:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'Resolution too low ({width}x{height}). Minimum {min_width}x{min_height} required.',
            severity=9
        ))
    elif width < recommended_width or height < recommended_height:
        context.issues.append(ValidationIssue(
            category='warning',
            message=f'Resolution is acceptable but could be higher for best quality (recommended: >{recommended_width}x{recommended_height}px).',
            severity=4
        ))


def check_face_sharpness(context: ValidationContext, warn_below: float, fail_below: float, face_width: int = 256):
    """Check that the face is sharp: Laplacian energy relative to gradient energy, on the face scaled to a fixed width.

    Dividing by the gradient energy makes the measure independent of exposure and contrast, and scaling the
    face makes it independent of the photo's resolution. Sharp studio faces score about 0.15-0.24, a blur
    of sigma 2 px at 1063 px height about 0.08-0.11.

    :param warn_below: Ratio below which it warns.
    :param fail_below: Ratio below which it fails.
    """
    bbox = context.face_data.get('bbox')
    if not bbox:
        raise CheckSkipped('No face found.')
    face = context.image_np[max(0, bbox['y']):bbox['y'] + bbox['height'], max(0, bbox['x']):bbox['x'] + bbox['width']]
    if face.size == 0:
        raise CheckSkipped('Face region is outside the photo.')

    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    height = max(1, int(round(face_width * gray.shape[0] / gray.shape[1])))
    gray = cv2.resize(gray, (face_width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F)).mean()
    gradient = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)).mean()
    if gradient < 1e-3:
        raise CheckSkipped('The face region is flat.')

    sharpness = float(laplacian / gradient)
    context.measurements['face_sharpness'] = sharpness
    judge(context, sharpness, warn_below=warn_below, fail_below=fail_below,
          fail=('Face is blurry. The photo must be sharp.', 8), warn=('Face may be slightly blurry.', 5))


def check_compression_blocks(context: ValidationContext, warn_above: float, fail_above: float):
    """Check for JPEG block artefacts: brightness jumps at 8-pixel block edges compared with inside the blocks.

    About 1.0-1.15 for a well-compressed photo, above 5 for JPEG quality 4. Misses artefacts once the
    photo was resized after compression, since the block grid no longer lines up.

    :param warn_above: Blockiness above which it warns.
    :param fail_above: Blockiness above which it fails.
    """
    gray = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ratios = []
    for axis in (0, 1):
        jumps = np.abs(np.diff(gray, axis=axis))
        at_edge = (np.arange(jumps.shape[axis]) % 8) == 7
        inside = jumps.take(np.nonzero(~at_edge)[0], axis=axis).mean()
        ratios.append(jumps.take(np.nonzero(at_edge)[0], axis=axis).mean() / max(inside, 1e-3))

    blockiness = float(np.mean(ratios))
    context.measurements['jpeg_blockiness'] = blockiness
    judge(context, blockiness, warn_above=warn_above, fail_above=fail_above,
          fail=('Strong compression artefacts (blocky image). Save with higher quality.', 8),
          warn=('Visible compression artefacts.', 5))


def check_print_defects(context: ValidationContext, fail_at: int, warn_at: int = 1, min_area_share: float = 0.0002,
                        max_area_share: float = 0.01, contrast: float = 15.0):
    """Check the background for marks a print or scan leaves: ink spots, dust, scratches.

    A mark is a patch of background that differs from its surroundings (the median colour over a window of
    twice the band, see background._background_lightness) by more than `contrast` in CIELAB, and covers
    between `min_area_share` and `max_area_share` of the photo. Below the minimum it is grain; above the
    maximum it is an object, which check_background_uniformity and OFIQ's background check report. A patch
    that reaches the band around the person is taken for part of the person (stray hair, a collar), since
    the segmentation misses those; so marks right next to the person aren't found, nor are marks on the face
    or clothes, which can't be told from features.

    Clean studio crops (London Set) show 0 marks, 4% of them 1 or 2 (stray hair the band doesn't cover);
    three ink spots of about 1 mm and a scratch show as 3-5 on 97% of them.

    :param fail_at: Number of marks from which it fails.
    :param warn_at: Number of marks from which it warns.
    """
    from .background import _background_lightness

    _, distance, band = _background_lightness(context)
    background = distance > 1
    if background.mean() < 0.02:
        raise CheckSkipped('Too little background to look for marks.')

    lab = cv2.cvtColor(cv2.GaussianBlur(context.image_np, (0, 0), 1), cv2.COLOR_BGR2LAB)
    window = min(2 * band + 1, 255) | 1
    level = cv2.medianBlur(lab, window)
    difference = lab.astype(np.float32) - level.astype(np.float32)
    difference[..., 0] *= 100 / 255  # OpenCV scales L* to 0-255; a* and b* are only offset
    deviation = np.linalg.norm(difference, axis=2)
    candidates = ((deviation > contrast) & background).astype(np.uint8)

    area = context.width * context.height
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    marks = []
    for index in range(1, count):
        x, y, w, h, size = stats[index]
        if not min_area_share * area <= size <= max_area_share * area:
            continue
        patch = labels[y:y + h, x:x + w] == index
        if distance[y:y + h, x:x + w][patch].min() <= 1.1:
            continue
        marks.append(int(size))
    context.measurements['background_marks'] = len(marks)
    context.measurements['background_marks_area'] = float(sum(marks) / area)

    judge(context, len(marks), warn_at=warn_at, fail_at=fail_at,
          fail=(f'{len(marks)} marks on the background (spots, stains or scratches). The print or scan must be clean.', 6),
          warn=('A mark on the background (a spot, stain or scratch).', 3))
