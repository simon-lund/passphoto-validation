"""Image quality validation checks (sharpness, resolution, compression)."""
from typing import Tuple, Literal

import cv2
import numpy as np
from brisque import BRISQUE

from ..models import CheckSkipped, ValidationIssue, ValidationContext


_BRISQUE = None


def _brisque() -> BRISQUE:
    """Load the BRISQUE model once; building it reads the model files from disk."""
    global _BRISQUE
    if _BRISQUE is None:
        _BRISQUE = BRISQUE(url=False)
    return _BRISQUE


def check_resolution(context: ValidationContext, min_res: Tuple[int, int] = (413, 531), recommended_res: Tuple[int, int] = (827, 1063)):
    """Check if image has sufficient resolution.

    :param context: The validation context containing the image and results.
    :param min_res: The minimum allowed resolution as a (width, height) tuple.
    :param recommended_res: The recommended minimum resolution as a (width, height) tuple.
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


def check_distortions(context: ValidationContext,
                        area: Literal["full", "face"] = "full",
                        error_threshold: float = 30.0,
                        warning_threshold: float = 10.0,
                        debug: bool = False):
    """
    Assess image quality (blur, noise, artifacts) for a specific area using the BRISQUE model.

    :param context: The validation context.
    :param area: The area to check: 'full' for the entire image, 'face' for the detected face region.
    :param error_threshold: BRISQUE score above this is an error.
    :param warning_threshold: BRISQUE score above this is a warning.
    :param debug: If True, returns visualization data.
    """
    if area == 'face':
        if not context.face_data or 'bbox' not in context.face_data:
            raise CheckSkipped('No face found.')
        bbox = context.face_data['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        target_image_bgr = context.image_np[max(0, y):max(0, y + h), max(0, x):max(0, x + w)]
        measurement_key = 'brisque_score_face'
        area_name = 'Face region'
    elif area == 'full':
        target_image_bgr = context.image_np
        measurement_key = 'brisque_score_overall'
        area_name = 'Overall image'
    else:
        raise ValueError("Area must be either 'full' or 'face'.")

    if target_image_bgr.size == 0:
        raise CheckSkipped(f'{area_name} area is empty.')

    rgb_image = cv2.cvtColor(target_image_bgr, cv2.COLOR_BGR2RGB)
    brisque_score = _brisque().score(rgb_image)
    context.measurements[measurement_key] = float(brisque_score)

    if brisque_score > error_threshold:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'{area_name} area quality is poor (artifacts/blur detected). (BRISQUE Score: {brisque_score:.2f})',
            severity=8 if area == 'face' else 5
        ))
    elif brisque_score > warning_threshold:
        context.issues.append(ValidationIssue(
            category='warning',
            message=f'{area_name} area may have artifacts or be slightly blurry. (BRISQUE Score: {brisque_score:.2f})',
            severity=5 if area == 'face' else 2
        ))

    if debug:
        analyzed_image_for_debug = cv2.cvtColor(target_image_bgr, cv2.COLOR_BGR2GRAY)
        return {
            'brisque_score': brisque_score,
            'analyzed_image': analyzed_image_for_debug
        }
    return None


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
    if sharpness < fail_below:
        context.issues.append(ValidationIssue(
            category='error', message='Face is blurry. The photo must be sharp.', severity=8))
    elif sharpness < warn_below:
        context.issues.append(ValidationIssue(
            category='warning', message='Face may be slightly blurry.', severity=5))


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
    if blockiness > fail_above:
        context.issues.append(ValidationIssue(
            category='error', message='Strong compression artefacts (blocky image). Save with higher quality.', severity=8))
    elif blockiness > warn_above:
        context.issues.append(ValidationIssue(
            category='warning', message='Visible compression artefacts.', severity=5))
