"""Image quality validation checks (sharpness, resolution)."""
import warnings
from typing import Tuple, Literal

import cv2
from brisque import BRISQUE

from ..models import ValidationIssue, ValidationContext


def check_resolution(context: ValidationContext, min_res: Tuple[int, int] = (600, 600), recommended_res: Tuple[int, int] = (900, 900)):
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
            # Can't perform face check without a face
            return None
        bbox = context.face_data['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        target_image_bgr = context.image_np[y:y + h, x:x + w]
        measurement_key = 'brisque_score_face'
        area_name = 'Face region'
    elif area == 'full':
        target_image_bgr = context.image_np
        measurement_key = 'brisque_score_overall'
        area_name = 'Overall image'
    else:
        raise ValueError("Area must be either 'full' or 'face'.")

    if target_image_bgr.size == 0:
        return None  # Skip if the target area is empty

    try:
        rgb_image = cv2.cvtColor(target_image_bgr, cv2.COLOR_BGR2RGB)
        brisque_obj = BRISQUE(url=False)
        brisque_score = brisque_obj.score(rgb_image)
        context.measurements[measurement_key] = float(brisque_score)
    except Exception as e:
        warnings.warn(f"Failed to compute BRISQUE score for {area} area. Error: {e}")
        return None

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
