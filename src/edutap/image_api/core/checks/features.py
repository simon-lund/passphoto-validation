"""Facial feature validation checks."""
from ..models import ValidationIssue, ValidationContext


def check_face_features(context: ValidationContext):
    """Check if eyes, nose, mouth are clearly visible.

    :param context: The validation context containing the image and results.
    """
    landmarks = context.face_data.get('landmarks')
    if not landmarks:
        return

    left_eye = landmarks.landmark[33]
    right_eye = landmarks.landmark[263]

    # Check if eyes are at similar height
    eye_height_diff = abs(left_eye.y - right_eye.y)
    context.measurements['eye_level_diff'] = float(eye_height_diff)

    if eye_height_diff > 0.05:
        context.issues.append(ValidationIssue(
            category='warning',
            message='Head appears tilted. Keep head straight.',
            severity=6
        ))

    # Check eye visibility
    if left_eye.visibility < 0.5 or right_eye.visibility < 0.5:
        context.issues.append(ValidationIssue(
            category='error',
            message='One or both eyes are not clearly visible.',
            severity=9
        ))


def check_expression(context: ValidationContext):
    """Check for neutral expression (mouth closed).

    :param context: The validation context containing the image and results.
    """
    landmarks = context.face_data.get('landmarks')
    if not landmarks:
        return

    upper_lip = landmarks.landmark[13]
    lower_lip = landmarks.landmark[14]

    mouth_opening = abs(upper_lip.y - lower_lip.y)
    context.measurements['mouth_opening'] = float(mouth_opening)

    if mouth_opening > 0.02:
        context.issues.append(ValidationIssue(
            category='warning',
            message='Mouth appears to be open. Keep mouth closed with neutral expression.',
            severity=7
        ))


def check_hair_obscuring(context: ValidationContext):
    """Check if hair is covering eyes or face.

    :param context: The validation context containing the image and results.
    """
    landmarks = context.face_data.get('landmarks')
    if not landmarks:
        return

    forehead = landmarks.landmark[10]

    if forehead.visibility < 0.6:
        context.issues.append(ValidationIssue(
            category='warning',
            message='Forehead may be obscured by hair. Ensure face is fully visible.',
            severity=6
        ))


def check_glasses(context: ValidationContext):
    """Detect glasses and check for reflections.

    :param context: The validation context containing the image and results.
    """
    import cv2
    import numpy as np

    bbox = context.face_data.get('bbox')
    if not bbox:
        return

    eye_region = context.image_np[
        bbox['y']:bbox['y'] + bbox['height'] // 2,
        bbox['x']:bbox['x'] + bbox['width']
    ]

    if eye_region.size == 0:
        return

    gray_eyes = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

    _, bright_spots = cv2.threshold(gray_eyes, 240, 255, cv2.THRESH_BINARY)
    bright_pixel_ratio = np.sum(bright_spots > 0) / bright_spots.size

    context.measurements['eye_reflection_ratio'] = float(bright_pixel_ratio)

    if bright_pixel_ratio > 0.05:
        context.issues.append(ValidationIssue(
            category='error',
            message='Glare or reflections detected on glasses. Adjust angle or remove glasses.',
            severity=8
        ))
    elif bright_pixel_ratio > 0.02:
        context.issues.append(ValidationIssue(
            category='warning',
            message='Possible reflections on glasses detected.',
            severity=5
        ))
