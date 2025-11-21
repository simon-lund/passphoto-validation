"""Lighting and shadow validation checks."""
import cv2
import numpy as np
from ..models import ValidationIssue, ValidationContext


def check_lighting(context: ValidationContext):
    """Check overall lighting quality.

    :param context: The validation context containing the image and results.
    """
    gray = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2GRAY)

    # Overall brightness
    mean_brightness = np.mean(gray)
    context.measurements['overall_brightness'] = float(mean_brightness)

    if mean_brightness < 100:
        context.issues.append(ValidationIssue(
            category='error',
            message='Image is too dark. Increase lighting.',
            severity=8
        ))
    elif mean_brightness > 200:
        context.issues.append(ValidationIssue(
            category='error',
            message='Image is overexposed. Reduce lighting or flash intensity.',
            severity=8
        ))

    # Check contrast
    contrast = gray.std()
    context.measurements['contrast'] = float(contrast)

    if contrast < 30:
        context.issues.append(ValidationIssue(
            category='warning',
            message='Low contrast. Image may appear washed out.',
            severity=5
        ))


def check_shadows(context: ValidationContext):
    """Detect harsh shadows on face.

    :param context: The validation context containing the image and results.
    """
    bbox = context.face_data.get('bbox')
    if not bbox:
        return

    face_region = context.image_np[
        bbox['y']:bbox['y'] + bbox['height'],
        bbox['x']:bbox['x'] + bbox['width']
    ]

    if face_region.size == 0:
        return

    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

    # Use a kernel size relative to the face size for blurring
    kernel_size = max(gray_face.shape[0] // 10, 5)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(kernel_size, 3)

    blur = cv2.GaussianBlur(gray_face, (kernel_size, kernel_size), 0)
    variance = np.var(blur)

    context.measurements['face_lighting_variance'] = float(variance)

    if variance > 2000:
        context.issues.append(ValidationIssue(
            category='warning',
            message='Harsh shadows detected on face. Use diffused, even lighting.',
            severity=7
        ))
