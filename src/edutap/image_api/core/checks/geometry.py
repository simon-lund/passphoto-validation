"""Geometric validation checks for passport photos."""
import numpy as np
from typing import Optional, Tuple
from ..models import ValidationIssue, ValidationContext


def check_face_position(context: ValidationContext):
    """Check if face is centered and proper size.

    :param context: The validation context containing the image and results.
    """
    bbox = context.face_data.get('bbox')
    if not bbox:
        return

    # Center check
    face_center_x = bbox['x'] + bbox['width'] / 2
    face_center_y = bbox['y'] + bbox['height'] / 2
    image_center_x = context.width / 2

    # Horizontal centering
    horizontal_offset = abs(face_center_x - image_center_x) / context.width
    context.measurements['horizontal_offset'] = float(horizontal_offset)

    if horizontal_offset > 0.1:
        context.issues.append(ValidationIssue(
            category='warning',
            message='Face is not horizontally centered.',
            severity=6
        ))

    # Face size check (should be 70-80% of image height per ICAO)
    face_height_ratio = bbox['height'] / context.height
    context.measurements['face_height_ratio'] = float(face_height_ratio)

    if face_height_ratio < 0.60:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'Face is too small ({face_height_ratio * 100:.1f}% of image). Should be 70-80%.',
            severity=9
        ))
    elif face_height_ratio > 0.85:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'Face is too large ({face_height_ratio * 100:.1f}% of image). Should be 70-80%.',
            severity=9
        ))
    elif face_height_ratio < 0.70 or face_height_ratio > 0.80:
        context.issues.append(ValidationIssue(
            category='warning',
            message=f'Face size ({face_height_ratio * 100:.1f}%) is outside optimal range (70-80%).',
            severity=5
        ))

    # Vertical position
    if face_center_y > context.height * 0.6:
        context.issues.append(ValidationIssue(
            category='warning',
            message='Face is positioned too low in the frame.',
            severity=5
        ))


def check_head_pose(context: ValidationContext):
    """Check if head is straight using eye landmarks.

    Detects roll (tilt) using eye positions - the most critical check for passport photos.

    :param context: The validation context containing the image and results.
    """
    landmarks = context.face_data.get('landmarks')
    if not landmarks:
        return

    # Get eye corner landmarks (most reliable for roll detection)
    left_eye_outer = landmarks.landmark[33]
    right_eye_outer = landmarks.landmark[263]

    # Calculate roll angle from eye positions
    left_eye_y = left_eye_outer.y * context.height
    right_eye_y = right_eye_outer.y * context.height
    left_eye_x = left_eye_outer.x * context.width
    right_eye_x = right_eye_outer.x * context.width

    dy = right_eye_y - left_eye_y
    dx = right_eye_x - left_eye_x

    roll_rad = np.arctan2(dy, dx)
    roll_deg = np.degrees(roll_rad)

    context.measurements['head_roll_degrees'] = float(roll_deg)

    pitch_deg, yaw_deg = _estimate_pitch_yaw(landmarks)
    if pitch_deg is not None:
        context.measurements['head_pitch_degrees'] = float(pitch_deg)
    if yaw_deg is not None:
        context.measurements['head_yaw_degrees'] = float(yaw_deg)

    # Check roll (head tilt left/right)
    abs_roll = abs(roll_deg)
    if abs_roll > 15:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'Head is tilted {abs_roll:.1f}° {"right" if roll_deg > 0 else "left"}. '
            f'Keep head completely straight (max 5° allowed).',
            severity=10
        ))
    elif abs_roll > 5:
        context.issues.append(ValidationIssue(
            category='warning',
            message=f'Head has slight tilt of {abs_roll:.1f}° {"right" if roll_deg > 0 else "left"}. '
            f'Try to keep head perfectly straight.',
            severity=7
        ))

    # Check pitch (head tilt up/down)
    if pitch_deg is not None:
        abs_pitch = abs(pitch_deg)
        if abs_pitch > 15:
            context.issues.append(ValidationIssue(
                category='error',
                message=f'Head is tilted {abs_pitch:.1f}° {"up" if pitch_deg < 0 else "down"}. '
                f'Look straight ahead (max 10° allowed).',
                severity=9
            ))
        elif abs_pitch > 10:
            context.issues.append(ValidationIssue(
                category='warning',
                message=f'Head is slightly tilted {abs_pitch:.1f}° {"up" if pitch_deg < 0 else "down"}.',
                severity=6
            ))

    # Check yaw (head rotation left/right)
    if yaw_deg is not None:
        abs_yaw = abs(yaw_deg)
        if abs_yaw > 15:
            context.issues.append(ValidationIssue(
                category='error',
                message=f'Head is rotated {abs_yaw:.1f}° {"right" if yaw_deg > 0 else "left"}. '
                f'Face camera directly (max 10° allowed).',
                severity=9
            ))
        elif abs_yaw > 10:
            context.issues.append(ValidationIssue(
                category='warning',
                message=f'Head is slightly rotated {abs_yaw:.1f}° {"right" if yaw_deg > 0 else "left"}.',
            ))


def _estimate_pitch_yaw(landmarks) -> Tuple[Optional[float], Optional[float]]:
    """Estimate pitch and yaw using facial landmark proportions.

    :param landmarks: The facial landmarks from MediaPipe.
    :return: A tuple of (pitch, yaw) in degrees, or (None, None) if estimation fails.
    """
    try:
        nose_tip = landmarks.landmark[1]
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[263]
        left_mouth = landmarks.landmark[61]
        right_mouth = landmarks.landmark[291]

        face_center_x = (left_eye.x + right_eye.x) / 2
        face_center_y = (left_eye.y + right_eye.y) / 2

        nose_offset_x = nose_tip.x - face_center_x
        eye_distance = abs(right_eye.x - left_eye.x)
        yaw_deg = (nose_offset_x / eye_distance) * 45 if eye_distance > 0 else 0

        mouth_center_y = (left_mouth.y + right_mouth.y) / 2
        nose_to_mouth_dist = abs(mouth_center_y - nose_tip.y)
        eyes_to_nose_dist = abs(nose_tip.y - face_center_y)

        if eyes_to_nose_dist > 0:
            vertical_ratio = nose_to_mouth_dist / eyes_to_nose_dist
            pitch_deg = (1.1 - vertical_ratio) * 50
        else:
            pitch_deg = 0

        return pitch_deg, yaw_deg
    except Exception:
        return None, None

