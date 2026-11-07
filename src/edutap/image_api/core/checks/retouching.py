"""Retouching checks: edits that change how the person looks."""
import cv2
import numpy as np

from ..models import CheckSkipped, ValidationIssue, ValidationContext

# MediaPipe face mesh indices: the middle of each cheek, the forehead (top, and just above the brows),
# and the corners of each eye
CHEEKS = (50, 280)
FOREHEAD = (10, 151)
EYES = ((33, 133), (362, 263))


def _points(context: ValidationContext) -> np.ndarray:
    landmarks = context.face_data.get('landmarks')
    if not landmarks:
        raise CheckSkipped('No facial landmarks found.')
    return np.array([(p.x * context.width, p.y * context.height) for p in landmarks.landmark])


def _texture(gray: np.ndarray, centre, half_width: float, half_height: float):
    """Mean absolute Laplacian in a box around `centre`, or None when the box leaves the image."""
    x0, x1 = int(round(centre[0] - half_width)), int(round(centre[0] + half_width))
    y0, y1 = int(round(centre[1] - half_height)), int(round(centre[1] + half_height))
    if x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0] or x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return float(np.abs(cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_32F)).mean())


def check_skin_smoothing(context: ValidationContext, warn_below: float, face_width: int = 192, patch: float = 0.14):
    """Check that the skin keeps its texture: beauty filters and retouching flatten the cheeks and forehead
    while the eyes stay sharp.

    `skin_texture_ratio` is the fine detail (mean absolute Laplacian) of the cheeks and forehead over that
    of the eyes, on the face scaled to `face_width` pixels. Scaling the face down hides an optical blur of a
    pixel or two, which the sharpness check reports, while a filter's smoothing spans more. Natural skin
    scores about 0.25-0.6; filtered skin below 0.15. Warns only: smooth skin can be real.

    :param warn_below: Ratio below which it warns.
    """
    bbox = context.face_data.get('bbox')
    if not bbox or bbox['width'] <= 0:
        raise CheckSkipped('No face found.')
    points = _points(context)

    scale = face_width / bbox['width']
    gray = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    points = points * scale
    half = patch * face_width / 2

    skin = [_texture(gray, points[index], half, half) for index in CHEEKS]
    skin.append(_texture(gray, (points[FOREHEAD[0]] + points[FOREHEAD[1]]) / 2, half, half))
    skin = [value for value in skin if value is not None]
    eyes = []
    for outer, inner in EYES:
        width = float(np.linalg.norm(points[outer] - points[inner]))
        eyes.append(_texture(gray, (points[outer] + points[inner]) / 2, width * 0.7, width * 0.4))
    eyes = [value for value in eyes if value is not None]
    if len(skin) < 2 or not eyes:
        raise CheckSkipped('The cheeks or eyes are outside the photo.')
    eye_texture = float(np.mean(eyes))
    if eye_texture < 1e-3:
        raise CheckSkipped('The eye region is flat.')

    ratio = float(np.median(skin) / eye_texture)
    context.measurements['skin_texture'] = float(np.median(skin))
    context.measurements['eye_texture'] = eye_texture
    context.measurements['skin_texture_ratio'] = ratio
    if ratio < warn_below:
        context.issues.append(ValidationIssue(
            category='warning', severity=4,
            message=f'The skin looks smoothed, as by a beauty filter (texture ratio {ratio:.2f}). The photo must not be retouched.'))
