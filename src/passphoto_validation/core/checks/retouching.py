"""Retouching checks: edits that change how the person looks."""
import math

import cv2
import mediapipe as mp
import numpy as np

from ..check import Check
from ..models import CheckSkipped, ValidationContext
from ..utils import mesh_points

# MediaPipe face mesh indices: the middle of each cheek, the forehead (top, and just above the brows),
# and the corners of each eye
CHEEKS = (50, 280)
FOREHEAD = (10, 151)
EYES = ((33, 133), (362, 263))
CHIN = 199  # the middle of the chin, below the lower lip
_MESH = mp.solutions.face_mesh
# The features left out when the skin's colour is judged: eyes and brows (make-up), lips (lipstick)
_FEATURES = (_MESH.FACEMESH_LEFT_EYE, _MESH.FACEMESH_RIGHT_EYE, _MESH.FACEMESH_LEFT_EYEBROW,
             _MESH.FACEMESH_RIGHT_EYEBROW, _MESH.FACEMESH_LIPS)


def _texture(gray: np.ndarray, centre, half_width: float, half_height: float):
    """Mean absolute Laplacian in a box around `centre`, or None when the box leaves the image."""
    x0, x1 = int(round(centre[0] - half_width)), int(round(centre[0] + half_width))
    y0, y1 = int(round(centre[1] - half_height)), int(round(centre[1] + half_height))
    if x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0] or x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return float(np.abs(cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_32F)).mean())


class SkinSmoothing(Check):
    """Check that the skin keeps its texture: beauty filters and retouching flatten the cheeks and forehead
    while the eyes stay sharp.

    `skin_texture_ratio` is the fine detail (mean absolute Laplacian) of the cheeks and forehead over that
    of the eyes, on the face scaled to `face_width` pixels. Scaling the face down hides an optical blur of a
    pixel or two, which the sharpness check reports, while a filter's smoothing spans more. Natural skin
    scores about 0.25-0.6; filtered skin below 0.15. Warns only by default: smooth skin can be real.

    A filter over the whole face, eyes included, keeps that ratio normal, so `skin_gradient_ratio` compares
    the skin's fine detail with the typical gradient of the whole face (its median magnitude): an
    edge-preserving filter keeps the lids, nostrils and hairline while it flattens the skin, so the ratio
    drops; a darker, lower-contrast or colour-cast photo scales both and keeps it. An optical blur lowers it
    too, so this second warning is not given when the sharpness check has already failed the photo.

    :param warn_below: Texture ratio below which it warns.
    :param warn_gradient_ratio_below: Gradient ratio below which it warns; None turns that off.
    :param fail_below: Texture ratio below which it fails; off by default.
    :param fail_gradient_ratio_below: Gradient ratio below which it fails; off by default.
    """

    name = 'skin_smoothing'
    reads = ('face_sharpness',)

    warn_below: float
    warn_gradient_ratio_below: float = None
    fail_below: float = -math.inf
    fail_gradient_ratio_below: float = -math.inf
    face_width: int = 192
    patch: float = 0.14

    def measure(self, context: ValidationContext):
        bbox = context.face_data.get('bbox')
        if not bbox or bbox['width'] <= 0:
            raise CheckSkipped('No face found.')
        points = mesh_points(context)

        scale = self.face_width / bbox['width']
        gray = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        points = points * scale
        half = self.patch * self.face_width / 2

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

        skin_texture = float(np.median(skin))
        ratio = skin_texture / eye_texture
        gradient = _face_gradient(gray, _hull(gray.shape, points, _MESH.FACEMESH_FACE_OVAL))
        gradient_ratio = skin_texture / gradient if gradient > 1e-3 else None
        context.measurements['skin_texture'] = skin_texture
        context.measurements['eye_texture'] = eye_texture
        context.measurements['skin_texture_ratio'] = ratio
        if gradient_ratio is not None:  # a flat face has no gradient to compare with
            context.measurements['skin_gradient_ratio'] = gradient_ratio
        finding = (f'The skin looks smoothed, as by a beauty filter (texture ratio {ratio:.2f}). The photo must not be retouched.', 4)
        if self.judge(context, ratio, fail=finding, warn=finding) != 'pass':
            return
        sharpness = context.results.get('face_sharpness')
        blurred = sharpness is not None and sharpness.status == 'fail'
        if gradient_ratio is not None and not blurred:
            finding = (f'The skin has lost its fine texture while the edges of the face stayed sharp, as a filter over the '
                       f'whole face leaves it (gradient ratio {gradient_ratio:.3f}). The photo must not be retouched.', 4)
            self.judge(context, gradient_ratio, warn_below=self.warn_gradient_ratio_below, fail_below=self.fail_gradient_ratio_below,
                       fail=finding, warn=finding)


def _hull(shape, points: np.ndarray, connections, grow: int = 0) -> np.ndarray:
    """Convex hull of the mesh points the `connections` join, as a 0/1 mask, grown by `grow` pixels."""
    indices = sorted({index for edge in connections for index in edge})
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points[indices].astype(np.int32)), 1)
    return cv2.dilate(mask, np.ones((grow, grow), np.uint8)) if grow else mask


def _face_gradient(gray: np.ndarray, oval: np.ndarray) -> float:
    """The typical gradient of the face: the median gradient magnitude inside the face oval.

    An edge-preserving filter keeps most of it (lids, nostrils, the hairline, the shading of the face) while
    it flattens the skin; darkening or lowering the contrast scales it along with the skin texture.
    """
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(dx * dx + dy * dy)[oval > 0]
    return float(np.median(magnitude)) if magnitude.size else 0.0


class SkinColourPatches(Check):
    """Check the skin for patches of a colour that isn't skin: the stickers face filters paint on (a pink
    nose, hearts, butterflies), face paint, coloured tattoos.

    The skin's colour is the CIELAB chroma (a*, b*) of the cheeks and chin: the median of each patch, then the
    median of the three, so one sticker on a cheek can't shift it. A pixel of the lower face is foreign when
    its chroma is more than `chroma_distance` away from that and at least as saturated as the skin: dark
    hair, beards and shadows are less saturated, so they don't count, nor do white or black marks (sparkles,
    drawn whiskers), which this check cannot see. The lower face is the face oval below the eyes, inside the
    person mask and away from the oval's edge, without the lips: the forehead and the band across the eyes
    are left out because hats, hair and glasses frames sit there and would read as paint. Patches smaller
    than `min_patch_share` of the face oval are grain and moles. `skin_foreign_share` is the share of the
    oval the remaining patches cover, `skin_foreign_patches` their number. Face paint, a clown's nose and
    strongly coloured light on one side of the face read the same as stickers.

    :param warn_share_above: Share of the face oval covered by foreign patches above which it warns.
    :param fail_share_above: Share above which it fails.
    """

    name = 'skin_colour_patches'

    warn_share_above: float
    fail_share_above: float = math.inf
    chroma_distance: float = 25.0
    min_patch_share: float = 0.002
    face_width: int = 256

    def measure(self, context: ValidationContext):
        bbox = context.face_data.get('bbox')
        if not bbox or bbox['width'] <= 0:
            raise CheckSkipped('No face found.')
        points = mesh_points(context)

        scale = self.face_width / bbox['width']
        image = cv2.resize(context.image_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        points = points * scale
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma = lab[..., 1:] - 128  # a*, b* around 0

        oval = _hull(image.shape, points, _MESH.FACEMESH_FACE_OVAL)
        oval_area = int(oval.sum())
        if oval_area < 100:
            raise CheckSkipped('The face is too small to judge its colour.')
        margin = max(1, int(round(self.face_width * 0.04)))
        skin = cv2.erode(oval, np.ones((margin, margin), np.uint8))
        if context.person_mask_binary is not None:
            person = cv2.resize(context.person_mask_binary, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            skin &= person
        eyes = np.concatenate([points[sorted({i for e in _MESH.FACEMESH_LEFT_EYE for i in e})],
                               points[sorted({i for e in _MESH.FACEMESH_RIGHT_EYE for i in e})]])
        skin[:int(min(image.shape[0], eyes[:, 1].max() + self.face_width * 0.12)), :] = 0
        skin &= 1 - _hull(image.shape, points, _MESH.FACEMESH_LIPS, max(1, int(round(self.face_width * 0.05))))
        if skin.sum() < 0.15 * oval_area:
            raise CheckSkipped('Too little skin is visible to judge its colour.')

        half = 0.14 * self.face_width / 2
        medians = []
        for centre in (points[CHEEKS[0]], points[CHEEKS[1]], points[CHIN]):
            x0, x1 = int(round(centre[0] - half)), int(round(centre[0] + half))
            y0, y1 = int(round(centre[1] - half)), int(round(centre[1] + half))
            if 0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]:
                medians.append(np.median(chroma[y0:y1, x0:x1].reshape(-1, 2), axis=0))
        if len(medians) < 2:
            raise CheckSkipped('The cheeks are outside the photo.')
        reference = np.median(np.array(medians), axis=0)
        reference_saturation = float(np.linalg.norm(reference))

        distance = np.linalg.norm(chroma - reference, axis=2)
        saturation = np.linalg.norm(chroma, axis=2)
        foreign = ((distance > self.chroma_distance) & (saturation >= reference_saturation) & (skin > 0)).astype(np.uint8)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreign, connectivity=8)
        patches = [int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)
                   if stats[index, cv2.CC_STAT_AREA] >= self.min_patch_share * oval_area]
        share = float(sum(patches) / oval_area)
        context.measurements['skin_foreign_share'] = share
        context.measurements['skin_foreign_patches'] = len(patches)
        message = (f'Patches of a colour that is not skin cover {share:.0%} of the face, as the stickers of a face '
                   f'filter or face paint do. The photo must not be retouched or filtered.')
        self.judge(context, share, warn_above=self.warn_share_above, fail_above=self.fail_share_above, fail=(message, 7), warn=(message, 4))
