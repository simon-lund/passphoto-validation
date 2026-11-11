"""Build a labelled evaluation set of ID photos from the Face Research Lab London Set.

The London Set (DeBruine & Jones 2017, CC BY 4.0, https://doi.org/10.6084/m9.figshare.5047666) has
102 people photographed frontally on a plain grey background, neutral and smiling, plus 3/4 views.
The subjects consented to use "in lab-based and web-based studies"; keep the images and anything
derived from them out of the repository and don't redistribute them.

Every output photo has a known label: the compliant crop of the neutral photo, the real smiling and
3/4 photos (cropped the same way), and synthetic defects made from the compliant crop (each breaks
exactly one rule). The manifest lists which rule each variant breaks.

    uv run python scripts/build_eval_set.py --source data/london --out data/london-eval
"""
import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

HEAD_SHARE = 0.75  # head height (chin to crown) as a share of the photo height: German 32-36 mm of 45 mm
CROWN_MARGIN = 0.09  # space above the crown as a share of the photo height
ASPECT = 35 / 45
OUT_HEIGHT = 1063  # 45 mm at 600 dpi

# (variant, source folder, rule the variant breaks; None = should pass every rule)
REAL_VARIANTS = [
    ('compliant', 'neutral_front', None),
    ('smiling', 'smiling_front', 'neutral_expression'),
    ('yaw_left', 'neutral_left_3quarter', 'no_head_rotation'),
    ('yaw_right', 'neutral_right_3quarter', 'no_head_rotation'),
]
SYNTHETIC_VARIANTS = {
    'roll_4': None,  # within tolerance: must still pass
    'roll_12': 'no_head_tilt',
    'dark': 'not_underexposed',
    'bright': 'lighting_not_overexposed',
    'low_contrast': 'contrast',
    'blur': 'sharp',
    'blur_2': 'sharp',
    'jpeg': 'no_pixelation',
    'small_head': 'face_size',
    'off_center': 'face_centered',
    'red_eye': 'no_red_eyes',
    'sunglasses': 'glasses_not_tinted',
    'clear_glasses': None,  # glasses are allowed: must pass
    'glasses_glare': 'glasses_no_reflections',
    'bg_pattern': 'background_plain',
    'bg_object': 'background_no_objects',
    'colour_cast': 'natural_colour',
    'side_shadow': 'no_face_shadows',
    'bg_shadow': 'background_no_shadows',
    'bg_gradient': 'no_color_gradients',
    'ink_stain': 'no_print_errors',
    'smoothed': 'no_retouching',
    'face_stickers': 'no_retouching',
    'smoothed_all': 'no_retouching',
}

_CHIN, _FOREHEAD, _NOSE = 152, 10, 1
_RIGHT_EYE, _LEFT_EYE = (33, 133), (263, 362)
_IRISES = ((468, (469, 470, 471, 472)), (473, (474, 475, 476, 477)))


class Geometry:
    """Face mesh and person mask for one photo."""

    def __init__(self, mesh, segmenter, image: np.ndarray):
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = mesh.process(rgb)
        if not result.multi_face_landmarks:
            raise ValueError('no face')
        self.points = np.array([(p.x * w, p.y * h) for p in result.multi_face_landmarks[0].landmark])
        self.person = segmenter.process(rgb).segmentation_mask > 0.5

    def crown_and_chin(self):
        """Top of the head (person mask, hair capped at 0.45 face heights above the forehead) and chin, in y."""
        chin, forehead = self.points[_CHIN][1], self.points[_FOREHEAD][1]
        face = chin - forehead
        x = int(self.points[_NOSE][0])
        column = self.person[:int(forehead), max(0, x - 15):x + 16].any(axis=1)
        top = int(np.argmax(column)) if column.any() else forehead - 0.32 * face
        return max(top, forehead - 0.45 * face), chin

    def eye_centre(self):
        return (self.points[_RIGHT_EYE[0]] + self.points[_LEFT_EYE[0]]) / 2


def crop(image: np.ndarray, geometry: Geometry, head_share=HEAD_SHARE, shift_x=0.0, upscale=True) -> np.ndarray:
    """Crop to 35x45 with the head at `head_share` of the height, padding with the edge colour if needed.

    The crop is resized to OUT_HEIGHT; with upscale=False a smaller crop keeps its own size.
    """
    crown, chin = geometry.crown_and_chin()
    height = (chin - crown) / head_share
    width = height * ASPECT
    top = crown - CROWN_MARGIN * height
    # Centre on the whole face, so turned heads stay in the frame
    face_x = (geometry.points[:, 0].min() + geometry.points[:, 0].max()) / 2
    left = face_x - width / 2 + shift_x * width
    pad = int(max(height, width))
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    y0, x0 = int(round(top)) + pad, int(round(left)) + pad
    out = padded[y0:y0 + int(round(height)), x0:x0 + int(round(width))]
    if not upscale and out.shape[0] <= OUT_HEIGHT:
        return out
    return cv2.resize(out, (int(round(OUT_HEIGHT * ASPECT)), OUT_HEIGHT), interpolation=cv2.INTER_AREA)


def rotate(image: np.ndarray, centre, degrees: float) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), degrees, 1.0)
    return cv2.warpAffine(image, matrix, image.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)


def _iris(geometry: Geometry, index: int):
    centre_i, ring = _IRISES[index]
    centre = geometry.points[centre_i]
    return centre, float(np.mean([np.linalg.norm(geometry.points[i] - centre) for i in ring]))


def red_eye(image: np.ndarray, geometry: Geometry) -> np.ndarray:
    out = image.copy()
    for i in range(2):
        centre, radius = _iris(geometry, i)
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(mask, tuple(int(v) for v in centre), max(2, int(radius * 0.55)), 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), radius * 0.1 + 0.5)[..., None] / 255.0
        red = np.zeros_like(image)
        red[...] = (30, 30, 200)
        out = (out * (1 - mask) + red * mask).astype(np.uint8)
    return out


def glasses(image: np.ndarray, geometry: Geometry, tint: float = 0.0, glare: bool = False) -> np.ndarray:
    """Draw glasses: a dark frame, lenses darkened by `tint` (0 = clear), optionally with reflections."""
    out = image.copy()
    eye_distance = float(np.linalg.norm(geometry.points[_RIGHT_EYE[0]] - geometry.points[_LEFT_EYE[0]]))
    lens = np.zeros(image.shape[:2], np.uint8)
    centres = []
    for outer, inner in (_RIGHT_EYE, _LEFT_EYE):
        centre = (geometry.points[outer] + geometry.points[inner]) / 2
        centres.append(centre)
        cv2.ellipse(lens, tuple(int(v) for v in centre), (int(eye_distance * 0.3), int(eye_distance * 0.22)), 0, 0, 360, 255, -1)
    alpha = (lens[..., None] / 255.0) * tint
    out = (out * (1 - alpha) + np.array((25, 20, 20)) * alpha).astype(np.uint8)
    if glare:
        for centre in centres:
            spot = np.zeros(image.shape[:2], np.uint8)
            cv2.ellipse(spot, (int(centre[0] - eye_distance * 0.1), int(centre[1] - eye_distance * 0.08)),
                        (int(eye_distance * 0.12), int(eye_distance * 0.05)), -25, 0, 360, 255, -1)
            soft = cv2.GaussianBlur(spot, (0, 0), eye_distance * 0.01 + 0.5)[..., None] / 255.0
            out = (out * (1 - soft) + 252 * soft).astype(np.uint8)
    frame_width = max(2, int(eye_distance * 0.035))
    for centre in centres:
        cv2.ellipse(out, tuple(int(v) for v in centre), (int(eye_distance * 0.3), int(eye_distance * 0.22)), 0, 0, 360, (15, 15, 15), frame_width)
    a, b = centres
    cv2.line(out, (int(a[0] + eye_distance * 0.3), int(a[1])), (int(b[0] - eye_distance * 0.3), int(b[1])), (15, 15, 15), frame_width)
    return out


def replace_background(image: np.ndarray, geometry: Geometry, background: np.ndarray) -> np.ndarray:
    alpha = cv2.GaussianBlur(geometry.person.astype(np.float32), (0, 0), 3)[..., None]
    return (image * alpha + background * (1 - alpha)).astype(np.uint8)


def pattern(shape) -> np.ndarray:
    h, w = shape[:2]
    rows, cols = np.mgrid[:h, :w]
    stripes = ((cols + rows) // 40) % 2
    flowers = (np.sin(cols / 23.0) * np.sin(rows / 23.0) > 0.4)
    base = np.where(stripes[..., None], (150, 170, 190), (200, 210, 225)).astype(np.uint8)
    base[flowers] = (90, 120, 160)
    return base


def with_object(image: np.ndarray, geometry: Geometry) -> np.ndarray:
    h, w = image.shape[:2]
    background = image.copy()
    cv2.rectangle(background, (int(w * 0.03), int(h * 0.05)), (int(w * 0.2), int(h * 0.35)), (40, 90, 160), -1)
    cv2.circle(background, (int(w * 0.88), int(h * 0.2)), int(w * 0.07), (60, 160, 60), -1)
    return replace_background(image, geometry, background)


def colour_cast(image: np.ndarray) -> np.ndarray:
    return np.clip(image * np.array((0.6, 0.95, 1.3)), 0, 255).astype(np.uint8)


def side_shadow(image: np.ndarray, geometry: Geometry) -> np.ndarray:
    w = image.shape[1]
    nose_x = geometry.points[_NOSE][0]
    cols = np.arange(w, dtype=np.float32)
    factor = 0.4 + 0.6 / (1 + np.exp(-(cols - nose_x) / (w * 0.03)))
    shade = np.ones(image.shape[:2], np.float32) * factor[None, :]
    shade = np.where(geometry.person, shade, 1.0)
    return np.clip(image * shade[..., None], 0, 255).astype(np.uint8)


def bg_shadow(image: np.ndarray, geometry: Geometry) -> np.ndarray:
    """A soft shadow on the background to the right of the person, as a flash from the left throws."""
    h, w = image.shape[:2]
    dx, dy = int(w * 0.06), int(h * 0.02)
    shifted = np.zeros_like(geometry.person)
    shifted[dy:, dx:] = geometry.person[:h - dy, :w - dx]
    shade = cv2.GaussianBlur((shifted & ~geometry.person).astype(np.float32), (0, 0), w * 0.01)
    return np.clip(image * (1 - 0.4 * shade)[..., None], 0, 255).astype(np.uint8)


def bg_gradient(image: np.ndarray, geometry: Geometry) -> np.ndarray:
    """The background darkens from left to right, as a gradient backdrop or a lamp on one side gives."""
    w = image.shape[1]
    factor = np.where(geometry.person, 1.0, np.linspace(1.0, 0.5, w, dtype=np.float32)[None, :])
    return np.clip(image * factor[..., None], 0, 255).astype(np.uint8)


def ink_stain(image: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Three dark spots of about 1 mm and a scratch on the background, as a leaking pen or a dirty scanner leaves."""
    h, w = image.shape[:2]
    out = image.copy()
    for fx, fy, radius in ((0.1, 0.1, 0.02), (0.9, 0.12, 0.015), (0.07, 0.5, 0.012)):
        cv2.ellipse(out, (int(w * fx), int(h * fy)), (int(w * radius), int(w * radius * 0.7)), 20, 0, 360, (30, 40, 60), -1)
    cv2.line(out, (int(w * 0.75), int(h * 0.03)), (int(w * 0.92), int(h * 0.2)), (40, 40, 40), max(1, w // 300))
    out = cv2.GaussianBlur(out, (0, 0), 0.8)
    return np.where(geometry.person[..., None], image, out)


def _hull(shape, points: np.ndarray, connections, grow: int) -> np.ndarray:
    indices = sorted({index for edge in connections for index in edge})
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points[indices].astype(np.int32)), 1)
    return cv2.dilate(mask, np.ones((grow, grow), np.uint8)) if grow else mask


def smoothed(image: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Skin flattened by an edge-preserving filter, as beauty filters do; eyes, brows and mouth stay as they are."""
    w = image.shape[1]
    face_mesh = mp.solutions.face_mesh
    grow = max(1, int(w * 0.02))
    skin = _hull(image.shape, geometry.points, face_mesh.FACEMESH_FACE_OVAL, 0)
    for feature in (face_mesh.FACEMESH_LEFT_EYE, face_mesh.FACEMESH_RIGHT_EYE, face_mesh.FACEMESH_LEFT_EYEBROW,
                    face_mesh.FACEMESH_RIGHT_EYEBROW, face_mesh.FACEMESH_LIPS):
        skin &= 1 - _hull(image.shape, geometry.points, feature, grow)
    alpha = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), w * 0.01)[..., None]
    smooth = cv2.edgePreservingFilter(image, flags=cv2.RECURS_FILTER, sigma_s=40, sigma_r=0.25)
    return (image * (1 - alpha) + smooth * alpha).astype(np.uint8)


def smoothed_all(image: np.ndarray, geometry: Geometry) -> np.ndarray:
    """The whole face flattened by an edge-preserving filter, eyes and mouth included, as a whole-face beauty
    filter does; the edges stay, the skin texture goes."""
    w = image.shape[1]
    face = _hull(image.shape, geometry.points, mp.solutions.face_mesh.FACEMESH_FACE_OVAL, max(1, int(w * 0.04)))
    alpha = cv2.GaussianBlur(face.astype(np.float32), (0, 0), w * 0.01)[..., None]
    smooth = cv2.edgePreservingFilter(image, flags=cv2.RECURS_FILTER, sigma_s=40, sigma_r=0.25)
    return (image * (1 - alpha) + smooth * alpha).astype(np.uint8)


def face_stickers(image: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Coloured stickers on the face, as a face filter paints them: a pink nose and a butterfly on each cheek."""
    out = image.copy()
    face_width = float(np.linalg.norm(geometry.points[234] - geometry.points[454]))
    nose = tuple(int(v) for v in geometry.points[_NOSE])
    radius = int(face_width * 0.08)
    cv2.ellipse(out, nose, (radius, int(radius * 0.7)), 0, 0, 360, (160, 120, 240), -1, cv2.LINE_AA)
    for cheek, colour in ((50, (200, 80, 255)), (280, (255, 160, 0))):
        cx, cy = (int(v) for v in geometry.points[cheek])
        size = int(face_width * 0.07)
        for side in (-1, 1):
            cv2.ellipse(out, (cx + side * size // 2, cy), (size // 2, int(size * 0.7)), side * 25, 0, 360, colour, -1, cv2.LINE_AA)
        cv2.line(out, (cx, cy - int(size * 0.7)), (cx, cy + int(size * 0.7)), (40, 40, 40), max(2, size // 12), cv2.LINE_AA)
    return out


def synthetic(variant: str, base: np.ndarray, geometry: Geometry, source: np.ndarray, source_geometry: Geometry):
    if variant == 'roll_4':
        return rotate(base, geometry.eye_centre(), 4)
    if variant == 'roll_12':
        return rotate(base, geometry.eye_centre(), 12)
    if variant == 'dark':
        return (base * 0.35).astype(np.uint8)
    if variant == 'bright':
        return np.clip(base * 1.6 + 60, 0, 255).astype(np.uint8)
    if variant == 'low_contrast':
        return (base.mean() + (base - base.mean()) * 0.3).astype(np.uint8)
    if variant == 'blur':
        return cv2.GaussianBlur(base, (0, 0), 4)
    if variant == 'jpeg':
        _, data = cv2.imencode('.jpg', base, [cv2.IMWRITE_JPEG_QUALITY, 4])
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if variant == 'small_head':
        return crop(source, source_geometry, head_share=0.55)
    if variant == 'off_center':
        return crop(source, source_geometry, shift_x=0.18)
    if variant == 'red_eye':
        return red_eye(base, geometry)
    if variant == 'blur_2':
        return cv2.GaussianBlur(base, (0, 0), 2)
    if variant == 'sunglasses':
        return glasses(base, geometry, tint=0.85)
    if variant == 'clear_glasses':
        return glasses(base, geometry)
    if variant == 'glasses_glare':
        return glasses(base, geometry, glare=True)
    if variant == 'bg_pattern':
        return replace_background(base, geometry, pattern(base.shape))
    if variant == 'bg_object':
        return with_object(base, geometry)
    if variant == 'colour_cast':
        return colour_cast(base)
    if variant == 'side_shadow':
        return side_shadow(base, geometry)
    if variant == 'bg_shadow':
        return bg_shadow(base, geometry)
    if variant == 'bg_gradient':
        return bg_gradient(base, geometry)
    if variant == 'ink_stain':
        return ink_stain(base, geometry)
    if variant == 'smoothed':
        return smoothed(base, geometry)
    if variant == 'smoothed_all':
        return smoothed_all(base, geometry)
    if variant == 'face_stickers':
        return face_stickers(base, geometry)
    raise ValueError(variant)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True, help='Folder with the unpacked London Set zips')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def find(folder):
        return {p.stem.split('_')[0]: p for p in (args.source / folder).rglob('*.jpg')}

    sources = {folder: find(folder) for _, folder, _ in REAL_VARIANTS}
    rows = []
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as mesh, \
            mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0) as segmenter:
        for identity in sorted(sources['neutral_front']):
            split = 'calibrate' if int(identity) % 2 else 'test'
            for variant, folder, rule in REAL_VARIANTS:
                path = sources[folder].get(identity)
                if path is None:
                    continue
                image = cv2.imread(str(path))
                try:
                    geometry = Geometry(mesh, segmenter, image)
                except ValueError:
                    print(f'{path.name}: no face found, skipped')
                    continue
                cropped = crop(image, geometry)
                name = f'{identity}_{variant}.jpg'
                cv2.imwrite(str(args.out / name), cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append(dict(file=name, identity=identity, split=split, variant=variant, broken_rule=rule or ''))

                if variant != 'compliant':
                    continue
                base_geometry = Geometry(mesh, segmenter, cropped)
                for synthetic_variant, synthetic_rule in SYNTHETIC_VARIANTS.items():
                    out = synthetic(synthetic_variant, cropped, base_geometry, image, geometry)
                    name = f'{identity}_{synthetic_variant}.jpg'
                    cv2.imwrite(str(args.out / name), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rows.append(dict(file=name, identity=identity, split=split, variant=synthetic_variant,
                                     broken_rule=synthetic_rule or ''))
            print(identity, end=' ', flush=True)

    with open(args.out / 'manifest.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'identity', 'split', 'variant', 'broken_rule'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n{len(rows)} photos -> {args.out}')


if __name__ == '__main__':
    main()
