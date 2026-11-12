"""Behaviour of individual checks on altered copies of a photo that passes.

The OFIQ tests run only when OFIQ_CONFIG_DIR points at a built OFIQ (see core/ofiq.py).
"""
import io
import os
import pathlib

import cv2
import mediapipe as mp
import numpy as np
import pytest
from PIL import Image

from passphoto_validation.core import PassportPhotoValidator
from passphoto_validation.core.config import ValidatorConfig
from passphoto_validation.core.models import Crop
from passphoto_validation.core.utils import convert_to_bgr
from passphoto_validation.core.validator import CHECK_DEPENDENCIES

IMAGES = pathlib.Path(__file__).parent / 'test-images'
needs_ofiq = pytest.mark.skipif(not os.environ.get('OFIQ_CONFIG_DIR'), reason='OFIQ_CONFIG_DIR is not set')
needs_siglip = pytest.mark.skipif(not (os.environ.get('OFIQ_CONFIG_DIR') and os.environ.get('SIGLIP_ENABLED')),
                                  reason='SIGLIP_ENABLED or OFIQ_CONFIG_DIR is not set')


@pytest.fixture(scope='module')
def validator():
    return PassportPhotoValidator()


@pytest.fixture(scope='module')
def photo():
    # A tight, frontal crop on a plain background; downscaled to keep the tests fast
    image = Image.open(IMAGES / '9.jpg')
    image.thumbnail((1000, 1000))
    return convert_to_bgr(image)


@pytest.fixture(scope='module')
def id_photo(photo):
    # The same photo cut to the 35x45 format, so that no check finds an error
    width = round(photo.shape[0] * 35 / 45)
    left = (photo.shape[1] - width) // 2
    return np.ascontiguousarray(photo[:, left:left + width])


def _statuses(validator, image):
    # Check statuses only; the crop would run the checks a second time on these square photos
    return {check.name: check.status for check in validator(image, crop=False).checks}


def _jpeg(image_bgr, quality):
    buffer = io.BytesIO()
    Image.fromarray(image_bgr[..., ::-1]).save(buffer, format='JPEG', quality=quality)
    return convert_to_bgr(Image.open(buffer))


def test_original_passes_the_classic_checks(validator, photo):
    statuses = _statuses(validator, photo)
    for name in ('exposure', 'face_sharpness', 'compression', 'red_eyes', 'gaze', 'background_brightness',
                 'background_pattern', 'background_gradient', 'background_shadow', 'print_defects', 'skin_smoothing',
                 'skin_colour_patches', 'colour_cast'):
        assert statuses[name] == 'pass', name


def test_darkened_photo_fails_exposure(validator, photo):
    assert _statuses(validator, (photo * 0.3).astype(np.uint8))['exposure'] == 'fail'


def test_blurred_photo_fails_sharpness(validator, photo):
    blurred = np.array(Image.fromarray(photo).resize((250, 250)).resize(photo.shape[1::-1]))
    assert _statuses(validator, blurred)['face_sharpness'] == 'fail'


def test_heavily_compressed_photo_fails_compression(validator, photo):
    assert _statuses(validator, _jpeg(photo, 4))['compression'] == 'fail'


def _background(image_bgr):
    # Where the selfie segmentation sees background, as the background checks do
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0) as segmentation:
        return segmentation.process(np.ascontiguousarray(image_bgr[..., ::-1])).segmentation_mask < 0.5


def test_dark_background_fails_background_brightness(validator, photo):
    dark = photo.copy()
    background = _background(photo)
    dark[background] = (dark[background] * 0.45).astype(np.uint8)
    assert _statuses(validator, dark)['background_brightness'] == 'fail'


def test_faint_stripes_on_the_background_fail_background_pattern(validator, photo):
    # Light diagonal stripes, faint as a flag's emblem
    rows, cols = np.mgrid[:photo.shape[0], :photo.shape[1]]
    stripes = ((rows + cols) % 24 < 6) & _background(photo)
    striped = np.clip(photo + stripes[..., None] * 20, 0, 255).astype(np.uint8)
    assert _statuses(validator, striped)['background_pattern'] == 'fail'


def test_shadow_next_to_the_person_fails_background_shadow(validator, photo):
    # The person's silhouette shifted right and down, darkened: a cast shadow from a flash on the left
    h, w = photo.shape[:2]
    person = ~_background(photo)
    shifted = np.zeros_like(person)
    shifted[int(h * 0.02):, int(w * 0.06):] = person[:h - int(h * 0.02), :w - int(w * 0.06)]
    shade = cv2.GaussianBlur((shifted & ~person).astype(np.float32), (0, 0), w * 0.01)
    shadowed = np.clip(photo * (1 - 0.4 * shade)[..., None], 0, 255).astype(np.uint8)
    statuses = _statuses(validator, shadowed)
    assert statuses['background_shadow'] == 'fail'
    assert statuses['background_gradient'] == 'pass'


def test_shadow_hugging_the_person_is_found_in_the_rim_of_the_mask(validator, photo):
    # A narrow shadow right next to the person lies within the band the check leaves out around the
    # segmentation's edge (or inside the mask, when the segmentation takes it for part of the person);
    # the rim measurement finds it where the band measurement cannot
    h, w = photo.shape[:2]
    person = ~_background(photo)
    shifted = np.zeros_like(person)
    shifted[int(h * 0.01):, int(w * 0.03):] = person[:h - int(h * 0.01), :w - int(w * 0.03)]
    shade = cv2.GaussianBlur((shifted & ~person).astype(np.float32), (0, 0), w * 0.01)
    shadowed = np.clip(photo * (1 - 0.4 * shade)[..., None], 0, 255).astype(np.uint8)
    report = validator(shadowed, crop=False)
    assert report.measurements['background_shadow_depth'] < 5
    assert report.measurements['background_shadow_rim_depth'] > 9
    assert {check.name: check.status for check in report.checks}['background_shadow'] == 'fail'
    # The plain photo's rim is just as quiet as its band
    assert validator(photo, crop=False).measurements['background_shadow_rim_depth'] < 5


def test_background_running_light_to_dark_fails_background_gradient(validator, photo):
    # The background (L* 59) darkens to 30% from left to right: a change of about 40 L*
    factor = np.where(_background(photo), np.linspace(1.0, 0.3, photo.shape[1], dtype=np.float32)[None, :], 1.0)
    graded = np.clip(photo * factor[..., None], 0, 255).astype(np.uint8)
    statuses = _statuses(validator, graded)
    assert statuses['background_gradient'] == 'fail'
    assert statuses['background_shadow'] == 'pass'


def test_spots_on_the_background_fail_print_defects(validator, photo):
    h, w = photo.shape[:2]
    stained = photo.copy()
    for fx, fy, radius in ((0.08, 0.1, 0.02), (0.92, 0.1, 0.015), (0.06, 0.5, 0.012)):
        cv2.circle(stained, (int(w * fx), int(h * fy)), int(w * radius), (30, 40, 60), -1)
    stained = np.where(_background(photo)[..., None], stained, photo)
    assert _statuses(validator, stained)['print_defects'] == 'fail'


def test_smoothed_skin_warns_skin_smoothing(validator, photo):
    # A beauty filter: the skin flattened, the eyes left as they are
    smooth = cv2.edgePreservingFilter(photo, flags=cv2.RECURS_FILTER, sigma_s=40, sigma_r=0.25)
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as mesh:
        points = mesh.process(np.ascontiguousarray(photo[..., ::-1])).multi_face_landmarks[0].landmark
    h, w = photo.shape[:2]
    for outer, inner in ((33, 133), (362, 263)):
        x0, x1 = sorted((int(points[outer].x * w), int(points[inner].x * w)))
        y = int((points[outer].y + points[inner].y) / 2 * h)
        width = x1 - x0
        smooth[y - width // 2:y + width // 2, x0 - width // 4:x1 + width // 4] = photo[y - width // 2:y + width // 2, x0 - width // 4:x1 + width // 4]
    assert _statuses(validator, smooth)['skin_smoothing'] == 'warn'


def _mesh_points(image_bgr):
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as mesh:
        points = mesh.process(np.ascontiguousarray(image_bgr[..., ::-1])).multi_face_landmarks[0].landmark
    h, w = image_bgr.shape[:2]
    return np.array([(p.x * w, p.y * h) for p in points])


def test_whole_face_smoothing_warns_skin_smoothing(validator, photo):
    # A filter over the whole face, eyes included: the eyes are as flat as the skin, so the texture ratio stays
    # normal, but the edges of the face keep their strength while the skin's fine texture is gone
    points = _mesh_points(photo)
    x0, y0 = (points.min(axis=0) - 0.1 * points.ptp(axis=0)).clip(0).astype(int)
    x1, y1 = (points.max(axis=0) + 0.1 * points.ptp(axis=0)).astype(int)
    smooth = photo.copy()
    smooth[y0:y1, x0:x1] = cv2.edgePreservingFilter(photo[y0:y1, x0:x1], flags=cv2.RECURS_FILTER, sigma_s=40, sigma_r=0.25)
    statuses = _statuses(validator, smooth)
    assert statuses['skin_smoothing'] == 'warn'
    assert statuses['face_sharpness'] == 'pass'


def test_blurred_photo_fails_sharpness_without_a_retouching_warning(validator, photo):
    # An optical blur flattens the skin too, but the sharpness check reports it, so skin_smoothing stays quiet
    statuses = _statuses(validator, cv2.GaussianBlur(photo, (0, 0), 4))
    assert statuses['face_sharpness'] == 'fail'
    assert statuses['skin_smoothing'] == 'pass'


def _stickers(photo):
    # A face filter's stickers: a pink nose and a coloured butterfly on each cheek
    points = _mesh_points(photo)
    face_width = np.linalg.norm(points[234] - points[454])
    stickered = photo.copy()
    radius = int(face_width * 0.08)
    cv2.ellipse(stickered, tuple(points[1].astype(int)), (radius, int(radius * 0.7)), 0, 0, 360, (160, 120, 240), -1, cv2.LINE_AA)
    for cheek, colour in ((50, (200, 80, 255)), (280, (255, 160, 0))):
        cx, cy = points[cheek].astype(int)
        size = int(face_width * 0.07)
        for side in (-1, 1):
            cv2.ellipse(stickered, (cx + side * size // 2, cy), (size // 2, int(size * 0.7)), side * 25, 0, 360, colour, -1, cv2.LINE_AA)
    return stickered


def test_coloured_stickers_on_the_face_warn_skin_colour_patches(validator, photo):
    assert _statuses(validator, _stickers(photo))['skin_colour_patches'] == 'fail'


def test_stickers_still_fail_with_the_warn_level_turned_off(photo, tmp_path):
    # Turning the warning off must not take the fail with it
    file = tmp_path / 'thresholds.toml'
    file.write_text('[skin_colour_patches]\nwarn_share_above = false\n')
    validator = PassportPhotoValidator(ValidatorConfig(ofiq_config_dir=None, thresholds_file=str(file)))
    assert _statuses(validator, _stickers(photo))['skin_colour_patches'] == 'fail'


def test_the_head_size_message_names_only_levels_that_are_on(photo, tmp_path):
    file = tmp_path / 'thresholds.toml'
    file.write_text('[face_size]\nwarn_below = false\nwarn_above = false\nfail_below = 0.95\nfail_above = false\n')
    validator = PassportPhotoValidator(ValidatorConfig(ofiq_config_dir=None, thresholds_file=str(file)))
    check = {check.name: check for check in validator(photo, crop=False).checks}['face_size']
    assert check.status == 'fail'
    assert check.reasons[0].startswith('Head is too small') and check.reasons[0].endswith('Should be at least 95%.')


def _red_pupils(image_bgr):
    # Paints the pupils red, as a flash reflected by the retina does
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as mesh:
        points = mesh.process(np.ascontiguousarray(image_bgr[..., ::-1])).multi_face_landmarks[0].landmark
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()
    for iris, edge in ((468, 469), (473, 474)):
        centre = np.array([points[iris].x * w, points[iris].y * h])
        radius = np.linalg.norm(np.array([points[edge].x * w, points[edge].y * h]) - centre)
        cv2.circle(out, tuple(int(v) for v in centre.round()), int(radius * 0.6), (30, 30, 200), -1)
    return out


def test_red_pupils_fail_red_eyes(validator, photo):
    assert _statuses(validator, _red_pupils(photo))['red_eyes'] == 'fail'


def _warm(image_bgr):
    # The colour cast of the London Set's colour_cast variant: less blue, more red
    return np.clip(image_bgr * np.array((0.6, 0.95, 1.3)), 0, 255).astype(np.uint8)


def test_washed_out_photo_fails_contrast(validator, photo):
    # The London Set's low-contrast variant: the grey levels pulled towards the mean
    flat = (photo.mean() + (photo - photo.mean()) * 0.3).astype(np.uint8)
    assert _statuses(validator, flat)['contrast'] == 'fail'


def test_warm_colour_cast_passes_red_eyes(validator, photo):
    assert _statuses(validator, _warm(photo))['red_eyes'] == 'pass'


def test_warm_colour_cast_fails_colour_cast(validator, photo):
    # The grey background and the whites of the eyes turn orange together
    report = validator(_warm(photo), crop=False)
    check = {c.name: c for c in report.checks}['colour_cast']
    assert check.status == 'fail'
    assert check.measurements['colour_cast'] > 20
    assert check.measurements['background_colour_b'] > 20 and check.measurements['eye_whites_colour_b'] > 20


def test_coloured_background_behind_natural_eyes_is_no_colour_cast(validator, photo):
    # A light blue backdrop: the background is tinted, the eye whites are not, so nothing is shared
    blue = photo.copy().astype(np.float32)
    background = _background(photo)
    blue[background] = np.clip(blue[background] * np.array((1.15, 1.0, 0.8)), 0, 255)
    check = {c.name: c for c in validator(blue.astype(np.uint8), crop=False).checks}['colour_cast']
    assert check.status == 'pass'
    assert check.measurements['background_colour_b'] < -10
    assert check.measurements['colour_cast'] < 5


def test_without_ofiq_its_checks_are_skipped_with_the_reason(photo):
    validator = PassportPhotoValidator(ValidatorConfig(ofiq_config_dir=None))
    checks = {check.name: check for check in validator(photo).checks}
    assert checks['ofiq_head_yaw'].status == 'skipped'
    assert checks['ofiq_head_yaw'].reasons == ['OFIQ is not configured (set OFIQ_CONFIG_DIR).']
    assert checks['glasses_tint'].status == 'skipped'


def test_photo_with_rules_no_check_could_judge_is_not_valid(id_photo):
    report = PassportPhotoValidator(ValidatorConfig(ofiq_config_dir=None))(id_photo)
    assert not [issue for issue in report.issues if issue.category == 'error']
    assert 'no_hat' in report.skipped_rules
    assert not report.is_valid


@needs_ofiq
def test_photo_with_every_rule_judged_and_no_error_is_valid(validator, id_photo):
    # The rules no check covers (not_checked) don't keep a photo from being valid
    report = validator(id_photo)
    assert report.skipped_rules == []
    assert report.is_valid


def test_square_photo_gets_a_35x45_crop_that_fixes_the_format(photo):
    report = PassportPhotoValidator(ValidatorConfig(ofiq_config_dir=None))(photo)
    crop = report.crop
    assert crop.status == 'proposed'
    assert crop.height == photo.shape[0] and abs(crop.width / crop.height - 35 / 45) < 0.002
    assert 'photo_format' in crop.fixed_rules


def test_framed_photo_needs_no_crop(id_photo):
    report = PassportPhotoValidator(ValidatorConfig(ofiq_config_dir=None))(id_photo)
    assert report.crop == Crop('not_needed')


def test_the_crop_never_changes_the_verdict(validator, photo):
    with_crop, without_crop = validator(photo), validator(photo, crop=False)
    assert without_crop.crop is None
    assert with_crop.is_valid == without_crop.is_valid
    assert [(rule.id, rule.status) for rule in with_crop.rules] == [(rule.id, rule.status) for rule in without_crop.rules]


@needs_ofiq
def test_square_photo_is_valid_after_the_crop(validator, photo):
    report = validator(photo)
    assert not report.is_valid
    assert report.crop.is_valid and report.crop.remaining_rules == []


@needs_ofiq
def test_photo_with_a_crashed_check_is_not_valid(validator, id_photo, monkeypatch):
    def crash(context):
        raise ValueError('boom')

    # face_centered shares the template_zones rule with face_size, which passes. The validator resolves its
    # check list once, when it is built, so the crash goes into that list
    with_crash = [(name, crash if name == 'face_centered' else check, kwargs) for name, check, kwargs in validator._checks()]
    monkeypatch.setattr(validator, '_check_list', with_crash)
    report = validator(id_photo)
    checks = {check.name: check for check in report.checks}
    assert checks['face_centered'].status == 'error' and checks['face_size'].status == 'pass'
    assert {'face_centered', 'template_zones'} <= set(report.skipped_rules)
    assert not report.is_valid


def test_without_siglip_its_checks_are_skipped_with_the_reason(photo):
    validator = PassportPhotoValidator(ValidatorConfig(siglip_enabled=False))
    checks = {check.name: check for check in validator(photo).checks}
    for name in ('siglip_glasses_glare', 'siglip_gaze'):
        assert checks[name].status == 'skipped'
        assert checks[name].reasons == ['SigLIP is not enabled (set SIGLIP_ENABLED=true).']


@needs_siglip
def test_original_passes_the_siglip_checks(validator, photo):
    statuses = _statuses(validator, photo)
    assert statuses['siglip_glasses_glare'] == 'pass'
    assert statuses['siglip_gaze'] == 'pass'


@needs_siglip
def test_siglip_without_ofiq_is_skipped_with_the_reason(photo):
    validator = PassportPhotoValidator(ValidatorConfig(ofiq_config_dir=None, siglip_enabled=True))
    check = next(check for check in validator(photo).checks if check.name == 'siglip_gaze')
    assert check.status == 'skipped'
    assert check.reasons == ["SigLIP needs OFIQ's eye landmarks: OFIQ is not configured (set OFIQ_CONFIG_DIR)."]


@needs_ofiq
def test_original_passes_the_ofiq_checks(validator, photo):
    statuses = _statuses(validator, photo)
    for name in ('ofiq_head_yaw', 'ofiq_head_pitch', 'ofiq_head_roll', 'ofiq_expression', 'ofiq_eyes_open',
                 'ofiq_overexposure', 'ofiq_illumination', 'glasses_tint', 'glasses_glare'):
        assert statuses[name] == 'pass', name


def _sunglasses(image_bgr, tint=0.85):
    # Dark lenses in a dark frame over each eye, as scripts/build_eval_set.py draws its sunglasses variant
    points = _mesh_points(image_bgr)
    out = image_bgr.copy()
    eye_distance = float(np.linalg.norm(points[33] - points[263]))
    lens = np.zeros(image_bgr.shape[:2], np.uint8)
    centres = []
    for outer, inner in ((33, 133), (263, 362)):
        centre = tuple(int(v) for v in (points[outer] + points[inner]) / 2)
        centres.append(centre)
        cv2.ellipse(lens, centre, (int(eye_distance * 0.3), int(eye_distance * 0.22)), 0, 0, 360, 255, -1)
    alpha = (lens[..., None] / 255.0) * tint
    out = (out * (1 - alpha) + np.array((25, 20, 20)) * alpha).astype(np.uint8)
    frame = max(2, int(eye_distance * 0.035))
    for centre in centres:
        cv2.ellipse(out, centre, (int(eye_distance * 0.3), int(eye_distance * 0.22)), 0, 0, 360, (15, 15, 15), frame)
    (ax, ay), (bx, by) = centres
    cv2.line(out, (int(ax + eye_distance * 0.3), ay), (int(bx - eye_distance * 0.3), by), (15, 15, 15), frame)
    return out


@needs_ofiq
def test_covered_eyes_stop_the_checks_that_need_them(validator, photo):
    report = validator(_sunglasses(photo), crop=False)
    checks = {check.name: check for check in report.checks}
    root = checks['ofiq_eyes_visible']
    assert root.status == 'fail'
    reason = f'Not judged: {root.reasons[0]}'
    for name in CHECK_DEPENDENCIES['ofiq_eyes_visible']:
        assert checks[name].status == 'skipped' and checks[name].reasons == [reason], name
    rules = {rule.id: rule for rule in report.rules}
    for rule_id in ('eyes_open', 'no_red_eyes', 'eyes_facing_camera', 'glasses_no_reflections'):
        assert rules[rule_id].status == 'skipped' and rules[rule_id].reasons == [reason], rule_id
    # Rules another check still judges keep that verdict
    assert rules['natural_colour'].status == 'pass' and rules['no_reflections'].status == 'pass'
    assert not report.is_valid

    # One problem for the covered eyes: the rules it breaks and the ones it left unjudged, none for the stopped checks
    problem = next(problem for problem in report.problems if problem.check == 'ofiq_eyes_visible')
    assert problem.rules == ['eyes_not_covered', 'glasses_not_tinted', 'glasses_frame_clear']
    assert problem.not_judged == ['no_red_eyes', 'eyes_open', 'eyes_facing_camera', 'glasses_no_reflections']
    assert not {problem.check for problem in report.problems} & set(CHECK_DEPENDENCIES['ofiq_eyes_visible'])


@needs_ofiq
def test_a_root_turned_off_in_the_thresholds_file_stops_nothing(photo, tmp_path):
    file = tmp_path / 'thresholds.toml'
    file.write_text('[ofiq_eyes_visible]\nenabled = false\n')
    validator = PassportPhotoValidator(ValidatorConfig(thresholds_file=str(file)))
    checks = {check.name: check for check in validator(_sunglasses(photo), crop=False).checks}
    assert 'ofiq_eyes_visible' not in checks
    # Without the root, the eye checks run as before, on the lenses
    for name in CHECK_DEPENDENCIES['ofiq_eyes_visible']:
        assert checks[name].status != 'skipped' or 'Not judged' not in checks[name].reasons[0], name
    assert checks['glasses_tint'].status == 'fail'


@needs_ofiq
def test_a_root_measured_early_keeps_the_report_in_check_order(validator, photo):
    # ofiq_eyes_visible runs before colour_cast, its first dependent; the issues and measurements still
    # come in check order, as before
    report = validator(photo, crop=False)
    position = {name: i for i, name in enumerate(validator.check_names())}
    positions = [position[issue.check] for issue in report.issues]
    assert positions == sorted(positions)
    assert list(report.measurements) == list(dict.fromkeys(key for check in report.checks for key in check.measurements))


@needs_ofiq
def test_brightened_photo_fails_overexposure(validator, photo):
    statuses = _statuses(validator, np.clip(photo.astype(int) * 1.6 + 60, 0, 255).astype(np.uint8))
    assert statuses['ofiq_overexposure'] == 'fail'
    assert statuses['exposure'] == 'pass'


@needs_ofiq
def test_rotated_photo_fails_head_roll(validator, photo):
    rotated = np.array(Image.fromarray(photo).rotate(20, fillcolor=(140, 140, 140)))
    assert _statuses(validator, rotated)['ofiq_head_roll'] == 'fail'


@needs_ofiq
def test_mirrored_photo_keeps_the_head_pose(validator, photo):
    # The old solvePnP pose read 8 deg yaw on the original and 14 deg on the mirror
    report, mirrored = validator(photo), validator(np.ascontiguousarray(photo[:, ::-1]))
    yaw = [next(c for c in r.checks if c.name == 'ofiq_head_yaw').measurements['head_yaw_degrees_ofiq']
           for r in (report, mirrored)]
    assert abs(abs(yaw[0]) - abs(yaw[1])) < 2
