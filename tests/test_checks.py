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

from edutap.image_api.core import PassportPhotoValidator
from edutap.image_api.core.config import ValidatorConfig
from edutap.image_api.core.utils import convert_to_bgr

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
    return {check.name: check.status for check in validator(image).checks}


def _jpeg(image_bgr, quality):
    buffer = io.BytesIO()
    Image.fromarray(image_bgr[..., ::-1]).save(buffer, format='JPEG', quality=quality)
    return convert_to_bgr(Image.open(buffer))


def test_original_passes_the_classic_checks(validator, photo):
    statuses = _statuses(validator, photo)
    for name in ('exposure', 'face_sharpness', 'compression', 'red_eyes', 'gaze', 'background_brightness',
                 'background_pattern'):
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


def test_warm_colour_cast_passes_red_eyes(validator, photo):
    warm = np.clip(photo * np.array((0.6, 0.95, 1.3)), 0, 255).astype(np.uint8)
    assert _statuses(validator, warm)['red_eyes'] == 'pass'


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
