"""Pin the /v1/check response contract that API clients depend on."""
import io
import pathlib

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from edutap.image_api.check_api import app
from edutap.image_api.core.rules import RULES

IMAGES = pathlib.Path(__file__).parent / 'test-images'
CHECK_STATUSES = {'pass', 'warn', 'fail', 'skipped', 'error'}
RULE_STATUSES = {'pass', 'warn', 'fail', 'skipped', 'not_checked'}


@pytest.fixture(scope='module')
def client():
    with TestClient(app) as client:
        yield client


def _post(client, data: bytes, name='photo.jpg', content_type='image/jpeg'):
    return client.post('/v1/check', files={'file': (name, data, content_type)})


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def test_check_reports_every_check_and_rule(client):
    response = _post(client, (IMAGES / '9.jpg').read_bytes())
    assert response.status_code == 200
    body = response.json()

    names = app.state.validator.check_names()
    assert [check['name'] for check in body['checks']] == names
    assert all(check['status'] in CHECK_STATUSES for check in body['checks'])
    assert [rule['id'] for rule in body['rules']] == [rule.id for rule in RULES]
    assert all(rule['status'] in RULE_STATUSES for rule in body['rules'])
    assert {group['id'] for group in body['groups']} >= {rule['group'] for rule in body['rules']}
    assert body['image'] == {'width': 2000, 'height': 2000}

    # A tight, frontal crop on a plain background
    by_name = {check['name']: check for check in body['checks']}
    assert by_name['face_size']['status'] == 'pass'
    assert by_name['face_sharpness']['status'] == 'pass'
    assert by_name['exposure']['status'] == 'pass'


def test_photo_without_face_fails_the_gate_and_skips_the_rest(client):
    response = _post(client, _png(Image.new('RGB', (700, 900), (200, 200, 200))), 'grey.png', 'image/png')
    assert response.status_code == 200
    checks = response.json()['checks']

    assert checks[0] == {'name': 'face_detected', 'status': 'fail', 'reasons': ['No face detected'], 'measurements': {}}
    assert {check['status'] for check in checks[1:]} == {'skipped'}


def test_non_image_is_rejected_with_a_code(client):
    response = _post(client, b'not an image', 'notes.txt', 'text/plain')
    assert response.status_code == 400
    assert response.json()['code'] == 'invalid_image'


def test_missing_file_is_rejected_with_a_code(client):
    response = client.post('/v1/check')
    assert response.status_code == 422
    assert response.json()['code'] == 'invalid_request'


def test_rules_endpoint_lists_the_catalogue(client):
    body = client.get('/v1/rules').json()
    assert [rule['id'] for rule in body['rules']] == [rule.id for rule in RULES]
