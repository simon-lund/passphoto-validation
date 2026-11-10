"""Pin the /v1/check response contract that API clients depend on."""
import base64
import io
import json
import pathlib
import re

import pytest
from fastapi.testclient import TestClient
from markupsafe import escape
from PIL import Image

from passphoto_validation.api_schema import CheckReport, ErrorResponse
from passphoto_validation.check_api import app, prefers_html
from passphoto_validation.core.rules import RULES

IMAGES = pathlib.Path(__file__).parent / 'test-images'
CHECK_STATUSES = {'pass', 'warn', 'fail', 'skipped', 'error'}
RULE_STATUSES = {'pass', 'warn', 'fail', 'skipped', 'not_checked'}


CROP_KEYS = {'status', 'reason', 'x', 'y', 'width', 'height', 'is_valid', 'rules_after', 'fixed_rules', 'remaining_rules'}


@pytest.fixture(scope='module')
def client():
    # Crashes answer as 500, as on a server, so their contract can be tested too
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _post(client, data: bytes, name='photo.jpg', content_type='image/jpeg', accept=None):
    headers = {'Accept': accept} if accept else {}
    return client.post('/v1/check', files={'file': (name, data, content_type)}, headers=headers)


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def test_check_reports_every_check_and_rule(client):
    response = _post(client, (IMAGES / '9.jpg').read_bytes())
    assert response.status_code == 200
    # JSON stays the default (the test client sends Accept: */*)
    assert response.headers['content-type'] == 'application/json'
    assert response.headers['vary'] == 'Accept'
    body = response.json()

    names = app.state.validator.check_names()
    assert [check['name'] for check in body['checks']] == names
    assert all(check['status'] in CHECK_STATUSES for check in body['checks'])
    assert [rule['id'] for rule in body['rules']] == [rule.id for rule in RULES]
    assert all(rule['status'] in RULE_STATUSES for rule in body['rules'])
    assert {group['id'] for group in body['groups']} >= {rule['group'] for rule in body['rules']}
    assert body['image'] == {'width': 2000, 'height': 2000}
    # The answer matches the published model key for key (extra keys are forbidden there)
    CheckReport.model_validate(body)

    # A square photo: the planned 35x45 crop is part of the answer, with the same keys whatever its status
    crop = body['crop']
    assert set(crop) == CROP_KEYS
    assert crop['status'] == 'proposed' and crop['reason'] is None and 'photo_format' in crop['fixed_rules']
    assert set(crop['rules_after']) == {rule['id'] for rule in body['rules']}

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
    CheckReport.model_validate(response.json())


def test_non_image_is_rejected_with_a_code(client):
    response = _post(client, b'not an image', 'notes.txt', 'text/plain')
    assert response.status_code == 400
    assert response.json()['code'] == 'invalid_image'


def test_missing_file_is_rejected_with_a_code(client):
    response = client.post('/v1/check')
    assert response.status_code == 422
    assert response.json()['code'] == 'invalid_request'


def test_openapi_describes_the_answers(client):
    spec = client.get('/openapi.json').json()
    responses = spec['paths']['/v1/check']['post']['responses']
    assert responses['200']['content']['application/json']['schema'] == {'$ref': '#/components/schemas/CheckReport'}
    assert 'text/html' in responses['200']['content']
    for status in ('400', '413', '422', '500'):
        assert responses[status]['content']['application/json']['schema'] == {'$ref': '#/components/schemas/ErrorResponse'}
    crop = spec['components']['schemas']['Crop']['properties']
    assert set(crop) == CROP_KEYS


def test_errors_match_the_published_model(client):
    for response in (_post(client, b'', 'empty.jpg'), _post(client, b'not an image', 'notes.txt', 'text/plain'), client.post('/v1/check')):
        ErrorResponse.model_validate(response.json())


def test_a_crash_answers_json_with_a_code_and_no_internals(client, monkeypatch):
    def crash(image_np):
        raise RuntimeError('secret path /srv/models')

    monkeypatch.setattr(app.state, 'validator', crash)
    response = _post(client, (IMAGES / '9.jpg').read_bytes(), accept='text/html')
    assert response.status_code == 500
    assert response.headers['content-type'] == 'application/json'
    body = response.json()
    assert body['code'] == 'internal_error' and set(body) == {'code', 'detail'}
    assert 'secret' not in body['detail']


def test_html_report_when_the_request_accepts_html(client):
    response = _post(client, (IMAGES / '9.jpg').read_bytes(), '9.jpg', accept='text/html')
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/html')
    assert response.headers['vary'] == 'Accept'

    page = response.text
    assert all(str(escape(rule.title)) in page for rule in RULES)
    assert '9.jpg' in page
    # Self-contained: the photo is embedded and nothing is loaded from elsewhere
    links = re.findall(r'(?:src|href)="([^"]*)"', page)
    assert links and all(link.startswith(('data:image/jpeg;base64,', 'data:application/json;', '#rule-', '#status-codes')) for link in links)
    assert '<link' not in page and not re.search(r'<script[^>]*\ssrc=', page)

    # Copy and Download hand out the JSON answer for the same photo
    embedded = re.search(r'href="data:application/json;charset=utf-8;base64,([^"]+)" download="9.json"', page).group(1)
    assert json.loads(base64.b64decode(embedded)) == _post(client, (IMAGES / '9.jpg').read_bytes()).json()


def test_errors_stay_json_when_html_is_accepted(client):
    response = _post(client, b'not an image', 'notes.txt', 'text/plain', accept='text/html')
    assert response.status_code == 400
    assert response.json()['code'] == 'invalid_image'


@pytest.mark.parametrize('accept, html', [
    (None, False),
    ('*/*', False),
    ('application/json', False),
    ('text/html', True),
    ('text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', True),  # what browsers send
    ('application/json, text/html;q=0.5', False),
    ('text/html, application/json', False),  # a tie keeps JSON
    ('text/*', True),
    ('text/html;q=0', False),
    ('image/png', False),
])
def test_html_only_when_the_accept_header_prefers_it(accept, html):
    assert prefers_html(accept) is html


def test_rules_endpoint_lists_the_catalogue(client):
    body = client.get('/v1/rules').json()
    assert [rule['id'] for rule in body['rules']] == [rule.id for rule in RULES]


def test_thresholds_endpoint_lists_what_the_server_runs_with(client):
    body = client.get('/v1/thresholds').json()
    assert body['file'] is None
    assert body['checks']['colour_cast'] == dict(warn_above=12, fail_above=20, min_whites_share=0.02, band_share=0.03)
    assert body['checks']['ofiq_head_yaw'] == dict(warn_above=5, fail_above=10)
