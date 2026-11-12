"""scripts/review.py: the server behind the review page. No model is loaded here; the validator is a stand-in."""
import argparse
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import review  # noqa: E402


def _items(count):
    return [dict(i=i, file=f'{i}.jpg', split='', variant='', broken_rule='', reviewed=None) for i in range(count)]


def test_a_request_replaced_by_a_newer_one_answers_at_once():
    # No worker runs, so nothing is ever computed: a request can only end as superseded or timed out
    computer = review.Computer(validator=None, folder=Path('.'), items=_items(40), crop=False)
    answers = {}
    waiter = threading.Thread(target=lambda: answers.setdefault('old', computer.get(3, request=1, timeout=30)))
    waiter.start()
    time.sleep(0.05)
    started = time.monotonic()
    assert computer.get(30, request=2, timeout=0.05) == review.Computer.TIMED_OUT
    waiter.join(timeout=5)
    assert answers['old'] == review.Computer.SUPERSEDED and time.monotonic() - started < 5
    # A request that arrives late does not pull the window back from the photo on screen
    assert computer.get(3, request=1, timeout=30) == review.Computer.SUPERSEDED
    assert computer.wanted == 30


def test_a_photo_the_pipeline_cannot_draw_ends_as_an_error_not_as_a_dead_worker(tmp_path):
    from PIL import Image
    Image.new('RGB', (70, 90), 'white').save(tmp_path / '0.jpg')
    # inspect() answers with objects verdict() cannot read: the failure is after the validator ran
    validator = SimpleNamespace(inspect=lambda image, crop: (object(), object()))
    computer = review.Computer(validator, tmp_path, _items(1), crop=False)
    result = computer.compute(0)
    assert result['file'] == '0.jpg' and 'AttributeError' in result['error']
    # And the thread survives whatever compute raises
    computer.compute = lambda i: (_ for _ in ()).throw(RuntimeError('boom'))
    computer.start()
    assert 'boom' in computer.get(0, timeout=5)['error']
    assert computer.is_alive()


@pytest.fixture
def client(tmp_path):
    args = argparse.Namespace(set=tmp_path, split=None, variant=None, check=None, rule=None, shuffle=False, again=False)
    items = _items(3)
    computer = SimpleNamespace(get=lambda i, request=None: {0: dict(i=0, file='0.jpg', error=None),
                                                           1: review.Computer.SUPERSEDED,
                                                           2: review.Computer.TIMED_OUT}[i])
    app = review.make_app(args, items, {}, computer, SimpleNamespace(thresholds_file=None))
    return TestClient(app, raise_server_exceptions=False), tmp_path


def test_every_error_answer_carries_a_code(client):
    http, _ = client
    assert http.get('/api/item/0').json()['file'] == '0.jpg'
    for request, status, code in (
        (lambda: http.get('/api/item/7'), 404, 'photo_not_in_queue'),
        (lambda: http.get('/api/photo/7'), 404, 'photo_not_in_queue'),
        (lambda: http.get('/api/item/1'), 409, 'superseded'),
        (lambda: http.get('/api/item/2'), 504, 'pipeline_timeout'),
        (lambda: http.get('/api/item/x'), 422, 'invalid_request'),
        (lambda: http.post('/api/review', json=dict(i=7, agree=True)), 400, 'photo_not_in_queue'),
        (lambda: http.post('/api/review', json=dict(i=0)), 400, 'answer_missing'),
        (lambda: http.post('/api/review', content='no json'), 422, 'invalid_request'),
    ):
        response = request()
        assert (response.status_code, response.json()['code']) == (status, code)
        assert response.json()['detail']


def test_answering_a_photo_again_moves_its_count_and_adds_none(client):
    http, folder = client
    assert http.post('/api/review', json=dict(i=0, agree=True)).json() == dict(counts=dict(right=1))
    assert http.post('/api/review', json=dict(i=1, agree=False)).json() == dict(counts=dict(right=1, wrong=1))
    # The same photo again, with the other answer: the last answer counts
    assert http.post('/api/review', json=dict(i=0, agree=False)).json() == dict(counts=dict(wrong=2))
    records = [json.loads(line) for line in (folder / 'reviews.jsonl').read_text().splitlines()]
    assert [(r['file'], r['agree']) for r in records] == [('0.jpg', True), ('1.jpg', False), ('0.jpg', False)]
    assert http.get('/api/queue').json()['counts'] == dict(wrong=2)
