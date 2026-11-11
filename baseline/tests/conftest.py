"""A fake client and Decisions built from the documented JSON, so the tests never reach the network."""
from pathlib import Path

import pytest
from openai.types.decision import Decision
from PIL import Image

from passphoto_decisions.rules import GATE_NAME, RULES

FACE_COUNTS = ('none', 'one', 'several')


def make_decision(probabilities=None, gate='one', refuse=(), model='gpt-6-luna', input_tokens=1500) -> Decision:
    """A Decision as the API answers it. `gate=None` makes the model refuse the face count."""
    probabilities = probabilities or {}
    if gate is None:
        answers = [{'type': 'refusal', 'name': GATE_NAME}]
    else:
        answers = [{
            'type': 'choice', 'name': GATE_NAME, 'choice': gate, 'confidence': 0.9,
            'probabilities': [{'value': value, 'probability': 0.9 if value == gate else 0.05} for value in FACE_COUNTS],
        }]
    for rule in RULES:
        if rule.id in refuse:
            answers.append({'type': 'refusal', 'name': rule.id})
        else:
            answers.append({'type': 'predicate', 'name': rule.id, 'probability': probabilities.get(rule.id, 0.05)})
    return Decision.model_validate({
        'answers': answers,
        'model': model,
        'usage': {
            'input_tokens': input_tokens,
            'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
            'output_tokens': 0,
            'output_tokens_details': {'reasoning_tokens': 0},
            'total_tokens': input_tokens,
        },
    })


class FakeDecisions:
    def __init__(self, decision):
        self.decision = decision
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.decision


class FakeClient:
    """Quacks like openai.OpenAI as far as request.send needs."""
    def __init__(self, decision=None):
        self.decisions = FakeDecisions(decision or make_decision())


def write_photo(path: Path, size=(700, 900)) -> Path:
    Image.new('RGB', size, (200, 200, 200)).save(path, format='JPEG', quality=90)
    return path


@pytest.fixture
def photo_file(tmp_path) -> Path:
    return write_photo(tmp_path / 'photo.jpg')
