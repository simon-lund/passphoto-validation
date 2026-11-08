"""The request: every rule asked once, the photo sent once, nothing else."""
import base64
import io

from PIL import Image

from passphoto_decisions import request as req
from passphoto_decisions.rules import GATE_NAME, RULES


def test_questions_ask_the_gate_then_every_rule_once():
    questions = req.questions()
    names = [question['name'] for question in questions]
    assert names == [GATE_NAME] + [rule.id for rule in RULES]
    assert len(set(names)) == len(names)
    assert questions[0]['type'] == 'choice'
    assert [choice['value'] for choice in questions[0]['choices']] == ['none', 'one', 'several']
    assert all(question['type'] == 'predicate' for question in questions[1:])
    assert all(question['instructions'].endswith('?') for question in questions)


def test_request_carries_the_framing_the_size_and_the_photo_once(photo_file):
    photo = req.load_photo(photo_file)
    request = req.build_request(photo)

    assert request['model'] == req.MODEL
    developer, user = request['input']
    assert developer['role'] == 'developer' and developer['content'][0]['text'] == req.CONTEXT
    assert user['role'] == 'user'
    texts = [part for part in user['content'] if part['type'] == 'input_text']
    images = [part for part in user['content'] if part['type'] == 'input_image']
    assert len(texts) == 1 and '700x900' in texts[0]['text']
    assert len(images) == 1 and images[0]['image_url'].startswith('data:image/jpeg;base64,')
    assert images[0]['detail'] == 'high'
    assert request['questions'] == req.questions()


def _decode(photo: req.Photo) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(photo.data_url.split(',', 1)[1])))


def test_the_photo_is_shrunk_only_when_asked_and_keeps_its_original_size(photo_file):
    full = req.load_photo(photo_file)
    assert (full.width, full.height) == (700, 900) and _decode(full).size == (700, 900)

    small = req.load_photo(photo_file, max_side=300)
    assert (small.width, small.height) == (700, 900), 'the format rules refer to the photo as submitted'
    assert _decode(small).size == (233, 300)


def test_redacted_request_has_no_image_bytes(photo_file):
    request = req.build_request(req.load_photo(photo_file))
    shown = req.redacted(request)
    image = shown['input'][1]['content'][1]['image_url']
    assert image.startswith('<') and 'characters of base64' in image
    # the original is untouched
    assert request['input'][1]['content'][1]['image_url'].startswith('data:')
