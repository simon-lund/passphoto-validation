"""Build the one Decisions request a photo needs, and send it.

One request per photo: the photo as a base64 JPEG, one developer message with the framing, the gate question
(how many faces) and one predicate per rule. The answer is the SDK's `Decision`.
"""
import base64
import copy
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageOps

from .rules import GATE_NAME, RULES

MODEL = 'gpt-6-luna'
DETAIL = 'high'

CONTEXT = (
    'The photo was submitted for a German ID card or passport. Judge it strictly against the official '
    'requirements for biometric photos, the Fotomustertafel of the German Federal Ministry of the Interior. '
    'Each question asks whether one specific defect is present in this photo; answer from what is visible.'
)

GATE_QUESTION = 'How many faces does the photo show that could be the subject of an ID photo?'
GATE_CHOICES = [
    {'value': 'none', 'description': 'No face, or only a face too small or too cut off to judge.'},
    {'value': 'one', 'description': "Exactly one person's face, large enough to judge."},
    {'value': 'several', 'description': 'More than one face.'},
]


@dataclass(frozen=True)
class Photo:
    """A photo as it is sent: its name, its size in pixels and the JPEG data URL."""
    name: str
    width: int
    height: int
    data_url: str


def load_photo(path: Path, max_side: Optional[int] = None) -> Photo:
    """Read a photo (applying its EXIF rotation), optionally shrink it, and encode it as a JPEG data URL.

    `width` and `height` are the original size: the rules about format and resolution refer to the photo as
    submitted, not to what was sent.
    """
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        width, height = image.size
        if max_side and max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.convert('RGB').save(buffer, format='JPEG', quality=92)
    data_url = 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    return Photo(name=Path(path).name, width=width, height=height, data_url=data_url)


def questions() -> List[Dict[str, Any]]:
    """The gate as a choice, then one predicate per rule, named by the rule id."""
    gate = {'type': 'choice', 'name': GATE_NAME, 'instructions': GATE_QUESTION, 'choices': GATE_CHOICES}
    return [gate] + [{'type': 'predicate', 'name': rule.id, 'instructions': rule.question} for rule in RULES]


def build_request(photo: Photo, model: str = MODEL, detail: str = DETAIL) -> Dict[str, Any]:
    """The keyword arguments for `client.decisions.create`."""
    return {
        'model': model,
        'input': [
            {'role': 'developer', 'content': [{'type': 'input_text', 'text': CONTEXT}]},
            {'role': 'user', 'content': [
                {'type': 'input_text', 'text': f'The photo is {photo.width}x{photo.height} pixels.'},
                {'type': 'input_image', 'image_url': photo.data_url, 'detail': detail},
            ]},
        ],
        'questions': questions(),
    }


def redacted(request: Dict[str, Any]) -> Dict[str, Any]:
    """The request with the image bytes replaced by their size, for printing and logging."""
    shown = copy.deepcopy(request)
    for message in shown['input']:
        for part in message.get('content', []):
            if isinstance(part, dict) and part.get('type') == 'input_image':
                part['image_url'] = f"<{len(part['image_url'])} characters of base64 JPEG>"
    return shown


def send(client, request: Dict[str, Any]):
    """Send the request; `client` is an `openai.OpenAI` or anything with the same `decisions.create`."""
    return client.decisions.create(**request)
