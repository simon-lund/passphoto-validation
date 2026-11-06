"""SigLIP as a second opinion on the eyes.

SigLIP SO400M (Google, Apache 2.0) is an image-text model: it scores how well an image matches a sentence.
Asked zero-shot whether the eye region shows reflections on glasses, or eyes looking away, it ranks real
photos better than the pixel checks (notebooks/real_photos.ipynb, section 4). Its checks only ever warn.

It is optional, because it needs PyTorch and a 3.3 GB model:

1. Install the extra: `uv sync --extra siglip --inexact` (`--inexact` keeps a hand-installed OFIQ).
2. Set `SIGLIP_ENABLED=true`. The model downloads from Hugging Face on first use (to `HF_HOME`,
   by default `~/.cache/huggingface`).

It looks at the eyes OFIQ found, so it needs OFIQ too. Without either, its checks report `skipped` with
the reason. The thresholds in the validator are calibrated for this model and these prompts only.
"""
import logging
import threading
from typing import Dict, Optional, Tuple

import numpy as np

from .ofiq import LANDMARK_LEFT_EYE, LANDMARK_RIGHT_EYE

logger = logging.getLogger(__name__)

MODEL = 'ViT-SO400M-14-SigLIP'
PRETRAINED = 'webli'

# question: (problem prompts, fine prompts); a score is logit(problem) - logit(fine), each over the
# averaged prompts, so higher means more likely a problem. Same prompts as scripts/clip_zero_shot.py.
QUESTIONS = {
    'glare': ([
        'eyeglasses with a bright glare reflection on the lenses',
        'glasses lenses reflecting light, with a white shine across the glass',
        'a reflection of a window or lamp visible on the eyeglass lenses',
    ], [
        'eyeglasses with clean clear lenses and no reflections',
        'glasses through which the eyes are clearly visible, without any glare',
        'clear eyeglass lenses without any reflection',
    ]),
    'gaze_aside': ([
        'a close-up of eyes looking to the side',
        'a close-up of eyes looking away from the camera',
        'a close-up of eyes looking up or down',
    ], [
        'a close-up of eyes looking straight into the camera',
        'a close-up of eyes making direct eye contact with the viewer',
        'a close-up of eyes gazing directly at the lens',
    ]),
}


def eye_box(landmarks: np.ndarray, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """The region SigLIP looks at: both eyes, 2.2 eye distances wide and 1.1 high, from OFIQ's landmarks.

    Returns (left, top, right, bottom) clipped to the photo, or None if that leaves too little.
    """
    right = landmarks[list(LANDMARK_RIGHT_EYE)].mean(axis=0)
    left = landmarks[list(LANDMARK_LEFT_EYE)].mean(axis=0)
    centre, distance = (right + left) / 2, float(np.linalg.norm(right - left))
    box = (max(0, int(centre[0] - 1.1 * distance)), max(0, int(centre[1] - 0.55 * distance)),
           min(width, int(centre[0] + 1.1 * distance)), min(height, int(centre[1] + 0.55 * distance)))
    if box[2] - box[0] <= 10 or box[3] - box[1] <= 10:
        return None
    return box


class SiglipEngine:
    """A loaded SigLIP model with the question prompts encoded. Loading takes a few seconds and 3.5 GB of
    memory, so keep it around."""

    def __init__(self, threads: int):
        import open_clip
        import torch

        torch.set_num_threads(threads)
        self._torch = torch
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
        self._model.eval()
        tokenizer = open_clip.get_tokenizer(MODEL)
        self._scale = self._model.logit_scale.exp().item()
        self._texts = {question: (self._encode_text(tokenizer, problem), self._encode_text(tokenizer, fine))
                       for question, (problem, fine) in QUESTIONS.items()}
        self._lock = threading.Lock()

    def _encode_text(self, tokenizer, prompts):
        with self._torch.no_grad():
            embeddings = self._model.encode_text(tokenizer(prompts))
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        mean = embeddings.mean(0)
        return mean / mean.norm()

    def scores(self, image_rgb) -> Dict[str, float]:
        """Score a PIL image (RGB) on every question."""
        with self._lock, self._torch.no_grad():
            embedding = self._model.encode_image(self._preprocess(image_rgb).unsqueeze(0))[0]
        embedding = embedding / embedding.norm()
        # SigLIP's logit bias cancels out in the difference
        return {question: self._scale * float(embedding @ problem) - self._scale * float(embedding @ fine)
                for question, (problem, fine) in self._texts.items()}


def load_engine(enabled: bool, threads: int = 4) -> Tuple[Optional[SiglipEngine], Optional[str]]:
    """Load SigLIP, or return why it isn't available (shown as the reason of every skipped SigLIP check)."""
    if not enabled:
        return None, 'SigLIP is not enabled (set SIGLIP_ENABLED=true).'
    try:
        engine = SiglipEngine(threads)
    except ImportError:
        return None, 'SigLIP is not installed (uv sync --extra siglip).'
    except Exception as exc:
        logger.exception('Could not load SigLIP')
        return None, f'SigLIP could not be loaded: {exc}'
    logger.info('Loaded SigLIP %s (%s)', MODEL, PRETRAINED)
    return engine, None
