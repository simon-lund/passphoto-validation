"""OFIQ as a measurement backend.

OFIQ is the BSI's open-source implementation of ISO/IEC 29794-5 (face image quality), co-maintained
by eu-LISA: https://github.com/BSI-OFIQ/OFIQ-Project (MIT, models MIT under BSI's licence).

It is optional, because it isn't on PyPI and its models (about 430 MB) download from eu-LISA:

1. Build it: `scripts/build.sh --build-python` in the OFIQ checkout (needs conan 2.18.1 and cmake).
2. Install the wheel into this environment: `uv pip install <OFIQ>/install_x86_64_linux/Release/ofiq-*.whl`.
3. Point `OFIQ_CONFIG_DIR` at the checkout's `data/` folder (it holds `ofiq_config.jaxn` and `models/`).

Without it, every OFIQ check reports `skipped` with the reason.
"""
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CONFIG_FILE = 'ofiq_config.jaxn'

# Labels of OFIQ's face parsing map (BiSeNet trained on CelebAMask-HQ); 255 is outside the aligned face crop
PARSING_LABELS = {
    'background': 0, 'skin': 1, 'left_brow': 2, 'right_brow': 3, 'left_eye': 4, 'right_eye': 5,
    'glasses': 6, 'left_ear': 7, 'right_ear': 8, 'earring': 9, 'nose': 10, 'mouth': 11,
    'upper_lip': 12, 'lower_lip': 13, 'neck': 14, 'necklace': 15, 'clothing': 16, 'hair': 17, 'hat': 18,
}

# Indices into OFIQ's 98 landmarks (ADNet, WFLW layout)
LANDMARK_CHIN = 16
LANDMARK_RIGHT_EYE = range(60, 68)  # subject's right eye, on the left of the photo
LANDMARK_LEFT_EYE = range(68, 76)
LANDMARK_RIGHT_PUPIL = 96
LANDMARK_LEFT_PUPIL = 97


@dataclass(frozen=True)
class Measure:
    """One OFIQ quality component: the native value and the 0-100 quality score (higher is better)."""
    native: float
    quality: float
    assessed: bool  # False when OFIQ reported FailureToAssess for this component


@dataclass
class OfiqResult:
    """Everything OFIQ computed for one photo, in the photo's own pixel coordinates."""
    measures: Dict[str, Measure]
    faces: List[Dict]
    landmarks: Optional[np.ndarray]  # (98, 2) x/y, or None
    parsing: Optional[np.ndarray] = None  # face parsing labels (see PARSING_LABELS), same size as the photo
    occlusion: Optional[np.ndarray] = None  # 1 where the face is visible, 0 where something covers it
    extra: Dict = field(default_factory=dict)


class OfiqEngine:
    """A loaded OFIQ instance. Building one loads all models (a few seconds), so keep it around."""

    def __init__(self, config_dir: str, config_file: str = CONFIG_FILE):
        from ofiq import OFIQ
        from ofiq.ofiq_capi import QualityMeasure

        self._ofiq = OFIQ(config_dir=str(config_dir), config_file=config_file)
        self._names = {
            value: name for name, value in vars(QualityMeasure).items()
            if not name.startswith('_') and isinstance(value, int)
        }
        # OFIQ's C library is thread-safe since 1.1.0, but the wrapper shares one handle
        self._lock = threading.Lock()
        self.version: Tuple[int, int, int] = tuple(OFIQ.get_version())

    def assess(self, image_bgr: np.ndarray) -> OfiqResult:
        """Run every OFIQ measure on a BGR photo.

        OFIQ's C API takes BGR pixels, like OpenCV (its Python docs say RGB, but its own image reader
        returns BGR, and RGB input flips NaturalColour).
        """
        image = np.ascontiguousarray(image_bgr, dtype=np.uint8)
        with self._lock:
            try:
                ext = self._ofiq.vector_quality_ext(image)
            except ValueError:
                # The wrapper (1.2.1) dereferences the empty masks when OFIQ finds no face; the measures
                # alone still come back, all of them FailureToAssess
                ext = {'quality': self._ofiq.vector_quality(image), 'faces': [], 'landmarks': [],
                       'segmentationMask': None, 'occlusionMask': None}

        measures = {
            self._names.get(r['measure'], str(r['measure'])): Measure(
                native=float(r['rawScore']), quality=float(r['scalar']), assessed=r['code'] == 0,
            )
            for r in ext['quality']
        }
        landmarks = np.array(ext['landmarks'], dtype=np.float64) if ext['landmarks'] else None
        return OfiqResult(
            measures=measures,
            faces=ext['faces'],
            landmarks=landmarks,
            parsing=ext['segmentationMask'],
            occlusion=ext['occlusionMask'],
        )


def load_engine(config_dir: Optional[str], config_file: str = CONFIG_FILE) -> Tuple[Optional[OfiqEngine], Optional[str]]:
    """Load OFIQ, or return why it isn't available (shown as the reason of every skipped OFIQ check)."""
    if not config_dir:
        return None, 'OFIQ is not configured (set OFIQ_CONFIG_DIR).'
    if not (Path(config_dir) / config_file).is_file():
        return None, f'OFIQ config not found: {Path(config_dir) / config_file}.'
    try:
        engine = OfiqEngine(config_dir, config_file)
    except ImportError:
        return None, 'The ofiq package is not installed.'
    except Exception as exc:
        logger.exception('Could not load OFIQ')
        return None, f'OFIQ could not be loaded: {exc}'
    logger.info('Loaded OFIQ %s from %s', '.'.join(map(str, engine.version)), config_dir)
    return engine, None
