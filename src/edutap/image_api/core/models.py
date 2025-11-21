"""Data models for the passport photo validation process."""
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Literal

import numpy as np


@dataclass
class ValidationIssue:
    """Represents a single validation issue found in a photo."""
    category: Literal["error", "warning", "info"]
    message: str
    severity: int  # 1-10, 10 being most severe


@dataclass
class ValidationContext:
    """
    A container for the image and all data collected during a single validation run.
    This object is created for each validation and passed through the pipeline.
    """
    image_np: np.ndarray
    issues: List[ValidationIssue] = field(default_factory=list)
    measurements: Dict = field(default_factory=dict)
    face_data: Dict = field(default_factory=dict)
    segmentation_mask: Optional[np.ndarray] = None
    person_mask_binary: Optional[np.ndarray] = None

    @property
    def height(self) -> int:
        """The height of the image in pixels."""
        return self.image_np.shape[0]

    @property
    def width(self) -> int:
        """The width of the image in pixels."""
        return self.image_np.shape[1]


@dataclass
class PassportPhotoReport:
    """Represents the final report of a passport photo validation."""
    is_valid: bool
    overall_score: float  # 0-100, higher is better
    issues: List[ValidationIssue]
    measurements: Dict
    bbox: Optional[Dict] = None
    suggested_crop: Optional[Dict] = None

    def to_dict(self):
        """Convert report to a dictionary, suitable for JSON serialization."""
        report_dict = asdict(self)
        report_dict['summary'] = {
            'total_issues': len(self.issues),
            'errors': len([i for i in self.issues if i.category == 'error']),
            'warnings': len([i for i in self.issues if i.category == 'warning'])
        }
        return report_dict
