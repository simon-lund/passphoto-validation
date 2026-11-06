"""Data models for the passport photo validation process."""
from dataclasses import dataclass, asdict, field
from typing import TYPE_CHECKING, Dict, List, Optional, Literal

import numpy as np

if TYPE_CHECKING:
    from .ofiq import OfiqResult


@dataclass
class ValidationIssue:
    """Represents a single validation issue found in a photo."""
    category: Literal["error", "warning", "info"]
    message: str
    severity: int  # 1-10, 10 being most severe
    check: Optional[str] = None  # name of the check that raised it, set by the validator


class CheckSkipped(Exception):
    """Raised by a check that cannot judge this photo; the message is the reason shown to the user."""


CheckStatus = Literal["pass", "warn", "fail", "skipped", "error"]
"""Outcome of one check on one photo.

- pass: the check ran and found nothing.
- warn: the check ran and its worst finding is a warning.
- fail: the check ran and found at least one error.
- skipped: the check could not run on this photo (no face, too little background, ...).
- error: the check crashed; this is a pipeline bug, not a property of the photo.
"""


@dataclass
class CheckResult:
    """The outcome of a single check, including passes, so every check that exists is reported."""
    name: str
    status: CheckStatus
    reasons: List[str] = field(default_factory=list)
    measurements: Dict = field(default_factory=dict)


RuleStatus = Literal["pass", "warn", "fail", "skipped", "not_checked"]
"""Outcome of one photo rule (from the Fotomustertafel) on one photo.

- pass / warn / fail: worst outcome of the checks that cover the rule and ran.
- skipped: checks cover the rule, but none of them could run on this photo.
- not_checked: no check covers the rule yet.
"""


@dataclass
class RuleResult:
    """The outcome of one photo rule, derived from the checks that cover it."""
    id: str
    group: str
    title: str
    status: RuleStatus
    checks: List[str] = field(default_factory=list)
    coverage: Literal["full", "partial", "none"] = "none"
    reasons: List[str] = field(default_factory=list)
    note: Optional[str] = None


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
    ofiq: Optional["OfiqResult"] = None  # what OFIQ measured, if it ran
    ofiq_unavailable: Optional[str] = None  # why OFIQ did not run, shown by the skipped OFIQ checks
    siglip: Optional[Dict[str, float]] = None  # SigLIP's score per question on the eye region, if it ran
    siglip_unavailable: Optional[str] = None  # why SigLIP did not run, shown by the skipped SigLIP checks

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
    checks: List[CheckResult] = field(default_factory=list)
    rules: List[RuleResult] = field(default_factory=list)
    # Rules none of whose checks could run (their reasons are on the rule); a photo with any is not valid
    skipped_rules: List[str] = field(default_factory=list)

    def to_dict(self):
        """Convert report to a dictionary, suitable for JSON serialization."""
        report_dict = asdict(self)
        report_dict['summary'] = {
            'total_issues': len(self.issues),
            'errors': len([i for i in self.issues if i.category == 'error']),
            'warnings': len([i for i in self.issues if i.category == 'warning'])
        }
        return report_dict
