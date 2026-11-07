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
- skipped: the rule could not be judged: none of its checks could run on this photo, or one of them
  crashed and none failed. A photo with a skipped rule is never valid.
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


CropStatus = Literal["not_needed", "not_possible", "proposed"]


@dataclass
class Crop:
    """The 35x45 crop for a photo with framing problems (see PassportPhotoValidator._crop).

    The keys are the same whatever the status; the ones that don't apply are None or empty.

    - not_needed: no framing check warns or fails.
    - not_possible: no box fits inside the photo, or the head can't be measured; `reason` says why.
    - proposed: the box in the photo's pixels, whether the cropped photo is valid, each rule's status
      on it, the rules the crop fixes and the ones that still keep it from being valid.
    """
    status: CropStatus
    reason: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    is_valid: Optional[bool] = None
    rules_after: Dict[str, RuleStatus] = field(default_factory=dict)
    fixed_rules: List[str] = field(default_factory=list)
    remaining_rules: List[str] = field(default_factory=list)


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
    # The 35x45 crop for framing problems; None when cropping is turned off. Never changes is_valid.
    crop: Optional[Crop] = None

    def to_dict(self):
        """Convert report to a dictionary, suitable for JSON serialization."""
        report_dict = asdict(self)
        report_dict['summary'] = {
            'total_issues': len(self.issues),
            'errors': len([i for i in self.issues if i.category == 'error']),
            'warnings': len([i for i in self.issues if i.category == 'warning'])
        }
        return report_dict
