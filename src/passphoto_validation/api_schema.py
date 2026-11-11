"""The shape of the API's JSON answers, for OpenAPI (/docs) and for the tests that pin the contract.

The models describe what check_api.py returns; they don't produce it. The report is built from the dataclasses
in core/models.py and sent as it is, so nothing here can reorder or drop a key. tests/test_check_api.py validates
real answers against these models with extra keys forbidden, so the two can't drift apart unnoticed.
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .core.models import CheckStatus, CropStatus, ProblemStatus, RuleStatus


class _Exact(BaseModel):
    """A model that matches its answer key for key: an unknown key is a contract change, not noise."""
    model_config = ConfigDict(extra='forbid')


class ErrorResponse(_Exact):
    """Every non-2xx answer. Branch on the status and `code`; `detail` is for people and may change."""
    code: str = Field(description='Stable snake_case code: invalid_request, image_empty, invalid_image, image_too_large, internal_error.')
    detail: str


class Issue(_Exact):
    category: Literal['error', 'warning', 'info']
    message: str
    severity: int = Field(description='1-10, 10 being most severe; the score deducts it.')
    check: Optional[str] = Field(description='The check that raised it.')


class Box(_Exact):
    """A rectangle in the photo's pixels."""
    x: int
    y: int
    width: int
    height: int


class SuggestedCrop(Box):
    """The older crop suggestion (3:4, face at 40%), kept for compatibility; use `crop`."""
    reason: str


class CheckResult(_Exact):
    name: str = Field(description='Part of the contract: rules refer to checks by name.')
    status: CheckStatus
    reasons: List[str]
    measurements: Dict[str, Any] = Field(description='What the check measured: numbers, booleans or short strings.')


class RuleResult(_Exact):
    id: str = Field(description='Part of the contract: ids are added, never renamed.')
    group: str
    title: str
    status: RuleStatus
    checks: List[str] = Field(description='The checks that cover the rule.')
    coverage: Literal['full', 'partial', 'none']
    reasons: List[str]
    note: Optional[str] = Field(description='What the checks leave out.')


class Problem(_Exact):
    """One cause behind the rules that did not pass: a check that did not pass, and the rules it decided."""
    check: str = Field(description='The check that caused it, as named in `checks`.')
    status: ProblemStatus = Field(description='The check\'s status: fail or warn when it found something, skipped or error when it could not judge the photo.')
    headline: str = Field(description='The first sentence of the first reason, for a one-line view.')
    reasons: List[str] = Field(description='The check\'s reasons.')
    rules: List[str] = Field(description='The rules whose status this check set: the rules it covers that are at its status (a rule two failing checks break is under both), or, for a skipped or crashed check, the ones it left skipped.')
    not_judged: List[str] = Field(description='Rules skipped because this check\'s failure stopped the checks that cover them (covered eyes stop the eye checks; no face stops every check).')


class Crop(_Exact):
    """The 35x45 crop for framing problems. Every key is present whatever the status; the ones that don't
    apply are null or empty."""
    status: CropStatus
    reason: Optional[str] = Field(description='Why no crop is possible (status not_possible).')
    x: Optional[int]
    y: Optional[int]
    width: Optional[int]
    height: Optional[int]
    is_valid: Optional[bool] = Field(description='Whether the cropped photo is valid (status proposed).')
    rules_after: Dict[str, RuleStatus] = Field(description='Each rule\'s status on the cropped photo.')
    fixed_rules: List[str] = Field(description='Rules that warn or fail now and pass on the cropped photo.')
    remaining_rules: List[str] = Field(description='Rules that still fail or are skipped on the cropped photo.')


class Summary(_Exact):
    total_issues: int
    errors: int
    warnings: int


class Group(_Exact):
    id: str
    title: str


class ImageSize(_Exact):
    width: int
    height: int


class CheckReport(_Exact):
    """The JSON answer of POST /v1/check."""
    is_valid: bool = Field(description='True only if no check found an error, none crashed and no rule is skipped.')
    overall_score: float = Field(description='0-100, 100 minus the severities of all issues.')
    issues: List[Issue]
    measurements: Dict[str, Any] = Field(description='All checks\' measurements in one dict.')
    bbox: Optional[Box] = Field(description='The detected face, or null.')
    suggested_crop: Optional[SuggestedCrop]
    checks: List[CheckResult] = Field(description='Every check, in pipeline order.')
    rules: List[RuleResult] = Field(description='Every rule of the Fotomustertafel, in catalogue order.')
    skipped_rules: List[str] = Field(description='Rules no check could judge; any of them makes the photo not valid.')
    crop: Optional[Crop] = Field(description='Null when cropping is turned off (AUTOCROP=false).')
    problems: List[Problem] = Field(description='The rules that did not pass, grouped by the check that caused it: fails, warnings, then what could not be judged. Empty when every rule passes.')
    summary: Summary
    groups: List[Group] = Field(description='The rule groups, in catalogue order.')
    image: ImageSize = Field(description='The photo as checked, after EXIF rotation.')
