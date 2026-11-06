"""Checks backed by OFIQ (ISO/IEC 29794-5 quality components).

OFIQ reports every component as a native value and a 0-100 quality score (higher is better). The
standard defines the measurement, not where to draw the line for an ID photo, so the thresholds
passed in from the validator are calibrated on labelled photos (see notebooks/evaluation.ipynb).
"""
import re

from ..models import CheckSkipped, ValidationContext, ValidationIssue
from ..ofiq import Measure

_POSE_MEASURES = {'yaw': 'HeadPoseYaw', 'pitch': 'HeadPosePitch', 'roll': 'HeadPoseRoll'}


def _snake(name: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def measure(context: ValidationContext, name: str) -> Measure:
    """The OFIQ component `name` for this photo, recorded in the measurements, or CheckSkipped."""
    if context.ofiq is None:
        raise CheckSkipped(context.ofiq_unavailable or 'OFIQ did not run on this photo.')
    result = context.ofiq.measures.get(name)
    if result is None or not result.assessed:
        raise CheckSkipped(f'OFIQ could not assess {name} on this photo.')
    context.measurements[f'ofiq_{_snake(name)}_native'] = result.native
    context.measurements[f'ofiq_{_snake(name)}_quality'] = result.quality
    return result


def check_quality(context: ValidationContext, measure_name: str, problem: str,
                  warn_below: float, fail_below: float):
    """Grade one OFIQ component by its 0-100 quality score.

    :param measure_name: OFIQ component, e.g. 'ExpressionNeutrality'.
    :param problem: What a low score means, shown to the user, e.g. 'Expression is not neutral'.
    :param warn_below: Quality below this warns.
    :param fail_below: Quality below this fails.
    """
    quality = measure(context, measure_name).quality
    if quality < fail_below:
        context.issues.append(ValidationIssue(
            category='error', message=f'{problem} (OFIQ {measure_name} {quality:.0f}/100).', severity=8))
    elif quality < warn_below:
        context.issues.append(ValidationIssue(
            category='warning', message=f'{problem}? (OFIQ {measure_name} {quality:.0f}/100).', severity=5))


def check_pose(context: ValidationContext, axis: str, warn_above: float, fail_above: float):
    """Grade one head pose angle from OFIQ (3DDFA-V2), in degrees.

    :param axis: 'yaw' (turned), 'pitch' (nodding) or 'roll' (tilted sideways).
    :param warn_above: Absolute angle above this warns.
    :param fail_above: Absolute angle above this fails.
    """
    angle = measure(context, _POSE_MEASURES[axis]).native
    context.measurements[f'head_{axis}_degrees_ofiq'] = angle
    what = {'yaw': 'turned', 'pitch': 'tilted up or down', 'roll': 'tilted sideways'}[axis]
    if abs(angle) > fail_above:
        context.issues.append(ValidationIssue(
            category='error', message=f'Head is {what} by {abs(angle):.1f}° (max {warn_above:.0f}°).', severity=9))
    elif abs(angle) > warn_above:
        context.issues.append(ValidationIssue(
            category='warning', message=f'Head is slightly {what} ({abs(angle):.1f}°).', severity=6))
