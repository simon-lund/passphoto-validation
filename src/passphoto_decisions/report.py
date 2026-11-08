"""Turn a `Decision` into rule statuses, and into the row an evaluation set records.

Every rule's question asks whether its defect is present, so the API's probability is the probability that the
rule is broken. One threshold pair turns it into pass / warn / fail; the probability is kept next to the status
so the thresholds can be chosen again later without asking the API again.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

from .request import Photo
from .rules import GATE_NAME, RULES

Status = Literal['pass', 'warn', 'fail', 'skipped']


@dataclass(frozen=True)
class Thresholds:
    """A rule warns from `warn_at` and fails from `fail_at` (probability that it is broken). Placeholders until
    they are calibrated on labelled photos."""
    warn_at: float = 0.4
    fail_at: float = 0.7

    def status(self, probability: float) -> Status:
        if probability >= self.fail_at:
            return 'fail'
        if probability >= self.warn_at:
            return 'warn'
        return 'pass'


@dataclass
class RuleResult:
    id: str
    group: str
    title: str
    status: Status
    probability: Optional[float] = None  # that the rule is broken; None when the question wasn't answered
    reason: Optional[str] = None


@dataclass
class Gate:
    """The face count: `one` lets the rules be judged, anything else skips them all."""
    status: Literal['pass', 'fail', 'skipped']
    choice: Optional[str] = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    reason: Optional[str] = None


@dataclass
class Report:
    file: str
    is_valid: bool
    gate: Gate
    rules: List[RuleResult]
    model: str
    thresholds: Thresholds
    image: Dict[str, int]
    usage: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def interpret(decision, photo: Photo, thresholds: Thresholds = Thresholds()) -> Report:
    """Map the answers to the rules. A refusal skips its rule; a gate other than `one` skips every rule."""
    answers = {answer.name: answer for answer in decision.answers}

    gate_answer = answers.get(GATE_NAME)
    if gate_answer is None or gate_answer.type != 'choice':
        gate = Gate('skipped', reason='The model declined to count the faces.')
    else:
        probabilities = {str(p.value): p.probability for p in gate_answer.probabilities}
        choice = str(gate_answer.choice)
        if choice == 'one':
            gate = Gate('pass', choice, probabilities)
        else:
            reason = 'No face found.' if choice == 'none' else 'More than one face found.'
            gate = Gate('fail', choice, probabilities, reason)

    rules = []
    for rule in RULES:
        answer = answers.get(rule.id)
        if gate.status != 'pass':
            result = RuleResult(rule.id, rule.group, rule.title, 'skipped', reason=gate.reason)
        elif answer is None or answer.type != 'predicate':
            result = RuleResult(rule.id, rule.group, rule.title, 'skipped', reason='The model declined this question.')
        else:
            status = thresholds.status(answer.probability)
            reason = f'{answer.probability:.0%}: {rule.question}' if status != 'pass' else None
            result = RuleResult(rule.id, rule.group, rule.title, status, answer.probability, reason)
        rules.append(result)

    is_valid = gate.status == 'pass' and all(rule.status in ('pass', 'warn') for rule in rules)
    usage = decision.usage
    return Report(
        file=photo.name,
        is_valid=is_valid,
        gate=gate,
        rules=rules,
        model=decision.model,
        thresholds=thresholds,
        image={'width': photo.width, 'height': photo.height},
        usage={
            'input_tokens': usage.input_tokens,
            'cached_tokens': usage.input_tokens_details.cached_tokens,
            'output_tokens': usage.output_tokens,
        },
    )


def results_row(report: Report, manifest_row: Dict[str, str]) -> Dict[str, Any]:
    """One line of results.jsonl, in the shape passphoto-validation's scripts/run_eval.py writes.

    The gate is the check `face_detected`; each rule appears once more as a check of the same name whose
    measurement is the probability, so passphoto-validation's eval_report.py can read the file as it is.
    """
    checks = {
        GATE_NAME: {
            'status': report.gate.status,
            'reasons': [report.gate.reason] if report.gate.reason else [],
            'measurements': {f'faces_{value}': p for value, p in report.gate.probabilities.items()},
        },
    }
    for rule in report.rules:
        checks[rule.id] = {
            'status': rule.status,
            'reasons': [rule.reason] if rule.reason else [],
            'measurements': {'probability': rule.probability} if rule.probability is not None else {},
        }
    return {
        **manifest_row,
        'checks': checks,
        'rules': {rule.id: rule.status for rule in report.rules},
        'usage': report.usage,
    }


def error_row(manifest_row: Dict[str, str], error: str) -> Dict[str, Any]:
    """The line for a photo whose request failed: the gate crashed, every rule is skipped."""
    return {
        **manifest_row,
        'checks': {GATE_NAME: {'status': 'error', 'reasons': [error], 'measurements': {}}},
        'rules': {rule.id: 'skipped' for rule in RULES},
        'usage': {},
    }
