"""Pin the rule catalogue and how rule statuses are derived from checks."""
from edutap.image_api.core import PassportPhotoValidator
from edutap.image_api.core.models import CheckResult
from edutap.image_api.core import rules
from edutap.image_api.core.rules import GROUPS, RULES, Rule, evaluate_rules


def _check_names():
    return PassportPhotoValidator().check_names()


def test_rule_ids_are_unique():
    ids = [rule.id for rule in RULES]
    assert len(ids) == len(set(ids))


def test_rules_use_known_groups_and_checks():
    names = set(_check_names())
    for rule in RULES:
        assert rule.group in GROUPS, rule.id
        assert set(rule.checks) <= names, rule.id
        assert (rule.coverage == 'none') == (not rule.checks), rule.id


def test_every_check_covers_a_rule():
    covered = {name for rule in RULES for name in rule.checks}
    assert set(_check_names()) == covered


def _result(rule_id, checks):
    return next(rule for rule in evaluate_rules(checks) if rule.id == rule_id)


def test_rule_takes_the_worst_status_of_checks_that_ran():
    checks = [
        CheckResult('face_centered', 'pass'),
        CheckResult('ofiq_head_roll', 'warn', ['tilted']),
        CheckResult('ofiq_head_pitch', 'skipped', ['no landmarks']),
        CheckResult('ofiq_head_yaw', 'fail', ['turned']),
    ]
    rule = _result('facing_camera', checks)
    assert rule.status == 'fail'
    assert rule.reasons == ['tilted', 'turned']


def test_rule_is_skipped_when_no_covering_check_ran():
    rule = _result('eyes_open', [CheckResult('ofiq_eyes_open', 'error', ['ValueError: boom'])])
    assert rule.status == 'skipped'
    assert rule.reasons == ['ValueError: boom']


def test_rule_is_skipped_when_a_covering_check_crashed_and_none_failed():
    # template_zones has two checks; one passing does not make up for the other crashing
    checks = [CheckResult('face_centered', 'error', ['ValueError: boom']), CheckResult('face_size', 'pass')]
    rule = _result('template_zones', checks)
    assert rule.status == 'skipped'
    assert rule.reasons == ['ValueError: boom']


def test_a_failed_check_outweighs_a_crashed_one():
    checks = [CheckResult('face_centered', 'error', ['ValueError: boom']), CheckResult('face_size', 'fail', ['too small'])]
    rule = _result('template_zones', checks)
    assert rule.status == 'fail'
    assert rule.reasons == ['too small']


def test_rule_without_checks_is_not_checked(monkeypatch):
    # Every rule has a check now, so the catalogue is swapped for one that doesn't
    monkeypatch.setattr(rules, 'RULES', [Rule('future_rule', 'retouching', 'Not covered yet')])
    rule = _result('future_rule', [CheckResult('face_centered', 'pass')])
    assert rule.status == 'not_checked'
    assert rule.coverage == 'none'
