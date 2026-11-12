"""Pin the rule catalogue, how rule statuses are derived from checks, and how problems group them by cause."""
from passphoto_validation.core import PassportPhotoValidator
from passphoto_validation.core.config import ValidatorConfig
from passphoto_validation.core.models import CheckResult, Problem
from passphoto_validation.core import rules
from passphoto_validation.core.rules import GROUPS, RULES, Rule, evaluate_problems, evaluate_rules, headline
from passphoto_validation.core.validator import CHECK_DEPENDENCIES

# Every check name in report order, without loading the models
NAMES = ['face_detected'] + PassportPhotoValidator.registered_checks(ValidatorConfig()).names()
EYE_CHECKS = CHECK_DEPENDENCIES['ofiq_eyes_visible']


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


# --- Problems: the rules that did not pass, by the check that caused it

def _problems(dependencies=CHECK_DEPENDENCIES, **results):
    # Every check passes except the ones given as name=(status, reasons)
    checks = [CheckResult(name, *results.get(name, ('pass',))) for name in NAMES]
    return evaluate_problems(checks, evaluate_rules(checks), dependencies)


def test_a_clean_photo_has_no_problem():
    assert _problems() == []


def test_a_glare_photo_is_one_problem_with_two_rules():
    reasons = ['Reflections on the glasses. Tilt the glasses or change the lighting.']
    assert _problems(glasses_glare=('fail', reasons)) == [
        Problem('glasses_glare', 'fail', 'Reflections on the glasses.', reasons, ['no_reflections', 'glasses_no_reflections'])]


def test_a_small_head_is_one_problem_with_three_rules():
    reasons = ['Face is too small (39.3% of image). Should be 70-80%.']
    assert _problems(face_size=('fail', reasons)) == [
        Problem('face_size', 'fail', 'Face is too small (39.3% of image).', reasons, ['face_size', 'template_zones', 'face_height_mm'])]


def test_a_rule_two_failing_checks_break_is_under_both():
    problems = _problems(ofiq_head_yaw=('fail', ['turned']), ofiq_head_roll=('fail', ['tilted']))
    assert [(p.check, p.rules) for p in problems] == [
        ('ofiq_head_yaw', ['no_head_rotation', 'no_half_profile', 'facing_camera']),
        ('ofiq_head_roll', ['no_head_tilt', 'facing_camera']),
    ]


def test_a_warning_under_a_rule_another_check_fails_lists_no_rule():
    # background_plain fails on ofiq_background; the faint pattern is still reported, as a cause of nothing more
    problems = _problems(ofiq_background=('fail', ['not uniform']), background_pattern=('warn', ['faint pattern']))
    assert [(p.check, p.status, p.rules) for p in problems] == [
        ('ofiq_background', 'fail', ['background_no_objects', 'background_plain']),
        ('background_pattern', 'warn', []),
    ]


def test_fails_come_first_then_warnings_then_what_could_not_be_judged():
    problems = _problems(ofiq_head_covering=('skipped', ['OFIQ is not configured (set OFIQ_CONFIG_DIR).']),
                         contrast=('warn', ['low contrast']), face_centered=('error', ['ValueError: boom']),
                         exposure=('fail', ['too dark']))
    assert [(p.check, p.status) for p in problems] == [
        ('exposure', 'fail'), ('contrast', 'warn'), ('ofiq_head_covering', 'skipped'), ('face_centered', 'error')]


def test_a_check_that_could_not_run_is_a_problem_when_it_leaves_a_rule_skipped():
    reasons = ['OFIQ is not configured (set OFIQ_CONFIG_DIR).']
    assert _problems(ofiq_head_covering=('skipped', reasons)) == [
        Problem('ofiq_head_covering', 'skipped', 'OFIQ is not configured (set OFIQ_CONFIG_DIR).', reasons, ['no_hat'])]


def test_a_skipped_check_whose_rules_other_checks_judged_is_no_problem():
    # background_no_objects and background_plain are judged by the classic background checks
    assert _problems(ofiq_background=('skipped', ['OFIQ is not configured (set OFIQ_CONFIG_DIR).'])) == []


def test_a_crashed_check_is_always_a_problem():
    # It alone makes the photo not valid, so it is listed even when its rule failed on another check
    problems = _problems(ofiq_head_pitch=('error', ['ValueError: boom']), ofiq_head_yaw=('fail', ['turned']))
    assert [(p.check, p.status, p.rules) for p in problems] == [
        ('ofiq_head_yaw', 'fail', ['no_head_rotation', 'no_half_profile', 'facing_camera']),
        ('ofiq_head_pitch', 'error', []),
    ]
    assert _problems(face_centered=('error', ['ValueError: boom']))[0].rules == ['face_centered', 'template_zones', 'facing_camera']


def _covered_eyes(root_status='fail'):
    reason = 'Eyes are covered (hair, frame or tinted lenses) (OFIQ EyesVisible 0/100).'
    stopped = {name: ('skipped', [f'Not judged: {reason}']) for name in EYE_CHECKS}
    return dict(ofiq_eyes_visible=(root_status, [reason]), **stopped)


def test_a_failed_root_lists_the_rules_its_stopped_checks_left_unjudged():
    problems = _problems(**_covered_eyes())
    assert [p.check for p in problems] == ['ofiq_eyes_visible']
    assert problems[0].rules == ['eyes_not_covered', 'glasses_not_tinted', 'glasses_frame_clear']
    assert problems[0].not_judged == ['no_red_eyes', 'eyes_open', 'eyes_facing_camera', 'glasses_no_reflections']


def test_a_stopped_check_is_its_own_problem_when_no_root_failed():
    # The same statuses without the dependency table: each skipped check that left a rule skipped stands alone
    problems = _problems(dependencies={}, **_covered_eyes())
    assert [p.check for p in problems] == ['ofiq_eyes_visible', 'ofiq_eyes_open', 'gaze', 'red_eyes', 'glasses_glare',
                                           'siglip_glasses_glare', 'siglip_gaze']
    assert problems[0].not_judged == [] and problems[1].rules == ['eyes_open']


def test_the_face_gate_is_the_root_of_every_check():
    gate = {name: ('skipped', ['No face detected.']) for name in NAMES[1:]}
    problems = _problems(dependencies={'face_detected': NAMES[1:], **CHECK_DEPENDENCIES},
                         face_detected=('fail', ['No face detected']), **gate)
    assert len(problems) == 1 and problems[0].check == 'face_detected' and problems[0].headline == 'No face detected'
    assert problems[0].rules == ['face_visible']
    assert problems[0].not_judged == [rule.id for rule in RULES if rule.id != 'face_visible']


def test_problems_name_every_rule_that_did_not_pass_and_nothing_else():
    results = dict(_covered_eyes(), face_size=('fail', ['small']), contrast=('warn', ['flat']),
                   ofiq_head_covering=('skipped', ['no model']), face_centered=('error', ['boom']))
    checks = [CheckResult(name, *results.get(name, ('pass',))) for name in NAMES]
    rule_results = evaluate_rules(checks)
    named = {rule_id for problem in evaluate_problems(checks, rule_results, CHECK_DEPENDENCIES)
             for rule_id in problem.rules + problem.not_judged}
    assert named == {rule.id for rule in rule_results if rule.status in ('fail', 'warn', 'skipped')}


def test_headline_is_the_first_sentence_of_the_first_reason():
    assert headline(['Face is too small (39.3% of image). Should be 70-80%.']) == 'Face is too small (39.3% of image).'
    assert headline(['Head height (68.2%) is outside the optimal range (70-80%).']) == 'Head height (68.2%) is outside the optimal range (70-80%).'
    assert headline(['Expression is not neutral? (OFIQ ExpressionNeutrality 70/100).']) == 'Expression is not neutral?'
    assert headline(['', ' ValueError: boom ']) == 'ValueError: boom'
    assert headline([]) == ''


def test_dependencies_name_known_checks_and_do_not_chain():
    for root, dependents in CHECK_DEPENDENCIES.items():
        assert root in NAMES and set(dependents) <= set(NAMES), root
        assert len(set(dependents)) == len(dependents) and root not in dependents, root
        # A dependent is never a root itself, so a root always runs on its own
        assert not set(dependents) & set(CHECK_DEPENDENCIES), root
