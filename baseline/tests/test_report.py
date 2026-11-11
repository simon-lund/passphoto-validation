"""From answers to statuses, and to the row an evaluation set records."""
import pytest

from passphoto_decisions import request as req
from passphoto_decisions.report import Thresholds, error_row, interpret, results_row
from passphoto_decisions.rules import GATE_NAME, RULES

from conftest import make_decision

MANIFEST_ROW = {'file': 'photo.jpg', 'identity': '001', 'split': 'calibrate', 'variant': 'compliant', 'broken_rule': ''}


@pytest.fixture
def photo(photo_file):
    return req.load_photo(photo_file)


@pytest.mark.parametrize('probability, status', [(0.0, 'pass'), (0.39, 'pass'), (0.4, 'warn'), (0.69, 'warn'), (0.7, 'fail'), (1.0, 'fail')])
def test_thresholds_at_their_boundaries(probability, status):
    assert Thresholds(0.4, 0.7).status(probability) == status


def test_a_clean_photo_is_valid(photo):
    report = interpret(make_decision(), photo)
    assert report.is_valid
    assert report.gate.status == 'pass' and report.gate.choice == 'one'
    assert [rule.id for rule in report.rules] == [rule.id for rule in RULES]
    assert all(rule.status == 'pass' and rule.reason is None for rule in report.rules)
    assert report.image == {'width': 700, 'height': 900}
    assert report.usage == {'input_tokens': 1500, 'cached_tokens': 0, 'output_tokens': 0}
    assert report.model == 'gpt-6-luna'


def test_probabilities_become_statuses_with_the_question_as_reason(photo):
    report = interpret(make_decision({'sharp': 0.9, 'contrast': 0.5}), photo)
    by_id = {rule.id: rule for rule in report.rules}
    assert by_id['sharp'].status == 'fail' and by_id['sharp'].probability == 0.9
    assert by_id['sharp'].reason.startswith('90%: Is the photo blurry')
    assert by_id['contrast'].status == 'warn'
    assert not report.is_valid


def test_a_warning_alone_keeps_the_photo_valid(photo):
    assert interpret(make_decision({'contrast': 0.5}), photo).is_valid


def test_a_refused_question_skips_its_rule_and_blocks_validity(photo):
    report = interpret(make_decision(refuse={'no_hat'}), photo)
    rule = next(rule for rule in report.rules if rule.id == 'no_hat')
    assert rule.status == 'skipped' and rule.probability is None
    assert rule.reason == 'The model declined this question.'
    assert not report.is_valid


@pytest.mark.parametrize('gate, reason', [('none', 'No face found.'), ('several', 'More than one face found.')])
def test_without_exactly_one_face_every_rule_is_skipped(photo, gate, reason):
    report = interpret(make_decision({'sharp': 0.9}, gate=gate), photo)
    assert report.gate.status == 'fail' and report.gate.reason == reason
    assert {rule.status for rule in report.rules} == {'skipped'}
    assert all(rule.reason == reason for rule in report.rules)
    assert not report.is_valid


def test_a_refused_face_count_skips_everything(photo):
    report = interpret(make_decision(gate=None), photo)
    assert report.gate.status == 'skipped' and report.gate.choice is None
    assert {rule.status for rule in report.rules} == {'skipped'}


def test_the_results_row_has_the_shape_run_eval_writes(photo):
    row = results_row(interpret(make_decision({'sharp': 0.9}), photo), MANIFEST_ROW)
    assert set(row) == set(MANIFEST_ROW) | {'checks', 'rules', 'usage'}
    assert row['identity'] == '001'
    assert set(row['checks']) == {GATE_NAME} | {rule.id for rule in RULES}
    assert row['checks'][GATE_NAME]['status'] == 'pass'
    assert row['checks'][GATE_NAME]['measurements'] == {'faces_none': 0.05, 'faces_one': 0.9, 'faces_several': 0.05}
    assert row['checks']['sharp'] == {'status': 'fail', 'reasons': [row['checks']['sharp']['reasons'][0]], 'measurements': {'probability': 0.9}}
    assert row['rules']['sharp'] == 'fail' and row['rules']['contrast'] == 'pass'
    assert row['usage']['input_tokens'] == 1500


def test_a_failed_request_is_a_row_too():
    row = error_row(MANIFEST_ROW, 'APIConnectionError: boom')
    assert row['checks'] == {GATE_NAME: {'status': 'error', 'reasons': ['APIConnectionError: boom'], 'measurements': {}}}
    assert set(row['rules']) == {rule.id for rule in RULES} and set(row['rules'].values()) == {'skipped'}
    assert row['usage'] == {}
