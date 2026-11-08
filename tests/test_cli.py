"""The command line: nothing is sent without being asked, and the batch writes a resumable results file."""
import csv
import json

import pytest
from typer.testing import CliRunner

from passphoto_decisions import cli
from passphoto_decisions.rules import GATE_NAME, RULES

from conftest import FakeClient, make_decision, write_photo

runner = CliRunner()


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient(make_decision({'sharp': 0.9}))
    monkeypatch.setattr(cli, 'make_client', lambda: client)
    return client


@pytest.fixture
def no_client(monkeypatch):
    def refuse():
        raise AssertionError('a client was made')
    monkeypatch.setattr(cli, 'make_client', refuse)


def test_dry_run_prints_the_request_and_sends_nothing(photo_file, no_client):
    result = runner.invoke(cli.app, ['check', str(photo_file), '--dry-run'])
    assert result.exit_code == 0, result.output
    request = json.loads(result.output)
    assert len(request['questions']) == 1 + len(RULES)
    assert 'base64,' not in result.output and 'characters of base64 JPEG' in result.output


def test_check_prints_a_table_and_json(photo_file, fake_client):
    result = runner.invoke(cli.app, ['check', str(photo_file)])
    assert result.exit_code == 0, result.output
    assert 'not valid' in result.output and 'sharp' in result.output
    assert len(fake_client.decisions.requests) == 1

    result = runner.invoke(cli.app, ['check', str(photo_file), '--json'])
    report = json.loads(result.output)
    assert report['is_valid'] is False
    assert {rule['id'] for rule in report['rules']} == {rule.id for rule in RULES}


def test_thresholds_come_from_the_options(photo_file, fake_client):
    result = runner.invoke(cli.app, ['check', str(photo_file), '--json', '--fail-at', '0.95'])
    report = json.loads(result.output)
    assert next(rule for rule in report['rules'] if rule['id'] == 'sharp')['status'] == 'warn'
    assert report['thresholds'] == {'warn_at': 0.4, 'fail_at': 0.95}


def _write_set(folder, names):
    for name in names:
        write_photo(folder / name)
    with open(folder / 'manifest.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['file', 'identity', 'split', 'variant', 'broken_rule'])
        writer.writeheader()
        for name in names:
            writer.writerow({'file': name, 'identity': name[:3], 'split': 'calibrate', 'variant': 'compliant', 'broken_rule': ''})


def test_batch_asks_before_sending(tmp_path, no_client):
    _write_set(tmp_path, ['001_compliant.jpg'])
    result = runner.invoke(cli.app, ['batch', str(tmp_path)])
    assert result.exit_code == 2
    assert '1 of 1 photos' in result.output and '--yes' in result.output
    assert not (tmp_path / 'results_decisions.jsonl').exists()


def test_batch_writes_run_eval_rows_and_resumes(tmp_path, fake_client):
    _write_set(tmp_path, ['001_compliant.jpg', '002_compliant.jpg', '003_compliant.jpg'])
    result = runner.invoke(cli.app, ['batch', str(tmp_path), '--yes', '--limit', '2', '--workers', '1'])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in (tmp_path / 'results_decisions.jsonl').read_text().splitlines()]
    assert [row['file'] for row in rows] == ['001_compliant.jpg', '002_compliant.jpg']
    assert set(rows[0]) == {'file', 'identity', 'split', 'variant', 'broken_rule', 'checks', 'rules', 'usage'}
    assert rows[0]['rules']['sharp'] == 'fail' and rows[0]['checks'][GATE_NAME]['status'] == 'pass'
    assert len(fake_client.decisions.requests) == 2

    # The second run only sends what is missing
    result = runner.invoke(cli.app, ['batch', str(tmp_path), '--yes', '--workers', '1'])
    assert result.exit_code == 0, result.output
    assert '1 of 3 photos' in result.output
    rows = [json.loads(line) for line in (tmp_path / 'results_decisions.jsonl').read_text().splitlines()]
    assert [row['file'] for row in rows] == ['001_compliant.jpg', '002_compliant.jpg', '003_compliant.jpg']
    assert len(fake_client.decisions.requests) == 3


def test_a_failing_request_becomes_an_error_row(tmp_path, monkeypatch):
    class Broken:
        class decisions:  # noqa: N801 - mirrors the SDK's attribute
            @staticmethod
            def create(**kwargs):
                raise RuntimeError('boom')
    monkeypatch.setattr(cli, 'make_client', lambda: Broken())
    _write_set(tmp_path, ['001_compliant.jpg'])
    result = runner.invoke(cli.app, ['batch', str(tmp_path), '--yes', '--workers', '1'])
    assert result.exit_code == 0, result.output
    row = json.loads((tmp_path / 'results_decisions.jsonl').read_text())
    assert row['checks'][GATE_NAME] == {'status': 'error', 'reasons': ['RuntimeError: boom'], 'measurements': {}}
    assert '1 failed requests' in result.output


def test_failed_photos_are_sent_again_on_the_next_run(tmp_path, fake_client):
    _write_set(tmp_path, ['001_compliant.jpg', '002_compliant.jpg'])
    good = json.dumps({'file': '001_compliant.jpg', 'identity': '001', 'split': 'calibrate', 'variant': 'compliant',
                       'broken_rule': '', 'checks': {GATE_NAME: {'status': 'pass', 'reasons': [], 'measurements': {}}},
                       'rules': {}, 'usage': {'input_tokens': 7}})
    failed = json.dumps({'file': '002_compliant.jpg', 'identity': '002', 'split': 'calibrate', 'variant': 'compliant',
                         'broken_rule': '', 'checks': {GATE_NAME: {'status': 'error', 'reasons': ['RateLimitError: 429'], 'measurements': {}}},
                         'rules': {}, 'usage': {}})
    (tmp_path / 'results_decisions.jsonl').write_text(good + '\n' + failed + '\n')

    result = runner.invoke(cli.app, ['batch', str(tmp_path), '--yes', '--workers', '1'])
    assert result.exit_code == 0, result.output
    assert '1 of 2 photos' in result.output and '1 failed last time and are sent again' in result.output
    rows = [json.loads(line) for line in (tmp_path / 'results_decisions.jsonl').read_text().splitlines()]
    assert [row['file'] for row in rows] == ['001_compliant.jpg', '002_compliant.jpg']
    assert rows[0]['usage'] == {'input_tokens': 7}, 'the answered row is kept as it was'
    assert rows[1]['checks'][GATE_NAME]['status'] == 'pass'
    assert [request['input'][1]['content'][0]['text'] for request in fake_client.decisions.requests] == ['The photo is 700x900 pixels.']


def test_rate_limits_are_waited_out(tmp_path, monkeypatch):
    import httpx2 as httpx  # the SDK's HTTP library
    from openai import RateLimitError

    class Limited:
        calls = 0

        class decisions:  # noqa: N801 - mirrors the SDK's attribute
            @staticmethod
            def create(**kwargs):
                Limited.calls += 1
                if Limited.calls <= 2:
                    response = httpx.Response(429, request=httpx.Request('POST', 'https://api.openai.com/v1/decisions'))
                    raise RateLimitError('Rate limit reached', response=response, body=None)
                return make_decision()
    waits = []
    monkeypatch.setattr(cli, 'make_client', lambda: Limited())
    monkeypatch.setattr(cli.time, 'sleep', waits.append)
    _write_set(tmp_path, ['001_compliant.jpg'])
    result = runner.invoke(cli.app, ['batch', str(tmp_path), '--yes', '--workers', '1'])
    assert result.exit_code == 0, result.output
    assert Limited.calls == 3 and waits == [5, 10]
    row = json.loads((tmp_path / 'results_decisions.jsonl').read_text())
    assert row['checks'][GATE_NAME]['status'] == 'pass'
    assert '0 failed requests' in result.output


def test_an_api_error_on_check_is_one_line_not_a_traceback(photo_file, monkeypatch):
    from openai import OpenAIError

    class Refusing:
        class decisions:  # noqa: N801 - mirrors the SDK's attribute
            @staticmethod
            def create(**kwargs):
                raise OpenAIError('Error code: 429 - You have no credits remaining.')
    monkeypatch.setattr(cli, 'make_client', lambda: Refusing())
    result = runner.invoke(cli.app, ['check', str(photo_file)])
    assert result.exit_code == 1
    assert 'The API did not answer' in result.output and 'no credits remaining' in result.output
    assert 'Traceback' not in result.output


def test_rules_lists_the_catalogue():
    result = runner.invoke(cli.app, ['rules'])
    assert result.exit_code == 0
    assert 'no_retouching' in result.output


def test_the_key_comes_from_the_environment_or_a_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(Exception, match='OPENAI_API_KEY is not set'):
        cli.make_client()

    (tmp_path / '.env').write_text('OPENAI_API_KEY=sk-test-from-dotenv\n')
    client = cli.make_client()
    assert client.api_key == 'sk-test-from-dotenv'
    assert client.max_retries == 3
