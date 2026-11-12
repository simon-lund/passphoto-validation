"""The CLI prints the report the way the HTML report and the review page do: the problems first."""
import pathlib

import pytest
from typer.testing import CliRunner

from passphoto_validation.cli import app

IMAGES = pathlib.Path(__file__).parent / 'test-images'
needs_ofiq = pytest.mark.skipif(not __import__('os').environ.get('OFIQ_CONFIG_DIR'), reason='OFIQ is not configured')


@needs_ofiq
def test_validate_prints_the_problems_with_their_rules():
    result = CliRunner().invoke(app, ['validate', str(IMAGES / '6.jpg')])
    assert result.exit_code == 1, result.output  # not valid
    assert 'problem(s):' in result.output
    assert 'Head is too small' in result.output
    assert 'Head covers 70-80% of the' in result.output  # the rule the small head breaks
    assert 'Crop to 35x45' in result.output


@needs_ofiq
def test_verbose_lists_every_check():
    result = CliRunner().invoke(app, ['validate', str(IMAGES / '6.jpg'), '-v'])
    assert 'Checks:' in result.output and 'colour_cast' in result.output and 'Measurements:' in result.output


# --- The thresholds file from the environment, and what the CLI says about it (no model is loaded for `thresholds`)

def test_the_cli_reads_the_thresholds_file_of_the_environment(tmp_path, monkeypatch):
    import tomllib
    file = tmp_path / 'thresholds.toml'
    file.write_text('[colour_cast]\nfail_above = 15\n')
    monkeypatch.setenv('THRESHOLDS_FILE', str(file))
    # An option that is not given must not override the environment
    result = CliRunner().invoke(app, ['thresholds'])
    assert result.exit_code == 0, result.output
    assert tomllib.loads(result.output)['colour_cast']['fail_above'] == 15
    # The option wins over the environment
    other = tmp_path / 'other.toml'
    other.write_text('[colour_cast]\nfail_above = 17\n')
    result = CliRunner().invoke(app, ['thresholds', '--thresholds', str(other)])
    assert tomllib.loads(result.output)['colour_cast']['fail_above'] == 17


def test_a_bad_thresholds_file_is_named_with_its_check(tmp_path):
    file = tmp_path / 'thresholds.toml'
    file.write_text('[colour_cast]\nfail_abov = 15\n')
    result = CliRunner().invoke(app, ['thresholds', '--thresholds', str(file)])
    assert result.exit_code == 2
    # Rich read "[colour_cast]" as markup and printed the message without it
    assert '[colour_cast] has no parameter fail_abov' in ' '.join(result.output.split())


def test_validate_says_which_rules_were_not_checked(tmp_path):
    file = tmp_path / 'thresholds.toml'
    file.write_text('[skin_smoothing]\nenabled = false\n\n[skin_colour_patches]\nenabled = false\n')
    result = CliRunner().invoke(app, ['validate', str(IMAGES / '9.jpg'), '--thresholds', str(file)])
    output = ' '.join(result.output.split())
    assert 'Thresholds file:' in output
    assert 'rules were not checked' in output and '1 rule(s) not checked' in output
    assert 'Every rule passes' not in output
