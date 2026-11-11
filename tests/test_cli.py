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
