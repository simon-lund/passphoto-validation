"""scripts/make_test_photos.py turns a folder of portraits into labelled test photos with a manifest."""
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts' / 'make_test_photos.py'


@pytest.fixture(scope='module')
def made(tmp_path_factory):
    source = tmp_path_factory.mktemp('portraits')
    for name in ('3.jpg', '9.jpg'):
        (source / name).write_bytes((REPO / 'tests' / 'test-images' / name).read_bytes())
    out = tmp_path_factory.mktemp('photos')
    run = subprocess.run([sys.executable, str(SCRIPT), '--source', str(source), '--out', str(out),
                          '--count', '5', '--seed', '3', '--variants', 'compliant,dark,roll_12'],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    with open(out / 'manifest.csv') as f:
        rows = list(csv.DictReader(f))
    return out, rows, run


def test_the_manifest_has_the_evaluation_set_shape(made):
    out, rows, _ = made
    assert len(rows) == 5
    assert list(rows[0]) == ['file', 'identity', 'split', 'variant', 'broken_rule']
    assert {row['split'] for row in rows} == {'test'}
    assert all((out / row['file']).is_file() for row in rows)


def test_variants_are_spread_evenly_and_labelled(made):
    _, rows, _ = made
    counts = {}
    for row in rows:
        counts[row['variant']] = counts.get(row['variant'], 0) + 1
    assert counts == {'compliant': 2, 'dark': 2, 'roll_12': 1}
    assert {row['broken_rule'] for row in rows if row['variant'] == 'compliant'} == {''}
    assert {row['broken_rule'] for row in rows if row['variant'] == 'dark'} == {'not_underexposed'}
    assert {row['broken_rule'] for row in rows if row['variant'] == 'roll_12'} == {'no_head_tilt'}
    assert len({(row['identity'], row['variant']) for row in rows}) == 5, 'no portrait gets the same variant twice'


def test_photos_are_35x45_crops(made):
    out, rows, _ = made
    for row in rows:
        image = cv2.imread(str(out / row['file']))
        assert image.shape[1] / image.shape[0] == pytest.approx(35 / 45, abs=0.01)


def test_unknown_variant_is_refused(tmp_path):
    run = subprocess.run([sys.executable, str(SCRIPT), '--source', str(REPO / 'tests' / 'test-images'),
                          '--out', str(tmp_path / 'out'), '--variants', 'compliant,bogus'], capture_output=True, text=True)
    assert run.returncode != 0
    assert 'bogus' in run.stderr
    assert not (tmp_path / 'out').exists()
