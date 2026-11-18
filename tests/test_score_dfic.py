"""scripts/score_dfic.py: DFIC's protocol, the map from checks to its requirements, and how unjudged photos count."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import score_dfic  # noqa: E402

EYES_OPEN = {'warn_below': 95, 'fail_below': 70}


def test_eer_follows_dfics_rule():
    assert score_dfic.eer(np.array([3.0, 2.0]), np.array([1.0, 0.0])) == 0  # separated
    assert score_dfic.eer(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1  # reversed
    assert score_dfic.eer(np.array([3.0, 2.0]), np.array([2.5, 1.0])) == 0.5
    assert math.isclose(score_dfic.eer(np.array([4.0, 3.0, 1.0]), np.array([2.0, 0.0, -1.0])), 1 / 3, abs_tol=0.002)
    assert math.isnan(score_dfic.eer(np.array([1.0]), np.array([])))


def test_badness_is_the_distance_past_the_fail_line():
    eyes_open = score_dfic.Member('ofiq_eyes_open', 'ofiq_eyes_open_quality')
    yaw = score_dfic.REQUIREMENTS['roll_pitch_yaw'][0]
    red = score_dfic.Member('red_eyes', 'red_eye_share')
    check = lambda status, **m: {'status': status, 'measurements': m}  # noqa: E731
    # Lower is worse: the line over the number; higher is worse: the number over the line
    assert score_dfic.badness(eyes_open, check('fail', ofiq_eyes_open_quality=35), EYES_OPEN) == 2
    assert score_dfic.badness(yaw, check('fail', head_yaw_degrees_ofiq=-12), {'warn_above': 5, 'fail_above': 10}) == 1.2
    # Without a fail level, the warn level is the line
    assert score_dfic.badness(red, check('warn', red_eye_share=0.6), {'warn_above': 0.3, 'fail_above': math.inf}) == 2
    # Passed without a number (no glasses worn): fully compliant; skipped: not judged; a fail needs its number
    glasses = score_dfic.Member('glasses_tint', 'lens_brightness_ratio')
    assert score_dfic.badness(glasses, check('pass', glasses_worn=False), {'fail_below': 0.45}) == 0
    assert score_dfic.badness(eyes_open, check('skipped'), EYES_OPEN) is None
    assert score_dfic.badness(eyes_open, None, EYES_OPEN) is None
    with pytest.raises(ValueError):
        score_dfic.badness(eyes_open, check('fail'), EYES_OPEN)
    with pytest.raises(ValueError):
        score_dfic.fail_line({'warn_above': math.inf, 'fail_above': math.inf})
    # The gate: one face passes, any other number fails
    gate = score_dfic.Member(score_dfic.GATE)
    assert score_dfic.badness(gate, check('pass'), {}) == 0
    assert score_dfic.badness(gate, check('fail'), {}) == 2


def test_verify_refuses_a_number_that_does_not_reproduce_its_check():
    good = {'p.jpg': {'ofiq_eyes_open': {'status': 'fail', 'measurements': {'ofiq_eyes_open_quality': 20}}}}
    score_dfic.verify(good, {'ofiq_eyes_open': EYES_OPEN})
    wrong = {'p.jpg': {'ofiq_eyes_open': {'status': 'fail', 'measurements': {'ofiq_eyes_open_quality': 100}}}}
    with pytest.raises(ValueError, match='do not reproduce'):
        score_dfic.verify(wrong, {'ofiq_eyes_open': EYES_OPEN})


def test_published_table_matches_the_papers_averages():
    assert list(score_dfic.PUBLISHED.index) == list(score_dfic.REQUIREMENTS)
    assert score_dfic.PUBLISHED.notna().sum().to_dict() == {'ofiq': 11, 'biogaze': 15, 'icaonet': 23}
    # Table VI's "Average" row
    for column, average in (('ofiq', 0.104), ('biogaze', 0.207), ('icaonet', 0.301)):
        assert math.isclose(score_dfic.PUBLISHED[column].mean(), average, abs_tol=0.0006)
    assert [r for r, m in score_dfic.REQUIREMENTS.items() if not m] == ['rotated_shoulders', 'frame_too_heavy', 'posterization']


def _dfic(tmp_path: Path, eyes_closed_cmp, eyes_closed_uncmp) -> Path:
    """A DFIC folder in its own layout: test subjects 1 and 2 (99 trains), every requirement's curated subsets."""
    partitions = tmp_path / 'partitions'
    (partitions / 'curated_test').mkdir(parents=True)
    (partitions / 'ids_balanced_test.txt').write_text('1\n2\n')
    labelled = [p for p, _ in eyes_closed_cmp + eyes_closed_uncmp if not p.endswith('unlabelled.JPG')]
    pd.DataFrame({'im_path': [p.removeprefix('Data_Torso/') for p in labelled if p.startswith('Data_Torso/')]}).to_csv(
        tmp_path / 'camera_image_labels.csv', index=False)
    pd.DataFrame({'im_path': [p.removeprefix('Artificial_Torso/') for p in labelled if p.startswith('Artificial_Torso/')]}).to_csv(
        tmp_path / 'artificial_image_labels.csv', index=False)
    for requirement in score_dfic.REQUIREMENTS:
        folder = partitions / 'curated_test' / requirement
        folder.mkdir()
        for side, photos in (('cmp', eyes_closed_cmp), ('uncmp', eyes_closed_uncmp)):
            photos = photos if requirement == 'eyes_closed' else []
            pd.DataFrame(photos, columns=['im_path', 'subj_id']).to_csv(folder / f'{side}_df.csv', index=False)
    return tmp_path


CMP = [('Data_Torso/1/Camera/a.JPG', 1), ('Data_Torso/1/Camera/b.JPG', 1), ('Data_Torso/2/Camera/c.JPG', 2),
       ('Data_Torso/99/Camera/train.JPG', 99),         # a training subject: left out
       ('Data_Torso/2/Camera/unlabelled.JPG', 2)]      # in no label table: left out
UNCMP = [('Artificial_Torso/1/Camera/d.JPG', 1), ('Data_Torso/2/Camera/e.JPG', 2)]


def _eyes(status, quality=None):
    return {'ofiq_eyes_open': {'status': status, 'measurements': {} if quality is None else {'ofiq_eyes_open_quality': quality}}}


RESULTS = {
    'Data_Torso/1/Camera/a.JPG': _eyes('pass', 100),
    'Data_Torso/1/Camera/b.JPG': _eyes('pass', 90),
    'Data_Torso/2/Camera/c.JPG': _eyes('skipped'),        # compliant, unjudged
    'Artificial_Torso/1/Camera/d.JPG': _eyes('fail', 20),
    'Data_Torso/2/Camera/e.JPG': _eyes('warn', 75),       # non-compliant, judged, only warned
}


def test_subsets_and_manifest_follow_dfics_selection(tmp_path):
    dfic = _dfic(tmp_path, CMP, UNCMP)
    eyes = score_dfic.subsets(dfic)['eyes_closed']
    assert list(eyes['cmp']['im_path']) == ['Data_Torso/1/Camera/a.JPG', 'Data_Torso/1/Camera/b.JPG', 'Data_Torso/2/Camera/c.JPG']
    assert len(eyes['uncmp']) == 2
    photos = score_dfic.manifest(dfic)
    assert len(photos) == 5 and photos['file'].str.startswith('preprocessed/').all()
    assert set(photos['identity']) == {'1', '2'}


def test_score_counts_unjudged_photos_both_ways(tmp_path):
    table = score_dfic.score(_dfic(tmp_path, CMP, UNCMP), RESULTS, {'ofiq_eyes_open': EYES_OPEN})
    row = table.loc['eyes_closed']
    assert (row['cmp'], row['uncmp']) == (3, 2)
    assert math.isclose(row['unjudged'], 1 / 5)
    # Judged photos separate perfectly; counted as non-compliant, the unjudged compliant photo costs a third
    assert row['eer_dfic'] == 0
    assert math.isclose(row['eer_strict'], 1 / 3, abs_tol=0.002)
    # At the shipped levels: one of the two non-compliant photos failed (the other warned), no compliant one
    assert (row['caught'], row['false_fail']) == (0.5, 0)
    assert row['ofiq'] == 0.009
    # A requirement without a check has its photo counts and nothing else
    assert pd.isna(table.loc['posterization', 'eer_dfic'])


def test_score_refuses_photos_without_a_result(tmp_path):
    results = {k: v for k, v in RESULTS.items() if not k.endswith('d.JPG')}
    with pytest.raises(KeyError, match='no result'):
        score_dfic.score(_dfic(tmp_path, CMP, UNCMP), results, {'ofiq_eyes_open': EYES_OPEN})
