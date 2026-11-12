"""scripts/score_labels.py: the rates of the real-photo table are shares of the photos a check judged."""
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import score_labels  # noqa: E402


def _frame(rows):
    # (file, check, status, glasses): one row per photo and check, joined with the photo's label
    return pd.DataFrame([dict(file=file, check=check, status=status, glasses=glasses, face_judgeable=True)
                         for file, check, status, glasses in rows])


def test_rates_are_shares_of_the_photos_the_check_judged():
    frame = _frame([
        ('a', 'ofiq_eyes_visible', 'fail', 'tinted'),
        ('b', 'ofiq_eyes_visible', 'warn', 'tinted'),
        ('c', 'ofiq_eyes_visible', 'pass', 'tinted'),
        ('d', 'ofiq_eyes_visible', 'skipped', 'tinted'),   # not judged: in not_run, in no rate
        ('e', 'ofiq_eyes_visible', 'pass', 'clear'),
        ('f', 'ofiq_eyes_visible', 'fail', 'none'),
    ])
    row = score_labels.score(frame, [('glasses', 'tinted', 'ofiq_eyes_visible', 'all')]).iloc[0]
    assert (row['positives'], row['negatives']) == (4, 2)
    assert math.isclose(row['hit'], 2 / 3) and math.isclose(row['hit_fail'], 1 / 3)
    assert math.isclose(row['false_alarm'], 1 / 2) and math.isclose(row['false_fail'], 1 / 2)
    assert math.isclose(row['not_run'], 1 / 6)


def test_checks_scored_together_count_as_one():
    # glasses_tint is not run once ofiq_eyes_visible has failed; together a photo is flagged when either flags
    frame = _frame([
        ('a', 'ofiq_eyes_visible', 'fail', 'tinted'), ('a', 'glasses_tint', 'skipped', 'tinted'),
        ('b', 'ofiq_eyes_visible', 'pass', 'tinted'), ('b', 'glasses_tint', 'warn', 'tinted'),
        ('c', 'ofiq_eyes_visible', 'pass', 'clear'), ('c', 'glasses_tint', 'pass', 'clear'),
    ])
    table = score_labels.score(frame, [('glasses', 'tinted', ('ofiq_eyes_visible', 'glasses_tint'), 'all'),
                                       ('glasses', 'tinted', 'glasses_tint', 'all'),
                                       ('glasses', 'tinted', ('ofiq_eyes_visible', 'glasses_tint'), 'clear_glasses')])
    together, alone, on_clear = (table.iloc[i] for i in range(3))
    assert together.name == ('glasses=tinted', 'ofiq_eyes_visible+glasses_tint')
    assert (together['hit'], together['hit_fail'], together['false_alarm'], together['not_run']) == (1.0, 0.5, 0.0, 0.0)
    assert alone['hit'] == 1.0 and math.isclose(alone['not_run'], 1 / 3)  # judged on one of the two tinted photos
    # No positive among the clear glasses: no hit rate, only false alarms
    assert math.isnan(on_clear['hit']) and on_clear['false_alarm'] == 0.0


def test_the_table_of_the_docs_has_a_pair_for_each_row():
    checks = {check for _, _, check, _ in score_labels.PAIRS + score_labels.BACKGROUND_PAIRS}
    assert {'ofiq_head_covering', ('ofiq_eyes_visible', 'glasses_tint'), 'glasses_glare', 'ofiq_eyes_open', 'gaze',
            'ofiq_mouth_closed', 'background_brightness',
            ('ofiq_background', 'background_consistency', 'background_pattern')} <= checks
