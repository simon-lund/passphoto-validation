"""Score checks against per-photo labels (e.g. vlm_labels.jsonl next to an evaluation set).

    uv run --extra test python scripts/score_labels.py --set data/openimages/eval

For each label, the photos where it says there is a problem are positives, the rest negatives. A check
counts as flagging a photo when it warns or fails. hit = flagged positives, false_alarm = flagged
negatives, not_run = skipped or error. Photos labelled face_judgeable=false are left out.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

# (label, value that means a problem, check, which photos to score it on)
PAIRS = [
    ('glasses', 'tinted', 'glasses_tint', 'all'),
    ('glasses', 'tinted', 'ofiq_eyes_visible', 'all'),
    ('glasses_glare', True, 'glasses_glare', 'with_glasses'),
    ('glasses_glare', True, 'glasses_glare', 'clear_glasses'),
    ('glasses_glare', True, 'siglip_glasses_glare', 'with_glasses'),
    ('glasses_glare', True, 'siglip_glasses_glare', 'clear_glasses'),
    ('glasses_glare', True, 'glare', 'with_glasses'),
    ('frame_covers_eyes', True, 'ofiq_eyes_visible', 'with_glasses'),
    ('hair_over_eyes', True, 'ofiq_eyes_visible', 'all'),
    ('eyes_open', False, 'ofiq_eyes_open', 'eyes_visible'),
    ('gaze_at_camera', False, 'gaze', 'eyes_visible'),
    ('gaze_at_camera', False, 'siglip_gaze', 'eyes_visible'),
    ('head_covering', 'any', 'ofiq_head_covering', 'all'),
    ('expression_neutral', False, 'ofiq_expression', 'all'),
    ('mouth_closed', False, 'ofiq_mouth_closed', 'all'),
    ('head_turned', True, 'ofiq_head_yaw', 'all'),
    ('head_turned', True, 'ofiq_head_pitch', 'all'),
    ('head_tilted', True, 'ofiq_head_roll', 'all'),
    ('red_eyes', True, 'red_eyes', 'eyes_visible'),
]


def load(eval_set: Path, results='results.jsonl', labels='vlm_labels.jsonl') -> pd.DataFrame:
    """One row per (photo, check), joined with the photo's labels."""
    rows = []
    with open(eval_set / results) as f:
        for line in f:
            photo = json.loads(line)
            for name, check in photo['checks'].items():
                rows.append(dict(file=photo['file'], variant=photo['variant'], split=photo['split'], check=name,
                                 status=check['status'], **check['measurements']))
    checks = pd.DataFrame(rows)
    labelled = pd.read_json(eval_set / labels, lines=True)
    return checks.merge(labelled, on='file', how='inner')


def _problem(frame: pd.DataFrame, label: str, value) -> pd.Series:
    if value == 'any':
        return frame[label] != 'none'
    return frame[label] == value


def score(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame['face_judgeable'] != False]  # noqa: E712 - missing means judgeable
    subsets = {
        'all': lambda f: f,
        'with_glasses': lambda f: f[f['glasses'] != 'none'],
        'clear_glasses': lambda f: f[f['glasses'] == 'clear'],
        'eyes_visible': lambda f: f[f['glasses'] != 'tinted'],
    }
    out = []
    for label, value, check, subset in PAIRS:
        rows = subsets[subset](frame[frame['check'] == check])
        if rows.empty:
            continue
        problem = _problem(rows, label, value)
        flagged = rows['status'].isin(['warn', 'fail'])
        ran = rows['status'].isin(['pass', 'warn', 'fail'])
        out.append(dict(
            label=f'{label}={value}', check=check, scored_on=subset,
            positives=int(problem.sum()), negatives=int((~problem).sum()),
            hit=(flagged & problem & ran).sum() / max(1, (problem & ran).sum()),
            hit_fail=(rows['status'].eq('fail') & problem).sum() / max(1, (problem & ran).sum()),
            false_alarm=(flagged & ~problem & ran).sum() / max(1, (~problem & ran).sum()),
            not_run=(~ran).mean(),
        ))
    return pd.DataFrame(out).set_index(['label', 'check'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', type=Path, required=True)
    parser.add_argument('--results', default='results.jsonl')
    parser.add_argument('--labels', default='vlm_labels.jsonl')
    args = parser.parse_args()
    pd.set_option('display.width', 200)
    table = score(load(args.set, results=args.results, labels=args.labels))
    print(table.to_string(formatters={c: '{:.0%}'.format for c in ('hit', 'hit_fail', 'false_alarm', 'not_run')}))


if __name__ == '__main__':
    main()
