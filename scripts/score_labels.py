"""Score checks against per-photo labels (e.g. vlm_labels.jsonl next to an evaluation set).

    uv run --extra test python scripts/score_labels.py --set data/openimages/eval

For each label, the photos where it says there is a problem are positives, the rest negatives. A check
counts as flagging a photo when it warns or fails. Every rate is a share of the photos the check judged
(it ended as pass, warn or fail): hit = flagged positives, hit_fail = failed positives, false_alarm =
flagged negatives, false_fail = failed negatives. not_run is the share it could not judge (skipped or
error), counted apart. Photos labelled face_judgeable=false (or background_judgeable=false) are left out.

The table in PASSPORT_VALIDATOR.md ("Real photos with glasses, hats and closed eyes") is this output: the
face labels first, then the background labels (vlm_background_labels.jsonl) when the set has them.

A check that depends on another (validator.CHECK_DEPENDENCIES) is not run once that one has failed, and
shows as not_run on those photos. Where two checks judge the same defect they are scored together as one
row, `a+b`: flagged when either flags, run when either ran.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

# (label, value that means a problem, check or checks scored together, which photos to score it on)
PAIRS = [
    # Tinted lenses fail ofiq_eyes_visible, and glasses_tint is not run once that has failed, so on its own it
    # would be scored only on the tinted glasses the first check let through
    ('glasses', 'tinted', ('ofiq_eyes_visible', 'glasses_tint'), 'all'),
    # The same pair on the photos that should pass it, by whether glasses are worn: only false alarms there
    ('glasses', 'tinted', ('ofiq_eyes_visible', 'glasses_tint'), 'clear_glasses'),
    ('glasses', 'tinted', ('ofiq_eyes_visible', 'glasses_tint'), 'no_glasses'),
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

# The second set of labels, on the background: plain, patterned or objects, and light, medium or dark
BACKGROUND_LABELS = 'vlm_background_labels.jsonl'
BACKGROUND_PAIRS = [
    # A dark background against a light one; the medium tones are neither
    ('background_tone', 'dark', 'background_brightness', 'dark_or_light'),
    # The rule "background is plain, no pattern": the worst of its three checks, patterned against plain
    ('background', 'patterned', ('ofiq_background', 'background_consistency', 'background_pattern'), 'patterned_or_plain'),
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


def _together(frame: pd.DataFrame, checks) -> pd.DataFrame:
    """One row per photo for several checks scored as one: the worst status among those that ran."""
    rows = frame[frame['check'].isin(checks)]
    rank = rows['status'].map({'fail': 0, 'warn': 1, 'pass': 2}).fillna(3)  # a check that did not run comes last
    return rows.assign(rank=rank).sort_values('rank', kind='stable').drop_duplicates('file').drop(columns='rank')


def score(frame: pd.DataFrame, pairs=PAIRS) -> pd.DataFrame:
    for column in ('face_judgeable', 'background_judgeable'):
        if column in frame:
            frame = frame[frame[column] != False]  # noqa: E712 - missing means judgeable
    subsets = {
        'all': lambda f: f,
        'with_glasses': lambda f: f[f['glasses'] != 'none'],
        'clear_glasses': lambda f: f[f['glasses'] == 'clear'],
        'no_glasses': lambda f: f[f['glasses'] == 'none'],
        'eyes_visible': lambda f: f[f['glasses'] != 'tinted'],
        'dark_or_light': lambda f: f[f['background_tone'].isin(['dark', 'light'])],
        'patterned_or_plain': lambda f: f[f['background'].isin(['patterned', 'plain'])],
    }
    out = []
    for label, value, check, subset in pairs:
        if isinstance(check, tuple):
            rows, check = subsets[subset](_together(frame, check)), '+'.join(check)
        else:
            rows = subsets[subset](frame[frame['check'] == check])
        if rows.empty:
            continue
        problem = _problem(rows, label, value)
        flagged = rows['status'].isin(['warn', 'fail'])
        failed = rows['status'].eq('fail')
        ran = rows['status'].isin(['pass', 'warn', 'fail'])

        def share(mask, of):
            # Of the photos the check judged; no such photo, no rate
            return (mask & of & ran).sum() / (of & ran).sum() if (of & ran).any() else float('nan')

        out.append(dict(
            label=f'{label}={value}', check=check, scored_on=subset,
            positives=int(problem.sum()), negatives=int((~problem).sum()),
            hit=share(flagged, problem), hit_fail=share(failed, problem),
            false_alarm=share(flagged, ~problem), false_fail=share(failed, ~problem),
            not_run=(~ran).mean(),
        ))
    return pd.DataFrame(out).set_index(['label', 'check'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', type=Path, required=True)
    parser.add_argument('--results', default='results.jsonl')
    parser.add_argument('--labels', default='vlm_labels.jsonl')
    args = parser.parse_args()
    pd.set_option('display.width', 220)

    def show(table):
        percent = lambda v: '-' if pd.isna(v) else f'{v:.0%}'  # noqa: E731
        print(table.to_string(na_rep='-', formatters={c: percent for c in ('hit', 'hit_fail', 'false_alarm', 'false_fail', 'not_run')}))

    show(score(load(args.set, results=args.results, labels=args.labels)))
    if (args.set / BACKGROUND_LABELS).exists():
        print()
        show(score(load(args.set, results=args.results, labels=BACKGROUND_LABELS), BACKGROUND_PAIRS))


if __name__ == '__main__':
    main()
