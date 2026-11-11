"""Summarise results from run_eval.py, per rule and per check.

    uv run --extra test python scripts/eval_report.py --set ~/datasets/frl-london-eval [--split test]
        [--compare results_classic.jsonl]

Per rule: how often the rule a defect variant breaks ends as fail (caught) or warn, and how often
photos that should pass get a warn or fail on it (false alarms). Per check: the same, for the checks
aimed at each defect.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

SHOULD_PASS = ('compliant', 'roll_4', 'clear_glasses')
# Rules every photo built from a compliant crop should pass (the 3/4 and smiling photos are real photos,
# cropped the same way, so they keep the format)
FORMAT_RULES = ('photo_format', 'face_size', 'face_height_mm', 'face_centered', 'template_zones')

# Defect variant -> the checks aimed at it
TARGETS = {
    'smiling': ['ofiq_expression', 'ofiq_mouth_closed'],
    'yaw_left': ['ofiq_head_yaw'],
    'yaw_right': ['ofiq_head_yaw'],
    'roll_12': ['ofiq_head_roll'],
    'dark': ['exposure'],
    'bright': ['ofiq_overexposure'],
    'low_contrast': ['contrast'],
    'blur': ['face_sharpness'],
    'blur_2': ['face_sharpness'],
    'jpeg': ['compression'],
    'small_head': ['face_size'],
    'off_center': ['face_centered'],
    'red_eye': ['red_eyes'],
    # glasses_tint is stopped once ofiq_eyes_visible fails (CHECK_DEPENDENCIES), so sunglasses count through the root
    'sunglasses': ['ofiq_eyes_visible'],
    'glasses_glare': ['glasses_glare'],
    'bg_pattern': ['ofiq_background', 'background_consistency', 'background_pattern'],
    'bg_object': ['background_uniformity', 'ofiq_background'],
    'colour_cast': ['ofiq_natural_colour'],
    'side_shadow': ['ofiq_illumination'],
    'bg_shadow': ['background_shadow'],
    'bg_gradient': ['background_gradient'],
    'ink_stain': ['print_defects'],
    'smoothed': ['skin_smoothing'],
    'smoothed_all': ['skin_smoothing'],
    'face_stickers': ['skin_colour_patches'],
}


def load_photos(path: Path, split=None) -> list:
    with open(path) as f:
        photos = [json.loads(line) for line in f]
    return [p for p in photos if not split or p['split'] == split]


def checks_frame(photos) -> pd.DataFrame:
    """One row per (photo, check): status, reasons and the check's numeric measurements as columns."""
    return pd.DataFrame([
        dict(file=p['file'], identity=p['identity'], split=p['split'], variant=p['variant'], check=name,
             status=check['status'], reason='; '.join(check['reasons']), **check['measurements'])
        for p in photos for name, check in p['checks'].items()
    ])


def rules_frame(photos) -> pd.DataFrame:
    """One row per (photo, rule) with the rule status and the rule the photo's variant breaks."""
    return pd.DataFrame([
        dict(file=p['file'], identity=p['identity'], split=p['split'], variant=p['variant'],
             broken_rule=p['broken_rule'], rule=rule, status=status)
        for p in photos for rule, status in p['rules'].items()
    ])


def _share(statuses: pd.Series, wanted) -> float:
    return statuses.isin(wanted).mean() if len(statuses) else float('nan')


def rule_summary(rules: pd.DataFrame) -> pd.DataFrame:
    """Per broken rule: caught (fail), warned, and the false-alarm rate of that rule on photos that should pass."""
    good = rules[rules['variant'].isin(SHOULD_PASS)]
    out = []
    for (rule, variant), bad in rules[rules['broken_rule'] != ''].groupby(['broken_rule', 'variant']):
        bad = bad[bad['rule'] == rule]
        ok = good[good['rule'] == rule]['status']
        out.append(dict(rule=rule, defect=variant, caught=_share(bad['status'], ['fail']),
                        warned=_share(bad['status'], ['warn']), missed=_share(bad['status'], ['pass']),
                        not_judged=_share(bad['status'], ['skipped', 'not_checked']),
                        false_fail=_share(ok, ['fail']), false_warn=_share(ok, ['warn'])))
    return pd.DataFrame(out).set_index(['rule', 'defect'])


def false_alarms(rules: pd.DataFrame) -> pd.DataFrame:
    """Per rule, how often photos that should pass get a warn or a fail on it."""
    good = rules[rules['variant'].isin(SHOULD_PASS)]
    return (good.groupby('rule')['status']
            .agg(fail=lambda s: _share(s, ['fail']), warn=lambda s: _share(s, ['warn']))
            .sort_values(['fail', 'warn'], ascending=False))


def check_summary(checks: pd.DataFrame) -> pd.DataFrame:
    """Per check: false alarms on photos that should pass, and catch rate on its target defects."""
    good = checks[checks['variant'].isin(SHOULD_PASS)]
    out = []
    for variant, names in TARGETS.items():
        for name in names:
            bad = checks[(checks['variant'] == variant) & (checks['check'] == name)]['status']
            ok = good[good['check'] == name]['status']
            out.append(dict(defect=variant, check=name,
                            false_fail=_share(ok, ['fail']), false_warn=_share(ok, ['warn']),
                            caught=_share(bad, ['fail']), warned=_share(bad, ['warn']),
                            skipped=_share(bad, ['skipped', 'error']), n=len(bad)))
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', type=Path, required=True)
    parser.add_argument('--results', default='results.jsonl')
    parser.add_argument('--compare', help='A second results file (e.g. the old pipeline) to put next to it')
    parser.add_argument('--split', choices=['calibrate', 'test'])
    args = parser.parse_args()
    pd.set_option('display.width', 220)
    percent = lambda v: f'{v:.0%}'

    photos = load_photos(args.set / args.results, args.split)
    summary = rule_summary(rules_frame(photos))
    if args.compare:
        old = rule_summary(rules_frame(load_photos(args.set / args.compare, args.split)))
        summary = summary[['caught', 'warned', 'false_fail', 'false_warn']].join(
            old[['caught', 'warned', 'false_fail', 'false_warn']], rsuffix='_old')
    print(summary.to_string(float_format=percent))
    print()
    print(check_summary(checks_frame(photos)).to_string(float_format=percent))


if __name__ == '__main__':
    main()
