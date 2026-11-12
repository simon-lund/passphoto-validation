"""Compare the Decisions baseline with the pipeline on the London eval set, per rule, and sweep the thresholds.

    uv run --group dev python scripts/compare_london.py <set folder> [--split test]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SHOULD_PASS = ('compliant', 'roll_4', 'clear_glasses')
FLAG = ('warn', 'fail')

folder = Path(sys.argv[1])
split = sys.argv[sys.argv.index('--split') + 1] if '--split' in sys.argv else None


def load(name):
    rows = [json.loads(line) for line in open(folder / name)]
    return [r for r in rows if not split or r['split'] == split]


pipe = {r['file']: r for r in load('results.jsonl')}
base_all = load('results_decisions.jsonl')
base = {r['file']: r for r in base_all if r['checks']['face_detected']['status'] == 'pass'}
errors = sum(r['checks']['face_detected']['status'] == 'error' for r in base_all)
gate_fail = sum(r['checks']['face_detected']['status'] == 'fail' for r in base_all)
files = sorted(set(pipe) & set(base))
print(f'split={split or "all"}: {len(base_all)} baseline rows, {errors} unanswered (rate limit), {gate_fail} without exactly one face, '
      f'{len(files)} compared with the pipeline')

# One row per (photo, rule): the pipeline's status, the baseline's probability and status
rows = []
for f in files:
    p, b = pipe[f], base[f]
    for rule, status in p['rules'].items():
        bc = b['checks'].get(rule, {})
        rows.append(dict(file=f, variant=p['variant'], broken_rule=p['broken_rule'], rule=rule, pipe=status,
                         prob=bc.get('measurements', {}).get('probability', np.nan), base=b['rules'].get(rule)))
df = pd.DataFrame(rows)
good = df[df['variant'].isin(SHOULD_PASS)]

# A. per defect: what each system does on the rule the variant breaks, and how separable the baseline's probability is
out = []
for (rule, variant), bad in df[df['broken_rule'] != ''].groupby(['broken_rule', 'variant']):
    bad = bad[bad['rule'] == rule]
    ok = good[good['rule'] == rule]
    out.append(dict(
        rule=rule, defect=variant, n=len(bad),
        pipe_caught=(bad['pipe'] == 'fail').mean(), pipe_warned=(bad['pipe'] == 'warn').mean(),
        base_caught=(bad['base'] == 'fail').mean(), base_warned=(bad['base'] == 'warn').mean(),
        p_defect_median=bad['prob'].median(), p_good_p95=ok['prob'].quantile(0.95), p_good_max=ok['prob'].max(),
        pipe_false=(ok['pipe'].isin(FLAG)).mean(), base_false=(ok['base'].isin(FLAG)).mean(),
    ))
table = pd.DataFrame(out).set_index(['rule', 'defect'])
pd.set_option('display.width', 250)
pct = lambda v: f'{v:.0%}'
print('\nPer defect (test photos of that variant; false = warn or fail on photos that should pass):')
print(table.to_string(formatters={c: pct for c in table.columns if c != 'n'}))

# B. photo level: should-pass photos with no fail, and with neither fail nor warn
def photo_level(col):
    g = good.groupby('file')[col]
    return (g.apply(lambda s: not (s == 'fail').any()).mean(), g.apply(lambda s: not s.isin(FLAG).any()).mean())
print('\nPhotos that should pass (%d): no failing rule / neither fail nor warn' % good['file'].nunique())
print('  pipeline  %.0f%% / %.0f%%' % tuple(100 * v for v in photo_level('pipe')))
print('  baseline  %.0f%% / %.0f%%  (at warn 0.4, fail 0.7)' % tuple(100 * v for v in photo_level('base')))

# C. threshold sweep for the baseline: one global pair
print('\nBaseline threshold sweep (photos that should pass with a false fail on any rule | mean catch over the defects):')
defects = df[(df['broken_rule'] != '') & (df['rule'] == df['broken_rule'])]
for fail_at in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    false_fail = good.groupby('file')['prob'].apply(lambda s: (s >= fail_at).any()).mean()
    caught = defects.groupby('variant')['prob'].apply(lambda s: (s >= fail_at).mean()).mean()
    print(f'  fail_at {fail_at:.2f}: false fail {false_fail:.0%} | caught {caught:.0%}')
for warn_at in (0.2, 0.3, 0.4, 0.5):
    false_warn = good.groupby('file')['prob'].apply(lambda s: (s >= warn_at).any()).mean()
    flagged = defects.groupby('variant')['prob'].apply(lambda s: (s >= warn_at).mean()).mean()
    print(f'  warn_at {warn_at:.2f}: any flag on good photo {false_warn:.0%} | defect flagged {flagged:.0%}')

# E. per-rule thresholds chosen on the OTHER half (calibrate), applied here: fail at the 99th percentile of the
# probability on photos that should pass, warn at the 90th (each capped at 0.99 so a certain defect still fails)
other = 'calibrate' if split == 'test' else 'test'
cal_pipe = {r['file'] for r in [json.loads(l) for l in open(folder / 'results.jsonl')] if r['split'] == other}
cal_rows = []
for r in [json.loads(l) for l in open(folder / 'results_decisions.jsonl')]:
    if r['split'] == other and r['checks']['face_detected']['status'] == 'pass' and r['variant'] in SHOULD_PASS:
        for rule, check in r['checks'].items():
            if rule != 'face_detected':
                cal_rows.append(dict(rule=rule, prob=check['measurements'].get('probability', np.nan)))
cal = pd.DataFrame(cal_rows).groupby('rule')['prob']
# Strictly above the threshold, and never below 0.05 / 0.02: many good photos sit at exactly 0
per_rule = pd.DataFrame({'fail_at': cal.quantile(0.99).clip(lower=0.05, upper=0.99), 'warn_at': cal.quantile(0.90).clip(lower=0.02, upper=0.99)})
df['base2'] = [('fail' if p > per_rule.loc[r, 'fail_at'] else 'warn' if p > per_rule.loc[r, 'warn_at'] else 'pass')
               if not np.isnan(p) else None for r, p in zip(df['rule'], df['prob'])]
good = df[df['variant'].isin(SHOULD_PASS)]
print(f'\nWith per-rule thresholds chosen on the {other} half (fail = p99 of good photos, warn = p90):')
print('  thresholds above 0.5:', ', '.join(f'{r} {v:.2f}' for r, v in per_rule['fail_at'].items() if v > 0.5))
print('  good photos: no failing rule %.0f%% / neither fail nor warn %.0f%%' % tuple(100 * v for v in photo_level('base2')))
out2 = []
for (rule, variant), bad in df[df['broken_rule'] != ''].groupby(['broken_rule', 'variant']):
    bad = bad[bad['rule'] == rule]
    out2.append(dict(rule=rule, defect=variant, pipe_caught=(bad['pipe'] == 'fail').mean(), pipe_warned=(bad['pipe'] == 'warn').mean(),
                     base_caught=(bad['base2'] == 'fail').mean(), base_warned=(bad['base2'] == 'warn').mean(), fail_at=per_rule.loc[rule, 'fail_at']))
t2 = pd.DataFrame(out2).set_index(['rule', 'defect'])
print(t2.to_string(formatters={c: pct for c in t2.columns if c != 'fail_at'}, float_format=lambda v: f'{v:.2f}'))

# D. which rules fire falsely most on good photos, baseline vs pipeline
print('\nFalse alarms per rule on photos that should pass (share with warn or fail), top 10 each:')
fb = good.groupby('rule')['base'].apply(lambda s: s.isin(FLAG).mean()).sort_values(ascending=False).head(10)
fp = good.groupby('rule')['pipe'].apply(lambda s: s.isin(FLAG).mean()).sort_values(ascending=False).head(10)
print(pd.DataFrame({'baseline': fb}).to_string(float_format=pct))
print(pd.DataFrame({'pipeline': fp}).to_string(float_format=pct))

# F. the headline numbers of the paper, over all defect photos: caught = the rule the photo breaks fails,
# flagged = it warns or fails
defects = df[(df['broken_rule'] != '') & (df['rule'] == df['broken_rule'])]
print(f'\nDefect photos ({len(defects)}): caught / flagged')
print('  pipeline  %.0f%% / %.0f%%' % (100 * (defects['pipe'] == 'fail').mean(), 100 * defects['pipe'].isin(FLAG).mean()))
print('  baseline  %.0f%% / %.0f%%  (at warn 0.4, fail 0.7)' % (100 * (defects['base'] == 'fail').mean(), 100 * defects['base'].isin(FLAG).mean()))

# G. the per-rule reading the paper reports for the baseline: one fail level per rule, the 99th percentile of the
# probability on the good photos of the other half, never below 0.5; a probability at the level fails
floored = cal.quantile(0.99).clip(lower=0.5)
df['base3'] = [('fail' if p >= floored[r] else 'pass') if not np.isnan(p) else None for r, p in zip(df['rule'], df['prob'])]
good = df[df['variant'].isin(SHOULD_PASS)]
defects = df[(df['broken_rule'] != '') & (df['rule'] == df['broken_rule'])]
false_fails = good.groupby('rule')['base3'].apply(lambda s: (s == 'fail').mean()).sort_values(ascending=False).head(3)
print(f'\nBaseline with one fail level per rule (p99 of good photos on the {other} half, floored at 0.5):')
print('  good photos spared %.0f%%, defects caught %.0f%%' % (100 * photo_level('base3')[0], 100 * (defects['base3'] == 'fail').mean()))
print('  good photos failed, top rules: ' + ', '.join(f'{rule} {share:.0%}' for rule, share in false_fails.items()))
