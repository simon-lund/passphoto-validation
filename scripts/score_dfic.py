"""Score the pipeline on DFIC, the ISO/ICAO compliance dataset of the University of Coimbra, with DFIC's protocol.

    uv run --extra test python scripts/score_dfic.py --set data/dfic --manifest   # writes data/dfic/manifest.csv
    uv run python scripts/run_eval.py --set data/dfic                              # writes data/dfic/results.jsonl
    uv run --extra test python scripts/score_dfic.py --set data/dfic               # the table

DFIC (Gonçalves, Nunes, Guerra and Marcos 2026, arXiv:2602.10985, github.com/visteam-isr-uc/DFIC) labels 26
requirements of ISO/IEC 19794-5 and ICAO 9303 on real photos of 1,016 people, plus generated defects. Its
licence allows non-commercial research only and no redistribution: like the London Set it lives in data/dfic/,
as the request form delivers it (partitions/, preprocessed/, camera_image_labels.csv,
artificial_image_labels.csv), and nothing made from it enters git.

The protocol is DFIC's own (method/evaluate_model_DFIC.py there): per requirement, the curated test subsets
(partitions/curated_test/<requirement>/cmp_df.csv and uncmp_df.csv), restricted to the test subjects
(partitions/ids_balanced_test.txt) and to the labelled photos, are the compliant and the non-compliant photos;
a compliance score per photo gives the equal error rate (EER, their rule: a 0.001 grid of false accepts).

The pipeline has no compliance score of its own, so REQUIREMENTS names, per requirement, the checks that judge
it and the number each one grades. A requirement's score is the worst of its checks, each as how far its number
lies past the check's fail line (1 is on the line, less passes): compliance = -max(badness). The fail line is the
check's fail level, or its warn level for a warn-only check. A check that passed without a number (the glasses
checks when no glasses are worn) counts as 0; a skipped check, or one that crashed, did not judge the photo. A
photo none of a requirement's checks judged is unjudged for that requirement.

Unjudged photos are counted two ways. eer_dfic leaves them out, as DFIC's script does with a missing score;
eer_strict counts them as non-compliant, as the pipeline does (a rule it could not judge is never valid);
unjudged is their share. caught and false_fail are the pipeline's own verdict at its shipped levels, as shares of
the photos it judged: caught = non-compliant photos one of the checks failed, false_fail = compliant ones.

Beside them, the EERs the paper publishes in Table VI for three methods not trained on DFIC: OFIQ (11
requirements), BioGaze (15) and ICAONet (23). The averages are taken over the rows of each set only where a
column has every row. Three requirements have no check here: rotated_shoulders, frame_too_heavy, posterization.

The map was fixed before any DFIC photo was scored (this file's history says when). A change after the first
run must be tuned on the training subjects (partitions/ids_all_train.txt), never on the test subjects.
"""
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

GATE = 'face_detected'  # the face gate: a photo without exactly one face stops there, every check skipped


@dataclass(frozen=True)
class Member:
    """A check that judges a requirement, and the number it grades: a measurement's name, or how to compute it."""
    check: str
    value: Union[str, Callable[[dict], Optional[float]], None] = None


def _absolute(name: str) -> Callable[[dict], Optional[float]]:
    return lambda m: abs(m[name]) if name in m else None


def _gaze(m: dict) -> Optional[float]:
    # Gaze grades the larger of the two offsets
    if 'gaze_horizontal' not in m:
        return None
    return max(abs(m['gaze_horizontal']), abs(m['gaze_vertical']))


def _shadow(m: dict) -> Optional[float]:
    # BackgroundShadow grades the deeper of the band next to the person and the rim of the person mask
    if 'background_shadow_depth' not in m:
        return None
    return max(m['background_shadow_depth'], m['background_shadow_rim_depth'])


def _ofiq(check: str, measure: str) -> Member:
    return Member(check, f'ofiq_{measure}_quality')


EYES_VISIBLE = _ofiq('ofiq_eyes_visible', 'eyes_visible')
OVEREXPOSURE = _ofiq('ofiq_overexposure', 'over_exposure_prevention')

# DFIC's 26 requirements in its order (its names), each with the checks that judge it here
REQUIREMENTS: Dict[str, Tuple[Member, ...]] = {
    'eyes_closed': (_ofiq('ofiq_eyes_open', 'eyes_open'),),
    'non_neutral_expression': (_ofiq('ofiq_expression', 'expression_neutrality'),),
    'mouth_open': (_ofiq('ofiq_mouth_closed', 'mouth_closed'),),
    'rotated_shoulders': (),
    'roll_pitch_yaw': (Member('ofiq_head_yaw', _absolute('head_yaw_degrees_ofiq')),
                       Member('ofiq_head_pitch', _absolute('head_pitch_degrees_ofiq')),
                       Member('ofiq_head_roll', _absolute('head_roll_degrees_ofiq'))),
    'looking_away': (Member('gaze', _gaze),),
    'hair_across_eyes': (EYES_VISIBLE,),
    'head_coverings': (_ofiq('ofiq_head_covering', 'no_head_coverings'),),
    'veil_over_face': (_ofiq('ofiq_face_occlusion', 'face_occlusion_prevention'),
                       _ofiq('ofiq_mouth_occlusion', 'mouth_occlusion_prevention')),
    # Only other faces: the gate stops at any number of faces but one; objects have no check
    'other_faces_objects': (Member(GATE),),
    # Tinted lenses mostly fail ofiq_eyes_visible, and glasses_tint is not run once that has failed
    'dark_tinted_lenses': (EYES_VISIBLE, Member('glasses_tint', 'lens_brightness_ratio')),
    'frame_covering_eyes': (EYES_VISIBLE,),
    'flash_reflection_lenses': (Member('glasses_glare', 'lens_glare_share'),),
    'frame_too_heavy': (),
    'shadows_behind_head': (Member('background_shadow', _shadow),),
    'shadows_across_face': (_ofiq('ofiq_illumination', 'illumination_uniformity'),),
    'flash_reflection_skin': (Member('glare', 'eye_reflection_ratio'), OVEREXPOSURE),
    'unnatural_skin_tone': (_ofiq('ofiq_natural_colour', 'natural_colour'), Member('colour_cast', 'colour_cast')),
    'red_eyes': (Member('red_eyes', 'red_eye_share'),),
    'too_dark_light': (Member('exposure', 'highlight_lightness'), OVEREXPOSURE),
    'blurred': (Member('face_sharpness', 'face_sharpness'),),
    # The checks of the two background rules (uniform; plain, no pattern)
    'varied_background': (_ofiq('ofiq_background', 'background_uniformity'),
                          Member('background_uniformity', 'background_within_rect_variance'),
                          Member('background_consistency', 'background_color_coherence_cov'),
                          Member('background_pattern', 'background_median_rect_variance')),
    'pixelation': (Member('compression', 'jpeg_blockiness'),),
    'washed_out': (Member('contrast', 'contrast'),),
    # Marks on the background only: a mark on the face has no check
    'ink_marked_creased': (Member('print_defects', 'background_marks'),),
    'posterization': (),
}

# Table VI of the paper: EER on the DFIC test subjects, for the methods not trained on DFIC (None: no value)
PUBLISHED = pd.DataFrame.from_dict({
    'eyes_closed': (0.009, 0.065, 0.094),
    'non_neutral_expression': (0.064, 0.224, None),
    'mouth_open': (0.019, None, 0.315),
    'rotated_shoulders': (None, 0.202, None),
    'roll_pitch_yaw': (0.042, 0.229, 0.210),
    'looking_away': (None, 0.466, 0.293),
    'hair_across_eyes': (None, None, 0.397),
    'head_coverings': (0.089, 0.102, 0.227),
    'veil_over_face': (None, None, 0.349),
    'other_faces_objects': (None, None, 0.520),
    'dark_tinted_lenses': (0.054, 0.175, 0.104),
    'frame_covering_eyes': (None, None, 0.241),
    'flash_reflection_lenses': (None, None, 0.140),
    'frame_too_heavy': (None, None, 0.248),
    'shadows_behind_head': (None, None, 0.335),
    'shadows_across_face': (0.237, 0.305, 0.410),
    'flash_reflection_skin': (None, None, 0.116),
    'unnatural_skin_tone': (0.033, 0.280, 0.358),
    'red_eyes': (None, None, 0.427),
    'too_dark_light': (0.129, 0.144, 0.121),
    'blurred': (0.243, 0.133, 0.292),
    'varied_background': (0.227, 0.120, 0.317),
    'pixelation': (None, 0.214, 0.445),
    'washed_out': (None, 0.279, 0.466),
    'ink_marked_creased': (None, None, 0.492),
    'posterization': (None, 0.172, None),
}, orient='index', columns=['ofiq', 'biogaze', 'icaonet'], dtype=float)


# --- The data

def subsets(dfic: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Per requirement, its compliant ('cmp') and non-compliant ('uncmp') test photos: im_path, subj_id.

    As DFIC's script: the curated subsets, only the test subjects, only the photos in the two label tables.
    """
    test_subjects = set(pd.read_csv(dfic / 'partitions' / 'ids_balanced_test.txt', header=None)[0].astype(str))
    labelled = set()
    for table, folder in (('camera_image_labels.csv', 'Data_Torso/'), ('artificial_image_labels.csv', 'Artificial_Torso/')):
        labelled |= {folder + path for path in pd.read_csv(dfic / table, usecols=['im_path'])['im_path']}
    out = {}
    for requirement in REQUIREMENTS:
        out[requirement] = {}
        for side in ('cmp', 'uncmp'):
            frame = pd.read_csv(dfic / 'partitions' / 'curated_test' / requirement / f'{side}_df.csv')
            keep = frame['subj_id'].astype(str).isin(test_subjects) & frame['im_path'].isin(labelled)
            out[requirement][side] = frame.loc[keep, ['im_path', 'subj_id']].drop_duplicates('im_path')
    return out


def manifest(dfic: Path) -> pd.DataFrame:
    """Every photo some requirement scores, once, in the shape run_eval.py reads: file (under the set), identity."""
    frames = [side for requirement in subsets(dfic).values() for side in requirement.values()]
    photos = pd.concat(frames).drop_duplicates('im_path').sort_values('im_path')
    return pd.DataFrame({'file': 'preprocessed/' + photos['im_path'], 'identity': photos['subj_id'].astype(str),
                         'split': 'test'})


def load_results(path: Path) -> Dict[str, dict]:
    """results.jsonl of run_eval.py, by im_path: each photo's checks (status, measurements)."""
    out = {}
    with open(path) as f:
        for line in f:
            photo = json.loads(line)
            out[photo['file'].removeprefix('preprocessed/')] = photo['checks']
    return out


# --- From checks to a score

def fail_line(levels: dict) -> Tuple[str, float]:
    """('above' | 'below' | 'at', level): the check's fail level that is on, else its warn level."""
    for prefix in ('fail', 'warn'):
        for kind in ('above', 'below', 'at'):
            level = levels.get(f'{prefix}_{kind}')
            if level is not None and math.isfinite(level):
                if level <= 0:
                    raise ValueError(f'A line at {level} cannot scale a badness')
                return kind, level
    raise ValueError('The check has no level that is on')


def _value(member: Member, measurements: dict) -> Optional[float]:
    if callable(member.value):
        return member.value(measurements)
    return measurements.get(member.value)


def badness(member: Member, check: Optional[dict], levels: dict) -> Optional[float]:
    """How far past its fail line the check puts the photo (1 = on it), 0 for a pass without a number, None
    when the check did not judge the photo."""
    if check is None or check['status'] not in ('pass', 'warn', 'fail'):
        return None
    if member.check == GATE:
        return 0.0 if check['status'] == 'pass' else 2.0
    value = _value(member, check['measurements'])
    if value is None:
        if check['status'] != 'pass':
            raise ValueError(f'{member.check} {check["status"]} without the number it grades')
        return 0.0
    kind, line = fail_line(levels)
    return line / max(value, 1e-9) if kind == 'below' else value / line


def verify(results: Dict[str, dict], levels: Dict[str, dict]) -> None:
    """Refuse a map that does not reproduce the checks: graded with the check's levels, every number a member
    reads must give the status the check reported. A mismatch means a wrong measurement or results from
    other levels (a thresholds file)."""
    from passphoto_validation.core.levels import grade
    wrong = []
    for path, checks in results.items():
        for member in {m for members in REQUIREMENTS.values() for m in members if m.check != GATE}:
            check = checks.get(member.check)
            if check is None or check['status'] not in ('warn', 'fail', 'pass'):
                continue
            value = _value(member, check['measurements'])
            if value is not None and grade(value, **levels[member.check]) != check['status']:
                wrong.append(f'{path} {member.check}: {value} grades {grade(value, **levels[member.check])}, '
                             f'the check said {check["status"]}')
    if wrong:
        raise ValueError(f'{len(wrong)} numbers do not reproduce their check, e.g.\n' + '\n'.join(wrong[:5]))


def shipped_levels() -> Dict[str, dict]:
    """The levels of every check as the pipeline ships them."""
    from passphoto_validation.core.checks import default_checks
    return {check.name: check.levels() for check in default_checks()}


# --- The numbers

def eer(compliant: np.ndarray, non_compliant: np.ndarray) -> float:
    """The equal error rate as DFIC's script takes it: on a grid of false-accept rates (non-compliant photos
    scored at least the threshold) from 0 to 0.999 by 0.001, the true-accept rate read off the ROC polyline,
    and 1 minus it where false accepts plus true accepts are closest to 1."""
    if len(compliant) == 0 or len(non_compliant) == 0:
        return float('nan')
    c, n = np.sort(compliant), np.sort(non_compliant)
    thresholds = np.unique(np.concatenate([c, n]))[::-1]
    tpr = np.concatenate([[0.0], 1 - np.searchsorted(c, thresholds, 'left') / len(c)])
    fpr = np.concatenate([[0.0], 1 - np.searchsorted(n, thresholds, 'left') / len(n)])
    # Several points share a false-accept rate where only compliant photos cross: a vertical step from its
    # lowest (first) to its highest (last) true-accept rate
    xs, first = np.unique(fpr, return_index=True)
    last = np.concatenate([first[1:] - 1, [len(fpr) - 1]])
    low, high = tpr[first], tpr[last]
    grid = np.arange(0, 1, 0.001)
    k = np.searchsorted(xs, grid, 'right') - 1
    on_point = np.isclose(grid, xs[k])
    after = np.minimum(k + 1, len(xs) - 1)
    span = np.where(after > k, xs[after] - xs[k], 1.0)
    between = high[k] + (low[after] - high[k]) * (grid - xs[k]) / span
    curve = np.where(on_point, np.clip(1 - grid, low[k], high[k]), between)
    return float(1 - curve[np.argmin(np.abs(grid + curve - 1))])


def score(dfic: Path, results: Dict[str, dict], levels: Dict[str, dict]) -> pd.DataFrame:
    """One row per requirement: the photos, the unjudged share, both EERs, the verdict at the shipped levels,
    and the published EERs; then the averages over the requirements covered here and over OFIQ's 11."""
    rows = {}
    for requirement, sides in subsets(dfic).items():
        members = REQUIREMENTS[requirement]
        row = {'cmp': len(sides['cmp']), 'uncmp': len(sides['uncmp'])}
        if not members:
            rows[requirement] = row
            continue
        scored = {}
        for side, frame in sides.items():
            missing = [p for p in frame['im_path'] if p not in results]
            if missing:
                raise KeyError(f'{len(missing)} {requirement} photos have no result, e.g. {missing[0]}: '
                               f'run run_eval.py on the manifest --manifest writes')
            worst, failed = [], []
            for path in frame['im_path']:
                checks = results[path]
                bad = [b for m in members if (b := badness(m, checks.get(m.check), levels.get(m.check, {}))) is not None]
                worst.append(max(bad) if bad else np.nan)
                failed.append(any(checks.get(m.check, {}).get('status') == 'fail' for m in members))
            scored[side] = (-np.array(worst), np.array(failed))
        (c, c_failed), (n, n_failed) = scored['cmp'], scored['uncmp']
        c_judged, n_judged = ~np.isnan(c), ~np.isnan(n)
        total = len(c) + len(n)
        row.update(
            unjudged=1 - (c_judged.sum() + n_judged.sum()) / total if total else np.nan,
            eer_dfic=eer(c[c_judged], n[n_judged]),
            # Unjudged scores below every judged one: the worst compliance there is
            eer_strict=eer(np.where(c_judged, c, -np.inf), np.where(n_judged, n, -np.inf)),
            caught=n_failed[n_judged].mean() if n_judged.any() else np.nan,
            false_fail=c_failed[c_judged].mean() if c_judged.any() else np.nan,
        )
        rows[requirement] = row
    table = pd.DataFrame.from_dict(rows, orient='index').join(PUBLISHED)

    def average(names) -> dict:
        part = table.loc[names]
        return {column: part[column].mean() if part[column].notna().all() else np.nan
                for column in ('unjudged', 'eer_dfic', 'eer_strict', 'caught', 'false_fail', 'ofiq', 'biogaze', 'icaonet')}

    covered = [r for r, members in REQUIREMENTS.items() if members]
    table.loc['average, covered here'] = pd.Series(average(covered))
    table.loc["average, OFIQ's 11"] = pd.Series(average(list(PUBLISHED.index[PUBLISHED['ofiq'].notna()])))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', type=Path, required=True, help='The DFIC folder (partitions/, preprocessed/, the label tables)')
    parser.add_argument('--results', default='results.jsonl', help='File name inside --set')
    parser.add_argument('--manifest', action='store_true', help='Write manifest.csv for run_eval.py and stop')
    args = parser.parse_args()

    if args.manifest:
        photos = manifest(args.set)
        photos.to_csv(args.set / 'manifest.csv', index=False)
        print(f'{len(photos)} photos -> {args.set / "manifest.csv"}')
        return

    results = load_results(args.set / args.results)
    levels = shipped_levels()
    verify(results, levels)
    table = score(args.set, results, levels)
    pd.set_option('display.width', 220)
    rate = lambda v: '-' if pd.isna(v) else f'{v:.3f}'  # noqa: E731
    share = lambda v: '-' if pd.isna(v) else f'{v:.0%}'  # noqa: E731
    formatters = {**{c: rate for c in ('eer_dfic', 'eer_strict', 'ofiq', 'biogaze', 'icaonet')},
                  **{c: share for c in ('unjudged', 'caught', 'false_fail')},
                  **{c: (lambda v: '' if pd.isna(v) else f'{v:.0f}') for c in ('cmp', 'uncmp')}}
    print(table.to_string(na_rep='-', formatters=formatters))


if __name__ == '__main__':
    main()
