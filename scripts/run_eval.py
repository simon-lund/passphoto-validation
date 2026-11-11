"""Run the validator on every photo of an evaluation set and save each check's and rule's result.

    OFIQ_CONFIG_DIR=<OFIQ>/data uv run python scripts/run_eval.py --set ~/datasets/frl-london-eval

With several workers, give OFIQ a config that limits its threads (OFIQ_CONFIG_FILE, e.g. a copy of
ofiq_config.jaxn with "ort_intra_threads": 3), or the workers starve each other.

Writes results.jsonl (or --out) next to the manifest: one line per photo with its manifest row, every check
(status, reasons, numeric measurements), every rule status and the problems (the causes, with the rules each decided).
"""
import argparse
import csv
import json
import os
from multiprocessing import Pool
from pathlib import Path

_validator = None


def _init():
    global _validator
    import cv2
    # Every worker would otherwise start a thread per core in OpenCV
    cv2.setNumThreads(1)
    from passphoto_validation.core import PassportPhotoValidator
    _validator = PassportPhotoValidator()


def _run(job):
    row, path = job
    import cv2
    report = _validator(cv2.imread(path))
    return {
        **row,
        'checks': {
            c.name: {
                'status': c.status,
                'reasons': c.reasons,
                'measurements': {k: v for k, v in c.measurements.items() if isinstance(v, (int, float, bool))},
            }
            for c in report.checks
        },
        'rules': {r.id: r.status for r in report.rules},
        # The causes: one entry per check that did not pass, with the rules it decided (report.problems)
        'problems': [
            {'check': p.check, 'status': p.status, 'headline': p.headline, 'rules': p.rules, 'not_judged': p.not_judged}
            for p in report.problems
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', type=Path, required=True, help='Folder with manifest.csv and the photos')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 4))
    parser.add_argument('--out', default='results.jsonl', help='File name inside --set')
    args = parser.parse_args()

    with open(args.set / 'manifest.csv') as f:
        rows = list(csv.DictReader(f))
    jobs = [(row, str(args.set / row['file'])) for row in rows]

    out = args.set / args.out
    with Pool(args.workers, initializer=_init) as pool, open(out, 'w') as f:
        for i, result in enumerate(pool.imap(_run, jobs, chunksize=4), 1):
            f.write(json.dumps(result) + '\n')
            if i % 100 == 0:
                print(f'{i}/{len(jobs)}', flush=True)
    print(f'{len(jobs)} photos -> {out}')


if __name__ == '__main__':
    main()
