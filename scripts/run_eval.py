"""Run the validator on every photo of an evaluation set and save each check's and rule's result.

    OFIQ_CONFIG_DIR=<OFIQ>/data uv run python scripts/run_eval.py --set data/london-eval

With several workers OFIQ must not take every core in each of them, or the workers starve one another. So
unless OFIQ_CONFIG_FILE says otherwise, several workers run with ofiq_config_eval.jaxn (3 threads each),
which install.sh writes next to ofiq_config.jaxn; when that file is missing the run says so and goes on.

Writes results.jsonl (or --out) next to the manifest: one line per photo with its manifest row, every check
(status, reasons, numeric measurements), every rule status and the problems (the causes, with the rules each decided).
"""
import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

EVAL_OFIQ_CONFIG = 'ofiq_config_eval.jaxn'  # install.sh writes it: ofiq_config.jaxn with 3 ORT threads
_validator = None


def ofiq_config_file(workers: int):
    """The OFIQ config file the workers run with, or None for the configured one (OFIQ_CONFIG_FILE, .env).

    With several workers and nothing configured, the thread-limited copy is used when it is there.
    """
    from passphoto_validation.core import ValidatorConfig
    config = ValidatorConfig()
    if workers < 2 or 'ofiq_config_file' in config.model_fields_set or not config.ofiq_config_dir:
        return None
    if (Path(config.ofiq_config_dir) / EVAL_OFIQ_CONFIG).exists():
        return EVAL_OFIQ_CONFIG
    print(f'{workers} workers, and no {EVAL_OFIQ_CONFIG} in {config.ofiq_config_dir}: OFIQ takes every core in each '
          f'worker, which is slow. ./install.sh writes that file; --workers 1 avoids the problem.', file=sys.stderr)
    return None


def _init(ofiq_file=None):
    global _validator
    import cv2
    # Every worker would otherwise start a thread per core in OpenCV
    cv2.setNumThreads(1)
    from passphoto_validation.core import PassportPhotoValidator, ValidatorConfig
    _validator = PassportPhotoValidator(ValidatorConfig(ofiq_config_file=ofiq_file) if ofiq_file else None)


def _run(job):
    row, path = job
    import cv2
    # Without the crop: it checks the cropped photo a second time, and the rows carry nothing of it
    report = _validator(cv2.imread(path), crop=False)
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
    ofiq_file = ofiq_config_file(args.workers)
    if ofiq_file:
        print(f'{args.workers} workers, OFIQ with {ofiq_file}', flush=True)
    with Pool(args.workers, initializer=_init, initargs=(ofiq_file,)) as pool, open(out, 'w') as f:
        for i, result in enumerate(pool.imap(_run, jobs, chunksize=4), 1):
            f.write(json.dumps(result) + '\n')
            if i % 100 == 0:
                print(f'{i}/{len(jobs)}', flush=True)
    print(f'{len(jobs)} photos -> {out}')


if __name__ == '__main__':
    main()
