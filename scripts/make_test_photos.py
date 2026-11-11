"""Make a small set of test photos with known defects from a folder of frontal portraits.

    uv run python scripts/make_test_photos.py --source data/london/neutral_front --out data/test-photos --count 100

Takes portraits from --source (any folder of frontal photos with one face in them: the London Set's
neutral_front after scripts/download_london_set.py, or your own photos), crops each to a compliant 35x45
photo and draws the synthetic defects of scripts/build_eval_set.py into copies, so every output photo
breaks exactly one rule or none. The --count photos are spread evenly over the compliant crop and the
variants; --seed decides which portrait gets which variant, --variants limits the variants.

Writes manifest.csv (file, identity, split, variant, broken_rule) next to the photos, the same shape as the
evaluation set, so scripts/run_eval.py and notebooks/stichprobe.ipynb work on the folder:

    uv run python scripts/run_eval.py --set data/test-photos
    LONDON_SET=data/test-photos uv run --extra test jupyter lab notebooks/stichprobe.ipynb

Photos derived from the London Set stay under its consent terms: don't commit or redistribute them.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_eval_set import SYNTHETIC_VARIANTS, Geometry, crop, synthetic  # noqa: E402

VARIANTS = {'compliant': None, **SYNTHETIC_VARIANTS}
_LONDON_NAME = re.compile(r'^(\d{3})_\d{2}$')  # 001_03.jpg: person 001, photo 03


def identity_of(path: Path) -> str:
    match = _LONDON_NAME.match(path.stem)
    return match.group(1) if match else path.stem


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True, help='Folder of frontal portraits (searched recursively)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--variants', default=','.join(VARIANTS),
                        help=f'Comma-separated subset of: {", ".join(VARIANTS)}')
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        sys.exit(f'unknown variants: {", ".join(unknown)}; choose from {", ".join(VARIANTS)}')
    sources = sorted(p for p in args.source.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not sources:
        sys.exit(f'no photos in {args.source}')
    most = len(sources) * len(variants)
    count = min(args.count, most)
    if count < args.count:
        print(f'{len(sources)} portraits x {len(variants)} variants give at most {most} different photos, making {count}')

    # Spread the count evenly over the variants, and hand the portraits out in a shuffled cycle so no
    # portrait gets the same variant twice
    rng = np.random.default_rng(args.seed)
    order = list(rng.permutation(len(sources)))
    jobs = []  # (portrait index, variant)
    for i in range(count):
        variant = variants[i % len(variants)]
        round_ = i // len(variants)
        jobs.append((order[(round_ + variants.index(variant)) % len(sources)], variant))
    by_portrait = {}
    for index, variant in jobs:
        by_portrait.setdefault(index, []).append(variant)

    args.out.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as mesh, \
            mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0) as segmenter:
        for index in sorted(by_portrait):
            path = sources[index]
            image = cv2.imread(str(path))
            try:
                geometry = Geometry(mesh, segmenter, image)
                compliant = crop(image, geometry)
                base_geometry = Geometry(mesh, segmenter, compliant)
            except ValueError:
                skipped.append(path.name)
                continue
            identity = identity_of(path)
            for variant in by_portrait[index]:
                out = compliant if variant == 'compliant' else synthetic(variant, compliant, base_geometry, image, geometry)
                name = f'{identity}_{variant}.jpg'
                cv2.imwrite(str(args.out / name), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append(dict(file=name, identity=identity, split='test', variant=variant,
                                 broken_rule=VARIANTS[variant] or ''))
            print(f'\r{len(rows)} of {count}', end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)

    with open(args.out / 'manifest.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'identity', 'split', 'variant', 'broken_rule'])
        writer.writeheader()
        writer.writerows(rows)
    if skipped:
        print(f'no face found in {len(skipped)} portraits, skipped: {", ".join(skipped[:5])}{"..." if len(skipped) > 5 else ""}')
    print(f'{len(rows)} photos and manifest.csv -> {args.out}')


if __name__ == '__main__':
    main()
