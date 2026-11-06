"""Build an evaluation set of real photos with glasses, sunglasses and hats from Open Images.

Open Images (Google) photos come from Flickr under CC BY 2.0 (each image's licence and author are in the
metadata; Google notes it didn't verify them), and its boxes are CC BY 4.0. Validation and test have
complete human-drawn boxes, including Human face, Glasses, Sunglasses and hats. This script takes photos
with exactly one large, real (not drawn) face and samples four groups:

- glasses: a Glasses box on the face and no sunglasses
- sunglasses: a Sunglasses box on the face
- hat: a hat box (Hat, Fedora, Sun hat, Cowboy hat, Sombrero) at the top of the face, no eyewear
- plain: no eyewear, goggles, helmet or hat box anywhere

Each photo is cropped to 35x45 like scripts/build_eval_set.py (smaller crops keep their size). The
groups say what the boxes say; what a clerk would reject comes from labels (vlm_labels.jsonl). Keep the
photos out of the repository; credits.csv lists author, source and licence of each.

    uv run python scripts/build_openimages_set.py --meta ~/datasets/openimages-faces/meta \\
        --out ~/datasets/openimages-faces/eval --per-group 60
"""
import argparse
import csv
import hashlib
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

from build_eval_set import Geometry, crop

FACE, GLASSES, SUNGLASSES = '/m/0dzct', '/m/0jyfg', '/m/017ftj'
HATS = {'/m/02dl1y', '/m/02fq_6', '/m/02wbtzl', '/m/025rp__', '/m/02jfl0'}
OTHER_HEADWEAR = {'/m/02_n6y', '/m/0zvk5'}  # goggles, helmet
SPLITS = ('validation', 'test')
IMAGE_URL = 'https://s3.amazonaws.com/open-images-dataset/{subset}/{image_id}.jpg'


def candidates(meta: Path) -> pd.DataFrame:
    """One row per photo with exactly one large real face, with its group (or None)."""
    boxes = pd.concat([pd.read_csv(meta / f'{s}-annotations-bbox.csv') for s in SPLITS])
    info = pd.concat([pd.read_csv(meta / f'{s}-images-with-rotation.csv') for s in SPLITS]).set_index('ImageID')

    faces = boxes[boxes['LabelName'] == FACE]
    count = faces.groupby('ImageID').size()
    faces = faces[faces['ImageID'].isin(count[count == 1].index)]
    faces = faces[(faces['IsGroupOf'] == 0) & (faces['IsDepiction'] == 0) & (faces['IsTruncated'] == 0)]
    faces = faces.assign(h=faces['YMax'] - faces['YMin'], w=faces['XMax'] - faces['XMin'])
    faces = faces[(faces['h'] >= 0.3) & (faces['w'] >= 0.15)].set_index('ImageID')
    # Only photos that need no rotation and carry a CC BY licence
    keep = info.loc[faces.index]
    faces = faces[(keep['Rotation'].fillna(0) == 0) & keep['License'].str.contains('/licenses/by/')]

    wear = boxes[boxes['ImageID'].isin(faces.index) & boxes['LabelName'].isin({GLASSES, SUNGLASSES} | HATS | OTHER_HEADWEAR)]
    face = faces.loc[wear['ImageID']]
    cx = ((wear['XMin'] + wear['XMax']) / 2).values
    cy = ((wear['YMin'] + wear['YMax']) / 2).values
    on_face = (cx >= face['XMin'].values) & (cx <= face['XMax'].values) & (cy >= face['YMin'].values) & (cy <= face['YMax'].values)
    on_top = (cx >= face['XMin'].values) & (cx <= face['XMax'].values) & (wear['YMax'].values >= face['YMin'].values) \
        & (cy <= face['YMin'].values + 0.3 * face['h'].values)

    tags = {}
    for image_id, label, face_hit, top_hit in zip(wear['ImageID'], wear['LabelName'], on_face, on_top):
        found = tags.setdefault(image_id, set())
        found.add('any')
        if label == GLASSES and face_hit:
            found.add('glasses')
        elif label == SUNGLASSES and face_hit:
            found.add('sunglasses')
        elif label in HATS and top_hit:
            found.add('hat')

    def group(image_id):
        found = tags.get(image_id, set())
        if not found:
            return 'plain'
        if 'sunglasses' in found:
            return 'sunglasses'
        if 'glasses' in found and 'hat' not in found:
            return 'glasses'
        if 'hat' in found and not found & {'glasses', 'sunglasses'}:
            return 'hat'
        return None

    out = faces.join(info[['Subset', 'OriginalLandingURL', 'Author', 'License', 'Title']])
    out['group'] = [group(i) for i in out.index]
    return out[out['group'].notna()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--meta', type=Path, required=True, help='Folder with the Open Images CSVs')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--per-group', type=int, default=60)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    raw = args.out.parent / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    pool = candidates(args.meta)
    print(pool['group'].value_counts().to_dict())
    rows, credits = [], []
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as mesh, \
            mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0) as segmenter:
        for group, photos in pool.groupby('group'):
            taken = 0
            # Sample a few extra, since MediaPipe misses some faces
            for image_id, photo in photos.sample(frac=1, random_state=args.seed).iterrows():
                if taken == args.per_group:
                    break
                path = raw / f'{image_id}.jpg'
                if not path.exists():
                    try:
                        with urllib.request.urlopen(IMAGE_URL.format(subset=photo['Subset'], image_id=image_id), timeout=30) as response:
                            path.write_bytes(response.read())
                    except OSError:
                        continue
                image = cv2.imread(str(path))
                if image is None:
                    continue
                try:
                    geometry = Geometry(mesh, segmenter, image)
                    crown, chin = geometry.crown_and_chin()
                    if chin <= crown:
                        raise ValueError('implausible landmarks')
                except ValueError:
                    continue
                name = f'oi_{image_id}.jpg'
                cv2.imwrite(str(args.out / name), crop(image, geometry, upscale=False), [cv2.IMWRITE_JPEG_QUALITY, 95])
                split = 'calibrate' if int(hashlib.md5(image_id.encode()).hexdigest(), 16) % 2 else 'test'
                rows.append(dict(file=name, identity=image_id, split=split, variant=group, broken_rule=''))
                credits.append(dict(file=name, source=photo['OriginalLandingURL'], author=photo['Author'],
                                    license=photo['License'], title=photo['Title']))
                taken += 1
            print(f'{group}: {taken}')

    for filename, table in (('manifest.csv', rows), ('credits.csv', credits)):
        with open(args.out / filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    print(f'{len(rows)} photos -> {args.out}')


if __name__ == '__main__':
    main()
