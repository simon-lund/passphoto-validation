"""Download the Face Research Lab London Set, the photos behind the evaluation.

    uv run python scripts/download_london_set.py --out data/london

Fetches the four photo sets the evaluation uses (neutral and smiling front views, left and right 3/4
views; 136 MB) and the info sheet from figshare (DeBruine & Jones 2017, CC BY 4.0,
https://doi.org/10.6084/m9.figshare.5047666), checks every file's MD5 against figshare's and unzips it.
Files already there with the right MD5 are not downloaded again, so the script can be re-run.
--all fetches the remaining views too (profiles and smiling 3/4 views, another 145 MB), which no check uses.

The subjects consented to use "in lab-based and web-based studies"; keep the photos and anything derived
from them out of the repository and don't redistribute them. Then build the evaluation set:

    uv run python scripts/build_eval_set.py --source data/london --out data/london-eval
"""
import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ARTICLE = 'https://api.figshare.com/v2/articles/5047666'
EVALUATION_FILES = (
    'london_faces_info.csv',
    'facelab_london.pdf',
    'neutral_front.zip',
    'smiling_front.zip',
    'neutral_left_3quarter.zip',
    'neutral_right_3quarter.zip',
)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, size: int):
    with urllib.request.urlopen(url, timeout=60) as response, open(target, 'wb') as f:
        done = 0
        for chunk in iter(lambda: response.read(1 << 20), b''):
            f.write(chunk)
            done += len(chunk)
            print(f'\r  {target.name}: {done / 1e6:.1f} of {size / 1e6:.1f} MB', end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)


def unzip(archive: Path, out: Path):
    """Extract the photos; the zips carry a top-level folder of their own name plus macOS cruft."""
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.namelist()
                   if not m.startswith('__MACOSX/') and not Path(m).name.startswith('.') and not m.endswith('/')]
        for member in members:
            target = out / member
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(target, 'wb') as dst:
                dst.write(src.read())
    return sum(1 for m in members if m.lower().endswith('.jpg'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True, help='Folder for the zips and the unpacked photos')
    parser.add_argument('--all', action='store_true', help='Also fetch the profile views (not used by the evaluation)')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(ARTICLE, timeout=60) as response:
        article = json.load(response)
    wanted = {f['name']: f for f in article['files']}
    names = [n for n in wanted if n.endswith('.zip') or n in EVALUATION_FILES] if args.all else list(EVALUATION_FILES)
    missing = [n for n in names if n not in wanted]
    if missing:
        sys.exit(f'figshare no longer lists {", ".join(missing)}; see {article.get("url_public_html", ARTICLE)}')

    print(f'{article["title"]} ({article["doi"]}, {article["license"]["name"]}) -> {args.out}')
    for name in names:
        info = wanted[name]
        target = args.out / name
        unpacked = args.out / target.stem
        if target.exists() and md5(target) == info['computed_md5']:
            if not name.endswith('.zip') or any(unpacked.rglob('*.jpg')):
                print(f'  {name}: already there')
                continue
        else:
            download(info['download_url'], target, info['size'])
            if md5(target) != info['computed_md5']:
                sys.exit(f'{name}: MD5 does not match figshare\'s, the download is damaged; delete it and run again')
        if name.endswith('.zip'):
            print(f'  {name}: {unzip(target, args.out)} photos unpacked')

    photos = sum(1 for _ in args.out.rglob('*.jpg'))
    print(f'{photos} photos in {args.out}. Next:\n'
          f'  uv run python scripts/build_eval_set.py --source {args.out} --out {args.out.parent / "london-eval"}')


if __name__ == '__main__':
    main()
