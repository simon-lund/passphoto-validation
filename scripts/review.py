"""Page through a photo set and say, photo by photo, whether the pipeline's verdict is right.

    uv run python scripts/review.py --set data/london-eval --split test --shuffle
    uv run python scripts/review.py --set data/openimages/eval
    uv run python scripts/review.py --set data/london-eval --check colour_cast=warn
    uv run python scripts/review.py --set data/london-eval --summary

Serves one page at http://127.0.0.1:9700: the photo with what the pipeline saw drawn over it (the person
mask and the band next to it, the face box with crown and chin, the face mesh, the eye contours and irises,
the whites of the eyes, OFIQ's landmarks and face parsing, the planned crop; each layer can be switched off),
the label when the set has one, and the verdict as problems: each check that failed, warned or could not run,
with its reason and the rules it decided, so one defect reads as one problem however many rules it breaks.

Keys: Y or the right arrow when the verdict is right, N or the left arrow when it is wrong, C to write a
comment first, a click on a rule's chip to mark that rule's status as wrong, S to skip, B to go back. On a touch
screen, swipe right or left. A held key acts once, and an answer counts only once the photo has been on screen
for a moment, so no photo is answered unseen. Each answer is appended to reviews.jsonl next to the set: the file, the answer,
the rules marked wrong, the comment, the verdict as shown, and the pipeline's git commit and thresholds file.
Photos already answered are left out on the next run (--again keeps them). --summary prints what the answers
say per rule and per variant, which is what the thresholds are tuned on.

The verdicts are computed live, a few photos ahead of the one on screen; --thresholds applies a thresholds
file (see `passport-validator thresholds`), and without it the file THRESHOLDS_FILE names, as for the CLI
and the API. The filters --split, --variant, --check NAME=STATUS and
--rule ID=STATUS read the set's manifest.csv and results.jsonl, so a run can show only, say, the photos
where colour_cast warned. The photos never leave this machine: the server listens on 127.0.0.1 and writes
nothing but reviews.jsonl.

Every error answer is `{"code": "<stable_snake_case>", "detail": "<human text>"}`, as in check_api.py; the
page branches on the status and the code only.
"""
import argparse
import base64
import csv
import json
import random
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

from passphoto_validation.check_api import ApiError, handle_api_error, handle_validation_error
from passphoto_validation.core import PassportPhotoValidator, ValidatorConfig
from passphoto_validation.core.checks.eyes import MP_LEFT_EYE, MP_RIGHT_EYE
from passphoto_validation.core.checks.lighting import LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR, eye_whites
from passphoto_validation.core.models import CheckSkipped
from passphoto_validation.core.ofiq import PARSING_LABELS
from passphoto_validation.core.rules import RULES
from passphoto_validation.core.thresholds import ThresholdsFileError
from passphoto_validation.core.utils import mesh_points, prepare_image

HERE = Path(__file__).resolve().parent
PAGE = HERE / 'review.html'
REVIEWS = 'reviews.jsonl'
PREFETCH = 6  # photos computed ahead of the one on screen
PHOTO_SUFFIXES = {'.jpg', '.jpeg', '.png'}
RULE_TITLES = {rule.id: rule.title for rule in RULES}

# One colour (R, G, B) per OFIQ face-parsing label; the background stays transparent
PARSING_COLOURS = {
    'skin': (240, 200, 170), 'left_brow': (120, 80, 40), 'right_brow': (120, 80, 40),
    'left_eye': (30, 120, 220), 'right_eye': (30, 120, 220), 'glasses': (0, 200, 220),
    'left_ear': (200, 150, 120), 'right_ear': (200, 150, 120), 'earring': (250, 220, 0), 'nose': (250, 150, 120),
    'mouth': (220, 40, 80), 'upper_lip': (220, 40, 80), 'lower_lip': (220, 40, 80), 'neck': (230, 190, 160),
    'necklace': (250, 220, 0), 'clothing': (90, 90, 200), 'hair': (90, 50, 20), 'hat': (160, 60, 180),
}


# --- The queue: which photos, in which order, with which labels

def read_rows(folder: Path):
    """The set's manifest rows, or one row per photo when there is no manifest."""
    manifest = folder / 'manifest.csv'
    if manifest.exists():
        with open(manifest, newline='') as f:
            return list(csv.DictReader(f))
    photos = sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)
    return [dict(file=p.name) for p in photos]


def read_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_reviews(folder: Path):
    """The answers given so far, by file (the last answer counts)."""
    return {record['file']: record for record in read_jsonl(folder / REVIEWS)}


def parse_pairs(values, what):
    pairs = []
    for value in values or ():
        if '=' not in value:
            sys.exit(f'--{what} takes NAME=STATUS, not {value!r}')
        pairs.append(tuple(value.split('=', 1)))
    return pairs


def build_queue(args):
    folder = args.set
    rows = read_rows(folder)
    results = {record['file']: record for record in read_jsonl(folder / 'results.jsonl')}
    notes = {record['file']: record.get('note', '') for record in read_jsonl(folder / 'vlm_labels.jsonl')}
    credits = {}
    if (folder / 'credits.csv').exists():
        with open(folder / 'credits.csv', newline='') as f:
            credits = {row['file']: row for row in csv.DictReader(f)}
    checks, rules = parse_pairs(args.check, 'check'), parse_pairs(args.rule, 'rule')
    if (checks or rules) and not results:
        sys.exit(f'--check and --rule need {folder / "results.jsonl"} (scripts/run_eval.py writes it)')
    variants = set(args.variant.split(',')) if args.variant else None

    items = []
    for row in rows:
        file = row['file']
        if args.split and row.get('split') != args.split:
            continue
        if variants and row.get('variant') not in variants:
            continue
        stored = results.get(file)
        if any(not stored or stored['checks'].get(name, {}).get('status') != status for name, status in checks):
            continue
        if any(not stored or stored['rules'].get(rule_id) != status for rule_id, status in rules):
            continue
        credit = credits.get(file)
        items.append(dict(
            file=file, split=row.get('split', ''), variant=row.get('variant', ''), broken_rule=row.get('broken_rule', ''),
            broken_rule_title=RULE_TITLES.get(row.get('broken_rule', ''), ''), note=notes.get(file, ''),
            credit=f"{credit['author']} ({credit['title']}), {credit['license']}" if credit else '',
        ))
    if args.shuffle:
        random.Random(args.seed).shuffle(items)
    reviewed = read_reviews(folder)
    if not args.again:
        items = [item for item in items if item['file'] not in reviewed]
    if args.limit:
        items = items[:args.limit]
    for i, item in enumerate(items):
        item['i'] = i
        item['reviewed'] = reviewed.get(item['file'], {}).get('agree')
    return items, reviewed


# --- What the pipeline saw, as geometry for the page to draw

def contours(mask: np.ndarray):
    """The outlines of a 0/1 mask as polygons, simplified to about a pixel."""
    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2).tolist() for c in found if len(c) >= 3]


def parsing_png(labels: np.ndarray) -> str:
    """OFIQ's face-parsing map as a transparent PNG, base64, one colour per label."""
    palette = np.zeros((max(PARSING_LABELS.values()) + 1, 4), np.uint8)
    for name, index in PARSING_LABELS.items():
        if name in PARSING_COLOURS:
            r, g, b = PARSING_COLOURS[name]
            palette[index] = (b, g, r, 255)
    image = palette[np.clip(labels, 0, len(palette) - 1)]
    ok, png = cv2.imencode('.png', image)
    return base64.b64encode(png.tobytes()).decode('ascii') if ok else ''


def overlays(context, report) -> dict:
    out = {}
    mask = context.person_mask_binary
    if mask is not None:
        out['mask'] = contours(mask)
        band = max(3, int(round(min(context.width, context.height) * 0.03)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
        out['band'] = contours(cv2.dilate(mask, kernel))
    out['face_box'] = context.face_data.get('bbox')
    out['crown_y'] = report.measurements.get('crown_y')
    out['chin_y'] = report.measurements.get('chin_y')
    try:
        points = mesh_points(context, iris=True)
    except CheckSkipped:
        points = None
    if points is not None:
        out['mesh'] = points.round(1).tolist()
        out['eyes'] = [points[list(contour)].round(1).tolist() for contour in (RIGHT_EYE_CONTOUR, LEFT_EYE_CONTOUR)]
        out['irises'] = []
        for eye in (MP_RIGHT_EYE, MP_LEFT_EYE):
            centre = points[eye['iris']]
            radius = float(np.mean([np.linalg.norm(points[i] - centre) for i in eye['ring']]))
            out['irises'].append(dict(x=round(float(centre[0]), 1), y=round(float(centre[1]), 1), r=round(radius, 1)))
        out['eye_whites'] = contours(eye_whites(context, points))
    if context.ofiq is not None:
        if context.ofiq.landmarks is not None:
            out['ofiq_landmarks'] = np.asarray(context.ofiq.landmarks).round(1).tolist()
        if context.ofiq.parsing is not None:
            out['parsing'] = parsing_png(context.ofiq.parsing)
    crop = report.crop
    if crop is not None and crop.status == 'proposed':
        out['crop_box'] = dict(x=crop.x, y=crop.y, width=crop.width, height=crop.height)
    return out


def verdict(report) -> dict:
    def numbers(measurements):
        return {key: (round(value, 3) if isinstance(value, float) else value)
                for key, value in measurements.items() if isinstance(value, (int, float, bool, str))}
    return dict(
        is_valid=report.is_valid,
        skipped_rules=list(report.skipped_rules),
        rules=[dict(id=r.id, title=r.title, status=r.status, reasons=r.reasons, checks=list(r.checks)) for r in report.rules],
        checks=[dict(name=c.name, status=c.status, reasons=c.reasons, measurements=numbers(c.measurements)) for c in report.checks],
        problems=[asdict(p) for p in report.problems],
        crop=asdict(report.crop) if report.crop is not None else None,
    )


class Computer(threading.Thread):
    """Runs the validator on the photos around the one on screen, in one thread (MediaPipe is not thread-safe).

    The page asks for one photo at a time and numbers its requests. Only the newest request moves the window
    the thread works on; a request that a newer one has replaced answers SUPERSEDED at once instead of
    waiting for a photo that is no longer computed (it would hold a server thread until the timeout, and a
    request arriving late would pull the window back from the photo on screen).
    """
    SUPERSEDED, TIMED_OUT = 'superseded', 'timed_out'

    def __init__(self, validator: PassportPhotoValidator, folder: Path, items: list, crop: bool):
        super().__init__(daemon=True)
        self.validator, self.folder, self.items, self.crop = validator, folder, items, crop
        self.cache = {}
        self.wanted = 0
        self.newest = 0  # the highest request number seen
        self.condition = threading.Condition()

    def get(self, i: int, request: int = None, timeout: float = 300):
        """The result for photo `i`, or SUPERSEDED / TIMED_OUT. `request` is the page's running number of
        the request; without one the call counts as the newest."""
        with self.condition:
            if request is None:
                request = self.newest + 1
            if request >= self.newest:
                self.newest, self.wanted = request, i
                self.condition.notify_all()
            deadline = time.monotonic() + timeout
            while i not in self.cache:
                if request < self.newest:
                    return self.SUPERSEDED
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self.TIMED_OUT
                self.condition.wait(remaining)
            return self.cache[i]

    def run(self):
        while True:
            with self.condition:
                while True:
                    window = range(self.wanted, min(self.wanted + PREFETCH + 1, len(self.items)))
                    todo = next((j for j in window if j not in self.cache), None)
                    if todo is not None:
                        break
                    self.condition.wait()
            try:
                result = self.compute(todo)
            except Exception as exc:  # never let one photo end the thread: every later request would time out
                result = dict(i=todo, file=self.items[todo]['file'], error=f'{type(exc).__name__}: {exc}')
            with self.condition:
                self.cache[todo] = result
                for j in list(self.cache):
                    if j < self.wanted - 3 or j > self.wanted + PREFETCH + 3:
                        del self.cache[j]
                self.condition.notify_all()

    def compute(self, i: int) -> dict:
        item = self.items[i]
        started = time.monotonic()
        try:
            image = prepare_image(Image.open(self.folder / item['file']))
            report, context = self.validator.inspect(image, crop=self.crop)
            return dict(i=i, file=item['file'], error=None, width=context.width, height=context.height,
                        seconds=round(time.monotonic() - started, 1), verdict=verdict(report),
                        overlays=overlays(context, report))
        except Exception as exc:  # shown on the card, the queue goes on
            return dict(i=i, file=item['file'], error=f'{type(exc).__name__}: {exc}')


# --- Answers

def git_commit() -> str:
    try:
        return subprocess.run(['git', '-C', str(HERE.parent), 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ''


def summary(folder: Path):
    reviews = read_reviews(folder)
    if not reviews:
        print(f'No answers yet in {folder / REVIEWS}')
        return
    answers = list(reviews.values())
    agree = sum(1 for r in answers if r['agree'])
    disagree = [r for r in answers if not r['agree']]
    print(f'{len(answers)} photos answered in {folder / REVIEWS}: {agree} right, {len(disagree)} wrong')

    by_variant = defaultdict(lambda: [0, 0])
    for r in answers:
        by_variant[r.get('variant') or '(no label)'][0 if r['agree'] else 1] += 1
    print('\nPer variant (right / wrong):')
    for variant, (right, wrong) in sorted(by_variant.items(), key=lambda kv: -kv[1][1]):
        print(f'  {variant:22} {right:4} / {wrong:4}')

    wrong_rules = Counter(rule for r in answers for rule in r.get('wrong_rules', []))
    if wrong_rules:
        print('\nRules marked wrong (times):')
        for rule, n in wrong_rules.most_common():
            print(f'  {rule:28} {n:4}  {RULE_TITLES.get(rule, "")}')

    print('\nWrong verdicts: what the pipeline said, by its failed rules:')
    said = Counter()
    for r in disagree:
        failed = r['verdict'].get('failed') or []
        said['valid' if r['verdict'].get('is_valid') else ('fail: ' + ', '.join(failed) if failed else 'not valid (skipped rules)')] += 1
    for what, n in said.most_common():
        print(f'  {n:4}  {what}')

    comments = [(r['file'], r['comment']) for r in answers if r.get('comment')]
    if comments:
        print('\nComments:')
        for file, comment in comments:
            print(f'  {file}: {comment}')


# --- The server

async def handle_crash(request: Request, exc: Exception):
    # A bug in this server. Starlette re-raises it after this answer, so the traceback is in the server's log
    return JSONResponse(status_code=500, content={
        'code': 'internal_error', 'detail': 'The review server hit an error; the traceback is in its log.'})


def make_app(args, items, reviewed, computer, config):
    app = FastAPI(title='passphoto-validation review', docs_url=None, redoc_url=None)
    # The same error answers as the check API: a stable code next to the text
    app.add_exception_handler(ApiError, handle_api_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(Exception, handle_crash)
    # The last answer per photo counts, so answering a photo again (after B) moves its count and adds none
    answers = {file: record['agree'] for file, record in reviewed.items()}
    commit = git_commit()

    def counts():
        return dict(Counter('right' if agree else 'wrong' for agree in answers.values()))

    def no_such_photo(status: int):
        return ApiError(status, 'photo_not_in_queue', 'No such photo in the queue.')

    @app.get('/', response_class=HTMLResponse)
    def page():
        return PAGE.read_text(encoding='utf-8')

    @app.get('/api/queue')
    def queue():
        return dict(set=args.set.name, path=str(args.set), reviews=str(args.set / REVIEWS), items=items,
                    counts=counts(), commit=commit, thresholds_file=config.thresholds_file,
                    filters=dict(split=args.split, variant=args.variant, check=args.check or [], rule=args.rule or [],
                                 shuffle=args.shuffle, again=args.again))

    @app.get('/api/item/{i}')
    def item(i: int, request: int = None):
        if not 0 <= i < len(items):
            raise no_such_photo(404)
        result = computer.get(i, request)
        if result == Computer.SUPERSEDED:
            raise ApiError(409, 'superseded', 'A later request for another photo replaced this one.')
        if result == Computer.TIMED_OUT:
            raise ApiError(504, 'pipeline_timeout', 'The pipeline did not answer in time.')
        return result

    @app.get('/api/photo/{i}')
    def photo(i: int):
        if not 0 <= i < len(items):
            raise no_such_photo(404)
        return FileResponse(args.set / items[i]['file'])

    @app.post('/api/review')
    def review(answer: dict):
        i = answer.get('i')
        if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < len(items):
            raise no_such_photo(400)
        item = items[i]
        if not isinstance(answer.get('agree'), bool):
            raise ApiError(400, 'answer_missing', 'The answer needs "agree": true or false.')
        record = dict(
            file=item['file'], set=args.set.name, time=datetime.now(timezone.utc).isoformat(timespec='seconds'),
            agree=answer['agree'], wrong_rules=[str(r) for r in answer.get('wrong_rules') or []],
            comment=str(answer.get('comment') or '').strip(), verdict=answer.get('verdict') or {},
            variant=item['variant'], broken_rule=item['broken_rule'], commit=commit, thresholds_file=config.thresholds_file,
        )
        try:
            with open(args.set / REVIEWS, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except OSError as exc:
            raise ApiError(500, 'review_not_saved', f'Could not write {args.set / REVIEWS}: {exc}')
        answers[item['file']] = record['agree']
        item['reviewed'] = record['agree']
        return dict(counts=counts())

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', type=Path, required=True, help='Folder with the photos (and manifest.csv, results.jsonl if any)')
    parser.add_argument('--port', type=int, default=9700)
    parser.add_argument('--split', help='Only this split of the manifest (calibrate or test)')
    parser.add_argument('--variant', help='Only these variants, comma-separated (compliant,colour_cast,...)')
    parser.add_argument('--check', action='append', metavar='NAME=STATUS', help='Only photos where this check had this status in results.jsonl')
    parser.add_argument('--rule', action='append', metavar='ID=STATUS', help='Only photos where this rule had this status in results.jsonl')
    parser.add_argument('--shuffle', action='store_true', help='Random order (--seed)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, help='At most this many photos')
    parser.add_argument('--again', action='store_true', help='Include photos already answered')
    parser.add_argument('--no-crop', action='store_true', help='Skip planning the 35x45 crop (faster)')
    parser.add_argument('--thresholds', type=Path,
                        help='A thresholds file to run with (passport-validator thresholds); without it, the one THRESHOLDS_FILE names')
    parser.add_argument('--summary', action='store_true', help='Print what the answers so far say, and exit')
    args = parser.parse_args()
    if not args.set.is_dir():
        sys.exit(f'{args.set} is not a folder')
    if args.summary:
        summary(args.set)
        return

    items, reviewed = build_queue(args)
    if not items:
        sys.exit(f'Nothing to review: every matching photo in {args.set} has an answer (--again shows them again)')
    # Without --thresholds the field is left out, so THRESHOLDS_FILE (or .env) decides, as for the CLI and the API
    config = ValidatorConfig(thresholds_file=str(args.thresholds)) if args.thresholds else ValidatorConfig()
    try:
        validator = PassportPhotoValidator(config)
    except ThresholdsFileError as exc:
        sys.exit(str(exc))
    if config.thresholds_file:
        print(f'Thresholds file: {config.thresholds_file}', file=sys.stderr)
    if validator.ofiq is None:
        print(f'OFIQ is not loaded ({validator.ofiq_unavailable}); the OFIQ rules will show as skipped', file=sys.stderr)
    computer = Computer(validator, args.set, items, crop=not args.no_crop)
    computer.start()
    print(f'{len(items)} photos to review ({len(reviewed)} answered before) -> http://127.0.0.1:{args.port}', flush=True)
    uvicorn.run(make_app(args, items, reviewed, computer, config), host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
