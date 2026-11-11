"""The /v1/check report as one self-contained HTML page, for people to read (the JSON is for programs).

The page embeds a downscaled copy of the photo, its own CSS and the JSON answer itself (to copy or
download), so it can be saved, mailed or attached as a single file. It shows the same data as the JSON:
nothing is computed here that the JSON lacks.
"""
import base64
import io
import json
from collections import Counter
from pathlib import Path
from typing import Optional

from fastapi.encoders import jsonable_encoder
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup, escape
from PIL import Image

# Longest side of the embedded photo; enough to see the face, small enough for a mail attachment
PREVIEW_MAX_SIDE = 800

# Every check and rule status, each with its own code (and letter, in the rule map) so none is told
# apart by colour alone; the template adds a texture for skipped and an open border for not_checked
STATUSES = {
    'fail': dict(code='FAIL', letter='F', label='Fail',
                 meaning='A check found that the photo breaks the rule.'),
    'warn': dict(code='WARN', letter='W', label='Warning',
                 meaning='Probably fine, but worth a look; a warning alone does not make a photo invalid.'),
    'pass': dict(code='PASS', letter='P', label='Pass',
                 meaning='The checks that ran found nothing.'),
    'skipped': dict(code='SKIP', letter='S', label='Skipped',
                    meaning="Could not be judged on this photo (no face, too little background, a model that "
                            "isn't installed, ...). A photo with a skipped rule is not valid."),
    'not_checked': dict(code='N/C', letter='N', label='Not checked',
                        meaning='No check covers this rule yet. It does not count against the photo.'),
    'error': dict(code='ERR', letter='E', label='Error',
                  meaning='The check crashed. This is a bug in the pipeline, not a property of the photo.'),
}
# How much of a rule its checks cover; full needs no mark
COVERAGE = {
    'full': None,
    'partial': dict(code='PART', meaning='Partly checked: the checks cover only part of this rule.'),
    'none': dict(code='NONE', meaning='No check covers this rule yet.'),
}
# Rule statuses in the verdict counts, worst first; checks have no not_checked, rules have no error
SUMMARY_ORDER = ['fail', 'warn', 'skipped', 'not_checked', 'pass']
LEGEND_ORDER = ['pass', 'warn', 'fail', 'skipped', 'not_checked', 'error']

_env = Environment(loader=FileSystemLoader(Path(__file__).parent / 'templates'),
                   autoescape=select_autoescape(['html']))


def render_report(payload: dict, image: Image.Image, filename: Optional[str] = None) -> str:
    """Render a /v1/check response body as an HTML page.

    :param payload: The JSON body of /v1/check (report, groups and image size).
    :param image: The photo as it was checked (after EXIF rotation), the size given in payload['image'].
    :param filename: Name of the uploaded file, shown in the header.
    """
    # Plain JSON types, exactly what the JSON answer carries
    payload = jsonable_encoder(payload)
    checks = {check['name']: check for check in payload['checks']}
    groups = []
    for group in payload['groups']:
        rules = [dict(rule, check_results=[checks[name] for name in rule['checks'] if name in checks])
                 for rule in payload['rules'] if rule['group'] == group['id']]
        if rules:
            groups.append(dict(group, rules=rules))
    in_a_rule = {name for rule in payload['rules'] for name in rule['checks']}
    # The Fotomustertafel repeats some rules in two groups; the flat lists name the group for those
    group_titles = {group['id']: group['title'] for group in payload['groups']}
    title_counts = Counter(rule['title'] for rule in payload['rules'])
    flat_title = {rule['id']: f"{rule['title']} ({group_titles.get(rule['group'], rule['group'])})"
                  if title_counts[rule['title']] > 1 else rule['title'] for rule in payload['rules']}

    return _env.get_template('report.html').render(
        report=payload,
        filename=filename,
        photo=_data_uri(image),
        json_uri=_json_uri(payload),
        json_filename=f'{Path(filename).stem if filename else "report"}.json',
        groups=groups,
        other_checks=[check for check in payload['checks'] if check['name'] not in in_a_rule],
        # The rules that did not pass, by cause: what the checks found (fails first, then warnings), and the
        # checks that could not judge the photo, each in its own box
        problems=[problem for problem in payload['problems'] if problem['status'] in ('fail', 'warn')],
        not_judged=[problem for problem in payload['problems'] if problem['status'] in ('skipped', 'error')],
        flat_title=flat_title,
        crop_proposed=bool(payload.get('crop')) and payload['crop']['status'] == 'proposed',
        crop_rows=_crop_rows(payload),
        group_titles=group_titles,
        counts=Counter(rule['status'] for rule in payload['rules']),
        statuses=STATUSES,
        coverage=COVERAGE,
        summary_order=SUMMARY_ORDER,
        legend_order=LEGEND_ORDER,
        fmt=_format_value,
        pct=_percent,
        breakable=_breakable,
        detail=_detail,
    )


def _data_uri(image: Image.Image) -> str:
    preview = image.convert('RGB')
    preview.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
    buffer = io.BytesIO()
    preview.save(buffer, format='JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def _crop_rows(payload: dict) -> list:
    # Rules the crop changes, and rules that still block the cropped photo, in catalogue order
    crop = payload.get('crop') or {}
    after = crop.get('rules_after') or {}
    return [dict(rule, after=after[rule['id']]) for rule in payload['rules']
            if rule['id'] in after and (after[rule['id']] != rule['status'] or after[rule['id']] in ('fail', 'skipped'))]


def _detail(problem: dict) -> str:
    # What a problem's reasons say beyond its headline
    text = ' '.join(reason.strip() for reason in problem['reasons'] if reason.strip())
    return text[len(problem['headline']):].strip() if text.startswith(problem['headline']) else text


def _json_uri(payload: dict) -> str:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    return 'data:application/json;charset=utf-8;base64,' + base64.b64encode(text.encode('utf-8')).decode('ascii')


def _breakable(name) -> Markup:
    # snake_case names may wrap after each underscore
    return Markup('_<wbr>').join(escape(part) for part in str(name).split('_'))


def _percent(value, total) -> float:
    return round(100 * value / total, 3) if total else 0.0


def _format_value(value) -> str:
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, float):
        return f'{value:.4g}'
    return str(value)
