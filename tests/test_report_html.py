"""The HTML report shows each rule with its own status, escapes what it shows and embeds a small photo."""
import base64
import io
import json
import re

from markupsafe import escape
from PIL import Image

from edutap.image_api.report_html import COVERAGE, PREVIEW_MAX_SIDE, STATUSES, SUMMARY_ORDER, render_report


def _check(name, status, reasons=(), **measurements):
    return {'name': name, 'status': status, 'reasons': list(reasons), 'measurements': measurements}


def _rule(rule_id, status, checks, coverage='full', reasons=()):
    return {'id': rule_id, 'group': 'eyes', 'title': f'Rule {rule_id}', 'status': status, 'checks': checks,
            'coverage': coverage, 'reasons': list(reasons), 'note': None}


def _payload(bbox=None, crop=None):
    return {
        'is_valid': False,
        'overall_score': 80.0,
        'bbox': bbox,
        'suggested_crop': None,
        'crop': crop,
        'checks': [
            _check('passing', 'pass'),
            _check('warning', 'warn', ['Maybe'], ratio=0.123456, worn=True),
            _check('failing', 'fail', ['<script>alert(1)</script>']),
            _check('skipping', 'skipped', ['No face']),
            _check('crashing', 'error', ['Boom']),
        ],
        'rules': [
            _rule('a', 'pass', ['passing']),
            _rule('b', 'warn', ['passing', 'warning'], coverage='partial'),
            _rule('c', 'fail', ['failing'], reasons=['<script>alert(1)</script>']),
            _rule('d', 'skipped', ['skipping'], reasons=['No face']),
            _rule('e', 'not_checked', [], coverage='none'),
        ],
        'skipped_rules': ['d'],
        'groups': [{'id': 'eyes', 'title': 'Eyes and glasses'}, {'id': 'empty', 'title': 'A group without rules'}],
        'image': {'width': 2000, 'height': 1000},
    }


def _crop(status, **fields):
    # The API sends every key whatever the status
    return {'status': status, 'reason': None, 'x': None, 'y': None, 'width': None, 'height': None, 'is_valid': None,
            'rules_after': {}, 'fixed_rules': [], 'remaining_rules': [], **fields}


def _proposed_crop(fixes_all=False):
    # Rule b warns and c fails now; both pass after the crop. d stays skipped unless the crop fixes it all.
    return _crop('proposed', x=0, y=0, width=500, height=640, is_valid=fixes_all,
                 rules_after={'a': 'pass', 'b': 'pass', 'c': 'pass', 'd': 'pass' if fixes_all else 'skipped', 'e': 'not_checked'},
                 fixed_rules=['b', 'c'], remaining_rules=[] if fixes_all else ['d'])


def _render(**payload):
    return render_report(_payload(**payload), Image.new('RGB', (2000, 1000), (200, 200, 200)), 'photo.jpg')


def _section(page, section_id):
    return page.split(f'<section id="{section_id}">')[1].split('</section>')[0]


def test_every_rule_shows_its_own_status():
    page = _render()
    shown = dict(re.findall(r'data-rule="(\w+)" data-status="(\w+)"', page))
    assert shown == {rule['id']: rule['status'] for rule in _payload()['rules']}
    # The rule map has one cell per rule, with the same status
    cells = re.findall(r'class="cell" href="#rule-(\w+)" data-status="(\w+)"', page)
    assert dict(cells) == shown and len(cells) == len(shown)


def test_failing_and_warning_rules_are_listed_first_with_links():
    to_fix = _section(_render(), 'to-fix')
    assert re.findall(r'href="#rule-(\w+)"', to_fix) == ['c', 'b']


def test_skipped_rules_are_called_out_and_loose_checks_listed():
    page = _render()
    not_judged = _section(page, 'not-judged')
    assert 'Rule d</a>' in not_judged and 'No face' in not_judged
    # The crashing check covers no rule, so it gets its own section
    assert 'crashing' in _section(page, 'other-checks')
    assert 'A group without rules' not in page


def test_a_rule_title_used_in_two_groups_names_its_group_in_the_lists():
    payload = _payload()
    payload['groups'].append({'id': 'light', 'title': 'Lighting'})
    payload['rules'].append(dict(payload['rules'][3], id='d2', group='light', title='Rule d'))
    payload['skipped_rules'].append('d2')
    page = render_report(payload, Image.new('RGB', (2000, 1000)), 'photo.jpg')
    not_judged = _section(page, 'not-judged')
    assert 'Rule d (Eyes and glasses)</a>' in not_judged and 'Rule d (Lighting)</a>' in not_judged


def test_status_and_cover_codes_explain_themselves():
    page = _render()
    for status in SUMMARY_ORDER:
        code, label, meaning = STATUSES[status]['code'], STATUSES[status]['label'], STATUSES[status]['meaning']
        assert f'<abbr title="{escape(f"{label}: {meaning}")}">{escape(code)}</abbr>' in page
    # Rule b is partly covered and rule e has no check; fully covered rules carry no code
    for coverage in ('partial', 'none'):
        assert page.count(f'<abbr title="{escape(COVERAGE[coverage]["meaning"])}">{COVERAGE[coverage]["code"]}</abbr>') == 1
    assert '<section id="status-codes">' in page


def test_text_from_the_report_is_escaped():
    page = _render()
    assert '<script>alert(1)</script>' not in page
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in page


def test_measurements_are_readable():
    page = _render()
    assert '<span class="k">ratio</span> <span class="v">0.1235</span>' in page
    assert '<span class="k">worn</span> <span class="v">yes</span>' in page


def test_face_and_crop_are_drawn_only_when_present():
    assert '<rect' not in _render()
    assert 'No face detected' in _render()

    page = _render(bbox={'x': 10, 'y': 20, 'width': 300, 'height': 400}, crop=_proposed_crop())
    assert page.count('<rect') == 2
    assert '<rect x="10" y="20" width="300" height="400"' in page
    assert '<rect x="0" y="0" width="500" height="640" class="crop"' in page


def test_the_crop_section_lists_what_changes_and_what_still_blocks():
    page = _render(crop=_proposed_crop())
    crop = _section(page, 'crop')
    # b and c change; d is still skipped, so it still blocks; a and e don't change
    assert re.findall(r'href="#rule-(\w+)"', crop) == ['b', 'c', 'd']
    assert 'still fails 1 rule' in crop
    assert 'after 35×45 crop' not in page


def test_a_photo_a_crop_makes_valid_says_so_on_the_verdict():
    page = _render(crop=_proposed_crop(fixes_all=True))
    assert 'The cropped photo is valid.' in _section(page, 'crop')
    assert 'after 35×45 crop</span>' in page


def test_a_crop_that_is_not_possible_says_why_and_not_needed_has_no_section():
    reason = 'The face is too close to the left edge to crop the photo to 35x45 without adding background.'
    page = _render(crop=_crop('not_possible', reason=reason))
    assert f'Not possible: {reason}' in _section(page, 'crop')
    page = _render(crop=_crop('not_needed'))
    assert '<section id="crop">' not in page and 'Not needed' in page


def test_the_json_to_copy_or_download_is_the_payload():
    page = _render()
    data = re.search(r'href="data:application/json;charset=utf-8;base64,([^"]+)" download="photo.json"', page).group(1)
    assert json.loads(base64.b64decode(data)) == _payload()
    assert 'id="copy-json" hidden' in page  # shown by the script, so a page without scripts offers Download only


def test_the_embedded_photo_is_downscaled():
    data = re.search(r'href="data:image/jpeg;base64,([^"]+)"', _render()).group(1)
    preview = Image.open(io.BytesIO(base64.b64decode(data)))
    assert preview.size == (PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE // 2)
