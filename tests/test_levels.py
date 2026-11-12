"""One function grades every check: the fail level holds whatever the warn level is, and a level that is off
is never reached and never printed."""
import math
from types import SimpleNamespace

import numpy as np

from passphoto_validation.core.checks import eyes, geometry
from passphoto_validation.core.checks import ofiq as ofiq_checks
from passphoto_validation.core.levels import first_on, grade, is_on, judge, span
from passphoto_validation.core.models import ValidationContext


def _context(width=350, height=450):
    return ValidationContext(image_np=np.zeros((height, width, 3), np.uint8))


def test_grade_reads_each_direction():
    assert grade(5, warn_above=4, fail_above=8) == 'warn'
    assert grade(9, warn_above=4, fail_above=8) == 'fail'
    assert grade(3, warn_above=4, fail_above=8) == 'pass'
    assert grade(35, warn_below=40, fail_below=30) == 'warn'
    assert grade(29, warn_below=40, fail_below=30) == 'fail'
    assert grade(0.9, warn_below=0.7, warn_above=0.8, fail_below=0.6, fail_above=0.85) == 'fail'
    assert grade(0.82, warn_below=0.7, warn_above=0.8, fail_below=0.6, fail_above=0.85) == 'warn'
    # A count reaches its level at that number, a measure only past it
    assert grade(3, warn_at=1, fail_at=3) == 'fail' and grade(1, warn_at=1, fail_at=3) == 'warn'
    assert grade(8, warn_above=4, fail_above=8) == 'warn'


def test_the_fail_level_holds_when_the_warn_level_is_off():
    for off in (math.inf, None):
        assert grade(9, warn_above=off, fail_above=8) == 'fail'
        assert grade(5, warn_above=off, fail_above=8) == 'pass'
    assert grade(29, warn_below=-math.inf, fail_below=30) == 'fail'
    assert grade(100, warn_at=math.inf, fail_at=3) == 'fail'
    # And the other way round: fail off, warn on
    assert grade(9, warn_above=4, fail_above=math.inf) == 'warn'
    assert grade(9) == 'pass'


def test_judge_records_the_finding_of_the_level_reached():
    context = _context()
    assert judge(context, 9, warn_above=4, fail_above=8, fail=('far too much', 7), warn=('a little much', 3)) == 'fail'
    assert judge(context, 5, warn_above=4, fail_above=8, fail=('far too much', 7), warn=('a little much', 3)) == 'warn'
    assert judge(context, 1, warn_above=4, fail_above=8, fail=('far too much', 7), warn=('a little much', 3)) == 'pass'
    assert [(issue.category, issue.message, issue.severity) for issue in context.issues] == [
        ('error', 'far too much', 7), ('warning', 'a little much', 3)]


def test_messages_never_name_a_level_that_is_off():
    assert is_on(4) and not is_on(math.inf) and not is_on(-math.inf) and not is_on(None)
    assert first_on(math.inf, 8) == 8 and first_on(None, -math.inf) is None
    percent = dict(fmt=lambda share: f'{share * 100:.0f}', unit='%')
    assert span(0.7, 0.8, **percent) == '70-80%'
    assert span(0.7, math.inf, **percent) == 'at least 70%'
    assert span(-math.inf, 0.8, **percent) == 'at most 80%'
    assert span(None, None) == ''


def _pose(angle, **levels):
    context = _context()
    context.ofiq = SimpleNamespace(measures={'HeadPoseRoll': SimpleNamespace(assessed=True, native=angle, quality=50.0)})
    ofiq_checks.OfiqPose('ofiq_head_roll', axis='roll', **levels).measure(context)
    return [(issue.category, issue.message) for issue in context.issues]


def test_the_head_pose_names_the_limit_that_is_on():
    assert _pose(10.0, warn_above=4, fail_above=8) == [('error', 'Head is tilted sideways by 10.0° (max 4°).')]
    # With the warn level off the fail level is the limit; "max inf°" was printed before
    assert _pose(10.0, warn_above=math.inf, fail_above=8) == [('error', 'Head is tilted sideways by 10.0° (max 8°).')]
    assert _pose(6.0, warn_above=math.inf, fail_above=8) == []
    assert _pose(-6.0, warn_above=4, fail_above=math.inf) == [('warning', 'Head is slightly tilted sideways (6.0°).')]


def test_a_siglip_opinion_can_fail_with_its_warn_level_off():
    context = _context()
    context.siglip = {'gaze_aside': 2.0}
    eyes.SiglipGaze(warn_above=math.inf, fail_above=1.0).measure(context)
    assert [issue.category for issue in context.issues] == ['error']


def test_the_photo_format_can_warn_before_it_fails():
    def issues(width, height, **levels):
        context = _context(width, height)
        geometry.PhotoFormat(**levels).measure(context)
        return [issue.category for issue in context.issues]

    assert issues(350, 450) == []
    assert issues(360, 450) == ['error']  # 2.9% off the 7:9 ratio
    assert issues(354, 450, warn_tolerance=0.005) == ['warning']
    assert issues(360, 450, tolerance=math.inf) == []


def test_the_vertical_position_can_fail():
    def issues(centre_y, **levels):
        context = _context()
        context.face_data = {'bbox': dict(x=100, y=centre_y - 100, width=150, height=200)}
        geometry.FaceCentered(**levels).measure(context)
        return [(issue.category, issue.message) for issue in context.issues]

    assert issues(270) == []
    assert issues(360) == [('warning', 'Face is positioned too low in the frame.')]
    assert issues(360, fail_vertical_range=(0.3, 0.75)) == [('error', 'Face is positioned far too low in the frame.')]
    assert issues(150, fail_vertical_range=(0.4, 0.75)) == [('error', 'Face is positioned far too high in the frame.')]
    assert issues(360, vertical_range=(-math.inf, math.inf)) == []


def test_the_siglip_glare_opinion_can_fail_with_its_warn_level_off():
    # Glasses are found in OFIQ's face parsing, so the stand-in carries a parsing map and the eye landmarks
    from passphoto_validation.core.ofiq import LANDMARK_LEFT_EYE, LANDMARK_RIGHT_EYE, PARSING_LABELS
    context = _context()
    parsing = np.zeros((context.height, context.width), np.uint8)
    parsing[180:220, 100:250] = PARSING_LABELS['glasses']
    landmarks = np.zeros((98, 2), np.float32)
    landmarks[list(LANDMARK_RIGHT_EYE)] = (130, 200)
    landmarks[list(LANDMARK_LEFT_EYE)] = (220, 200)
    context.ofiq = SimpleNamespace(parsing=parsing, landmarks=landmarks)
    context.siglip = {'glare': 2.0}
    eyes.SiglipGlassesGlare(min_glasses_share=0.03, warn_above=math.inf, fail_above=1.0).measure(context)
    assert [issue.category for issue in context.issues] == ['error']
    # Both levels on: between them it warns
    context.issues.clear()
    eyes.SiglipGlassesGlare(min_glasses_share=0.03, warn_above=1.5, fail_above=3.0).measure(context)
    assert [issue.category for issue in context.issues] == ['warning']
