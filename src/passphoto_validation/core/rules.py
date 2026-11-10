"""The German ID photo rules (Fotomustertafel) and which checks cover them.

Source: https://www.personalausweisportal.de/SharedDocs/downloads/Webs/PA/DE/informationsmaterial/weiterefuehrendes-material/Fotomustertafel.pdf

Every rule is listed, including the ones no check covers yet, so a report never hides a rule
it did not judge. Rule ids are part of the API contract: add new ones, never rename.
"""
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from .models import CheckResult, RuleResult

GROUPS: Dict[str, str] = {
    'format_and_size': 'Format and size',
    'sharpness_and_contrast': 'Sharpness and contrast',
    'lighting': 'Lighting',
    'background': 'Background',
    'photo_quality': 'Photo quality',
    'head_and_expression': 'Head position and facial expression',
    'eyes_and_glasses': 'Eyes and glasses',
    'head_coverings': 'Head coverings and clothing',
    'retouching': 'Retouching',
}


@dataclass(frozen=True)
class Rule:
    id: str
    group: str
    title: str
    checks: Tuple[str, ...] = ()
    coverage: Literal['full', 'partial', 'none'] = 'none'
    note: Optional[str] = None


RULES: List[Rule] = [
    # Format and size
    Rule('face_size', 'format_and_size', 'Head covers 70-80% of the photo height',
         ('face_size',), 'full',
         'Chin to crown. Hair cut off at the top edge is allowed; the crown is then estimated.'),
    Rule('face_centered', 'format_and_size', 'Face is centered',
         ('face_centered',), 'full'),
    Rule('template_zones', 'format_and_size', 'Chin, nose and eyes sit in the template zones',
         ('face_centered', 'face_size'), 'partial',
         'Checks centering, height in the frame and head size, not the exact zones of the template.'),
    Rule('face_height_mm', 'format_and_size', 'Head height is 32-36 mm (chin to crown)',
         ('face_size',), 'full',
         'Measured as a share of the photo height: 32-36 mm of a 45 mm photo is 71-80%.'),
    Rule('photo_format', 'format_and_size', 'Photo has the 35x45 mm format',
         ('photo_format',), 'full',
         'Width to height 7:9, within 2%.'),

    # Sharpness and contrast
    Rule('sharp', 'sharpness_and_contrast', 'Photo is sharp, not blurry',
         ('face_sharpness',), 'full',
         'Edge sharpness of the face, independent of exposure and resolution.'),
    Rule('contrast', 'sharpness_and_contrast', 'Contrast is good',
         ('contrast',), 'full'),
    Rule('sharpness_not_overexposed', 'sharpness_and_contrast', 'Not overexposed',
         ('ofiq_overexposure',), 'full'),

    # Lighting
    Rule('no_face_shadows', 'lighting', 'No shadows on the face',
         ('ofiq_illumination',), 'full',
         'Compares the lighting of the left and right half of the face; a shadow from above is not detected.'),
    Rule('no_reflections', 'lighting', 'No reflections',
         ('glare', 'glasses_glare'), 'partial',
         'Near-white spots in the upper face and on glasses lenses.'),
    Rule('lighting_not_overexposed', 'lighting', 'Not overexposed',
         ('ofiq_overexposure',), 'full'),
    Rule('not_underexposed', 'lighting', 'Not underexposed',
         ('exposure',), 'full',
         'Judged from the brightest parts of the photo (the light background), not from the skin, '
         'so dark skin is not mistaken for underexposure. A dark face in front of a bright background passes.'),
    Rule('no_red_eyes', 'lighting', 'No red eyes',
         ('red_eyes',), 'full'),

    # Background
    Rule('background_no_shadows', 'background', 'No shadows on the background',
         ('background_shadow',), 'full',
         'How much darker the background next to the person is than farther away, read in the band around the '
         'person and in the rim of the person mask, where the segmentation hides a shadow that touches the '
         'person. A darkening across the whole photo counts as a gradient (no_color_gradients) instead.'),
    Rule('background_no_objects', 'background', 'No objects in the background',
         ('background_uniformity', 'ofiq_background'), 'full'),
    Rule('background_plain', 'background', 'No patterned background (e.g. wallpaper)',
         ('ofiq_background', 'background_consistency', 'background_pattern'), 'full',
         'Faint patterns such as a flag are caught when they cover most of the visible background.'),
    Rule('background_contrast', 'background', 'Background contrasts with the face',
         ('background_brightness',), 'partial',
         'Only checks that the background is light: dark backgrounds fail, mid-tones warn. '
         'It is not compared with the face or hair.'),

    # Photo quality
    Rule('no_color_gradients', 'photo_quality', 'No color gradients',
         ('background_gradient',), 'partial',
         'The lightness change across the background (a plane fitted to it). A change of hue at the same '
         'lightness is not measured.'),
    Rule('no_print_errors', 'photo_quality', 'No errors such as ink stains',
         ('print_defects',), 'partial',
         'Spots, stains and scratches of up to about 4 mm on the background. Larger marks count as objects '
         '(background_no_objects); marks on the face or clothes are not told from features.'),
    Rule('no_pixelation', 'photo_quality', 'No pixelation',
         ('resolution', 'compression'), 'full',
         'Minimum resolution and JPEG block artefacts.'),
    Rule('natural_colour', 'photo_quality', 'Natural colours, no colour cast',
         ('ofiq_natural_colour', 'colour_cast'), 'full',
         'Skin colour in the CIELAB range of natural skin tones (ISO/IEC 29794-5), and no tint that the '
         'background and the whites of the eyes share.'),

    # Head position and facial expression
    Rule('neutral_expression', 'head_and_expression', 'Neutral expression: mouth closed, no big smile',
         ('ofiq_expression', 'ofiq_mouth_closed'), 'full',
         'Clear smiles are caught, and 8 in 10 open mouths on real photos; a faint closed-mouth smile often passes.'),
    Rule('no_head_tilt', 'head_and_expression', 'Head is not tilted',
         ('ofiq_head_roll',), 'full'),
    Rule('no_head_rotation', 'head_and_expression', 'Head is not turned',
         ('ofiq_head_yaw',), 'full'),
    Rule('no_half_profile', 'head_and_expression', 'No half profile',
         ('ofiq_head_yaw',), 'full'),
    Rule('facing_camera', 'head_and_expression', 'Face is centered, straight and facing the camera',
         ('face_centered', 'ofiq_head_roll', 'ofiq_head_pitch', 'ofiq_head_yaw'), 'full'),

    # Eyes and glasses
    Rule('eyes_open', 'eyes_and_glasses', 'Eyes are open',
         ('ofiq_eyes_open',), 'full',
         'Closed and squeezed-shut eyes fail; lids half-lowered over downcast eyes often pass.'),
    Rule('eyes_not_covered', 'eyes_and_glasses', 'Eyes are visible, not covered by hair',
         ('ofiq_eyes_visible',), 'full',
         'Hair that only reaches the upper eyelids passes.'),
    Rule('eyes_facing_camera', 'eyes_and_glasses', 'Eyes look into the camera',
         ('gaze', 'siglip_gaze'), 'partial',
         'Iris position within the eyes; on real photos it caught 3 in 5 averted gazes, with 1 in 5 false warnings. '
         'With SigLIP enabled, a second opinion catches 7 in 10, warning only.'),
    Rule('glasses_not_tinted', 'eyes_and_glasses', 'Glasses are not tinted',
         ('glasses_tint', 'ofiq_eyes_visible'), 'full',
         'Caught 9 in 10 real sunglasses and tinted glasses; lightly tinted lenses can pass.'),
    Rule('glasses_frame_clear', 'eyes_and_glasses', 'Glasses frame does not cover the eyes',
         ('ofiq_eyes_visible',), 'partial',
         'A frame across the eyes lowers OFIQ\'s eye visibility, but rarely far enough to warn.'),
    Rule('glasses_no_reflections', 'eyes_and_glasses', 'No reflections on glasses',
         ('glasses_glare', 'siglip_glasses_glare'), 'partial',
         'Blown-out white spots within the lenses miss most real reflections; with SigLIP enabled, the two '
         'together catch 4 in 10 on clear glasses, warning only. Passes when no glasses are worn.'),

    # Head coverings and clothing
    Rule('face_visible', 'head_coverings', 'Face is fully visible, not covered',
         ('face_detected', 'ofiq_face_occlusion', 'ofiq_mouth_occlusion'), 'full'),
    Rule('no_covering_shadows', 'head_coverings', 'No shadows from head coverings',
         ('ofiq_illumination',), 'partial',
         'Detects uneven lighting on the face, whatever its cause.'),
    Rule('no_hat', 'head_coverings', 'No hat or cap',
         ('ofiq_head_covering',), 'full',
         'Head covering pixels above the eyes (ISO/IEC 29794-5). Caught 9 in 10 real hats, with 8% false warnings.'),

    # Retouching
    Rule('no_retouching', 'retouching', 'Not retouched or filtered (e.g. Snapchat filters)',
         ('skin_smoothing', 'skin_colour_patches'), 'partial',
         'Skin smoothing, as beauty filters do: the cheeks and forehead lose their texture while the eyes stay '
         'sharp, or the whole face loses it while its edges stay. Coloured stickers or paint on the lower face '
         '(a filter\'s nose, hearts, butterflies) as patches of a colour that is not skin. Warnings only, '
         'calibrated on synthetic filters. Not detected: white or black marks (sparkles, drawn whiskers), '
         'stickers on the forehead, ears or crowns above the head (real hair accessories are allowed and look '
         'the same), a swapped background, reshaped features.'),
]

_RAN = ('pass', 'warn', 'fail')
_SEVERITY = {'pass': 0, 'warn': 1, 'fail': 2}


def evaluate_rules(checks: List[CheckResult]) -> List[RuleResult]:
    """Derive each rule's status from the checks that cover it.

    A rule gets the worst status of its checks that ran. If none of them ran, or one of them crashed
    and none failed, it is 'skipped': without every check's verdict the rule can't pass, while a fail
    stands on its own. A rule with no checks at all is 'not_checked'.
    """
    by_name = {check.name: check for check in checks}
    results = []

    for rule in RULES:
        covering = [by_name[name] for name in rule.checks if name in by_name]
        ran = [check for check in covering if check.status in _RAN]
        crashed = [check for check in covering if check.status == 'error']

        if not rule.checks:
            status, reasons = 'not_checked', []
        elif not ran or (crashed and not any(check.status == 'fail' for check in ran)):
            status = 'skipped'
            reasons = [reason for check in covering if check.status not in _RAN for reason in check.reasons]
            reasons = reasons or ['No covering check ran.']
        else:
            status = max((check.status for check in ran), key=_SEVERITY.__getitem__)
            reasons = [reason for check in ran if check.status != 'pass' for reason in check.reasons]

        results.append(RuleResult(
            id=rule.id,
            group=rule.group,
            title=rule.title,
            status=status,
            checks=list(rule.checks),
            coverage=rule.coverage,
            reasons=list(dict.fromkeys(reasons)),
            note=rule.note,
        ))

    return results
