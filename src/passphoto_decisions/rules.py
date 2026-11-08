"""The German ID photo rules (Fotomustertafel), each as one yes/no question for the Decisions API.

The ids, groups and titles are the ones passphoto-validation uses, so results from both can be laid side by
side. Every question asks whether the DEFECT is present: the API's `probability` is then the probability that
the rule is broken, and one threshold pair (warn, fail) turns it into a status.

Source of the rules: https://www.personalausweisportal.de/SharedDocs/downloads/Webs/PA/DE/informationsmaterial/weiterefuehrendes-material/Fotomustertafel.pdf
"""
from dataclasses import dataclass
from typing import Dict, List

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

# The gate: without exactly one usable face no rule can be judged. Also phrased as a defect.
GATE_NAME = 'face_detected'
GATE_QUESTION = ('Is there NOT exactly one person\'s face in the photo: no face at all, more than one face, or only '
                 'a tiny or mostly cut-off face?')


@dataclass(frozen=True)
class Rule:
    id: str
    group: str
    title: str
    question: str  # yes means the rule is broken


RULES: List[Rule] = [
    # Format and size
    Rule('face_size', 'format_and_size', 'Head covers 70-80% of the photo height',
         'Is the head, from the chin to the top of the hair, smaller than 70% or larger than 80% of the photo\'s '
         'height?'),
    Rule('face_centered', 'format_and_size', 'Face is centered',
         'Is the face noticeably off-centre horizontally, or placed too high or too low in the frame?'),
    Rule('template_zones', 'format_and_size', 'Chin, nose and eyes sit in the template zones',
         'Do the eyes, nose or chin sit outside the zones of the German ID photo template (face centred, eyes in '
         'the upper half, head filling 70-80% of the height)?'),
    Rule('face_height_mm', 'format_and_size', 'Head height is 32-36 mm (chin to crown)',
         'Would the head height fall outside 32-36 mm on a 35x45 mm print, that is outside 71-80% of the photo '
         'height?'),
    Rule('photo_format', 'format_and_size', 'Photo has the 35x45 mm format',
         'Does the photo have an aspect ratio other than 35:45 (width to height 7:9, portrait), for example '
         'square or landscape?'),

    # Sharpness and contrast
    Rule('sharp', 'sharpness_and_contrast', 'Photo is sharp, not blurry',
         'Is the photo blurry or out of focus, so that the face lacks sharp detail?'),
    Rule('contrast', 'sharpness_and_contrast', 'Contrast is good',
         'Is the photo low in contrast, flat or washed out?'),
    Rule('sharpness_not_overexposed', 'sharpness_and_contrast', 'Not overexposed',
         'Is the photo overexposed, with blown-out bright skin or lost detail in the bright areas?'),

    # Lighting
    Rule('no_face_shadows', 'lighting', 'No shadows on the face',
         'Are there shadows on the face from uneven lighting, for example one side darker, or shadows under the '
         'nose or in the eye sockets?'),
    Rule('no_reflections', 'lighting', 'No reflections',
         'Are there reflections or glare spots on the face or on glasses?'),
    Rule('lighting_not_overexposed', 'lighting', 'Not overexposed',
         'Is the photo overexposed, with blown-out bright skin or lost detail in the bright areas?'),
    Rule('not_underexposed', 'lighting', 'Not underexposed',
         'Is the photo underexposed, too dark overall, with the face in darkness?'),
    Rule('no_red_eyes', 'lighting', 'No red eyes',
         'Do the pupils show the red-eye effect?'),

    # Background
    Rule('background_no_shadows', 'background', 'No shadows on the background',
         'Does a shadow of the person fall on the background?'),
    Rule('background_no_objects', 'background', 'No objects in the background',
         'Are there objects, furniture, other people or visible edges and corners in the background?'),
    Rule('background_plain', 'background', 'No patterned background (e.g. wallpaper)',
         'Is the background patterned or textured, for example wallpaper, bricks, tiles or fabric folds, rather '
         'than plain?'),
    Rule('background_contrast', 'background', 'Background contrasts with the face',
         'Does the background fail to contrast with the face and hair, for example a dark, mid-grey or '
         'skin-coloured background?'),

    # Photo quality
    Rule('no_color_gradients', 'photo_quality', 'No color gradients',
         'Does the background show a colour or brightness gradient, lighter on one side than the other?'),
    Rule('no_print_errors', 'photo_quality', 'No errors such as ink stains',
         'Are there ink stains, spots, scratches, dust or other print or scan defects on the photo?'),
    Rule('no_pixelation', 'photo_quality', 'No pixelation',
         'Is the photo pixelated, of low resolution, or showing compression artefacts such as blocky patches?'),
    Rule('natural_colour', 'photo_quality', 'Natural colours, no colour cast',
         'Do the colours look unnatural, with a colour cast such as yellow, blue, green or magenta skin?'),

    # Head position and facial expression
    Rule('neutral_expression', 'head_and_expression', 'Neutral expression: mouth closed, no big smile',
         'Is the expression not neutral: smiling, laughing, mouth open, frowning or grimacing?'),
    Rule('no_head_tilt', 'head_and_expression', 'Head is not tilted',
         'Is the head tilted sideways, so that the eyes are not on a horizontal line?'),
    Rule('no_head_rotation', 'head_and_expression', 'Head is not turned',
         'Is the head turned to the left or right instead of facing straight at the camera?'),
    Rule('no_half_profile', 'head_and_expression', 'No half profile',
         'Is the face shown in half profile or profile rather than frontally?'),
    Rule('facing_camera', 'head_and_expression', 'Face is centered, straight and facing the camera',
         'Is the face not straight on to the camera: turned, tilted, nodded up or down, or off-centre?'),

    # Eyes and glasses
    Rule('eyes_open', 'eyes_and_glasses', 'Eyes are open',
         'Are the eyes closed, nearly closed or squinting?'),
    Rule('eyes_not_covered', 'eyes_and_glasses', 'Eyes are visible, not covered by hair',
         'Are the eyes covered or partly hidden by hair, a hand or anything else?'),
    Rule('eyes_facing_camera', 'eyes_and_glasses', 'Eyes look into the camera',
         'Is the person looking away from the camera rather than straight into the lens?'),
    Rule('glasses_not_tinted', 'eyes_and_glasses', 'Glasses are not tinted',
         'Is the person wearing tinted glasses or sunglasses that darken the eyes?'),
    Rule('glasses_frame_clear', 'eyes_and_glasses', 'Glasses frame does not cover the eyes',
         'Does a glasses frame cross or cover the eyes, with the rim running through the eye area?'),
    Rule('glasses_no_reflections', 'eyes_and_glasses', 'No reflections on glasses',
         'Are there reflections or glare on glasses lenses that hide parts of the eyes?'),

    # Head coverings and clothing
    Rule('face_visible', 'head_coverings', 'Face is fully visible, not covered',
         'Is part of the face covered by a hand, a mask, a scarf, hair, a hat brim or an object?'),
    Rule('no_covering_shadows', 'head_coverings', 'No shadows from head coverings',
         'Does a hat, a head covering or the hair cast a shadow on the face?'),
    Rule('no_hat', 'head_coverings', 'No hat or cap',
         'Is the person wearing a hat, a cap or another head covering?'),

    # Retouching
    Rule('no_retouching', 'retouching', 'Not retouched or filtered (e.g. Snapchat filters)',
         'Does the photo look retouched or filtered: smoothed skin, altered features, stickers, cartoon ears or '
         'other AR overlays?'),
]

RULES_BY_ID: Dict[str, Rule] = {rule.id: rule for rule in RULES}
