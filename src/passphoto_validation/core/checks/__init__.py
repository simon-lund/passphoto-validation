"""The checks, and the sequence the validator runs them in.

Each module holds the checks of one kind as classes (core/check.py says what a check is made of);
`default_checks` is the one place that lists them, in the order of the report.
"""
from ..check import Sequence
from ..thresholds import THRESHOLDS
from . import background, eyes, features, geometry, lighting, quality, retouching
from . import ofiq

__all__ = ["background", "eyes", "features", "geometry", "lighting", "ofiq", "quality", "retouching", "default_checks"]

EYES_COVERED = ('ofiq_eyes_visible',)


def default_checks() -> Sequence:
    """The checks that run once exactly one face is found, with the thresholds of core/thresholds.py.

    The order is the order of every report, and the names are part of the API contract (rules, thresholds
    files and stored results refer to them): add checks, never rename one. The numbers are not written
    here: a check starts with the defaults of its class, the table in core/thresholds.py is loaded on top
    (with a note on where each number came from), then `ValidatorConfig` and the deployment's thresholds file
    (PassportPhotoValidator.registered_checks).

    OFIQ's measures are two classes used under several names. Its quality scores are on a 0-100 scale, the
    pose in degrees. Not used, because they failed on the evaluation set: UnderExposurePrevention (scores
    well-lit dark skin as underexposed), Sharpness (drops from 84 to 38 when a photo is shrunk by 3%),
    CompressionArtifacts (blind to JPEG quality 4 at ID-photo resolution) and DynamicRange (doesn't separate
    low-contrast photos).

    Covered eyes (dark lenses, hair, a frame) leave the eye checks nothing to measure: gaze and red_eyes would
    read the lenses, ofiq_eyes_open lids it cannot see, glasses_glare and the SigLIP checks a lens surface,
    colour_cast the dark lenses as eye whites, and glasses_tint judges what already fails ofiq_eyes_visible.
    So each of them declares `stops_after = ('ofiq_eyes_visible',)`.
    """
    return Sequence(
        quality.Resolution(),
        geometry.PhotoFormat(),
        geometry.FaceSize(),
        geometry.FaceCentered(),
        quality.FaceSharpness(),
        quality.Compression(),
        lighting.Exposure(),
        lighting.Contrast(),
        lighting.ColourCast(),
        background.BackgroundBrightness(),
        background.BackgroundUniformity(),
        background.BackgroundConsistency(),
        background.BackgroundPattern(),
        background.BackgroundGradient(),
        background.BackgroundShadow(),
        quality.PrintDefects(),
        retouching.SkinSmoothing(),
        retouching.SkinColourPatches(),
        features.Glare(),
        ofiq.OfiqPose('ofiq_head_yaw', axis='yaw'),
        ofiq.OfiqPose('ofiq_head_pitch', axis='pitch'),
        ofiq.OfiqPose('ofiq_head_roll', axis='roll'),
        ofiq.OfiqQuality('ofiq_expression', measure_name='ExpressionNeutrality', problem='Expression is not neutral'),
        ofiq.OfiqQuality('ofiq_mouth_closed', measure_name='MouthClosed', problem='Mouth is open'),
        ofiq.OfiqQuality('ofiq_eyes_open', measure_name='EyesOpen', problem='Eyes are not fully open',
                         stops_after=EYES_COVERED),
        ofiq.OfiqQuality('ofiq_eyes_visible', measure_name='EyesVisible',
                         problem='Eyes are covered (hair, frame or tinted lenses)'),
        ofiq.OfiqQuality('ofiq_face_occlusion', measure_name='FaceOcclusionPrevention', problem='Part of the face is covered'),
        ofiq.OfiqQuality('ofiq_mouth_occlusion', measure_name='MouthOcclusionPrevention', problem='The mouth is covered'),
        ofiq.OfiqQuality('ofiq_head_covering', measure_name='NoHeadCoverings', problem='Head covering detected'),
        ofiq.OfiqQuality('ofiq_background', measure_name='BackgroundUniformity', problem='Background is not uniform'),
        ofiq.OfiqQuality('ofiq_illumination', measure_name='IlluminationUniformity', problem='Face is unevenly lit (shadows)'),
        ofiq.OfiqQuality('ofiq_overexposure', measure_name='OverExposurePrevention', problem='Face is overexposed'),
        ofiq.OfiqQuality('ofiq_natural_colour', measure_name='NaturalColour', problem='Colours look unnatural (colour cast)'),
        eyes.Gaze(),
        eyes.RedEyes(),
        eyes.GlassesTint(),
        eyes.GlassesGlare(),
        # Second opinions, warn only
        eyes.SiglipGlassesGlare(),
        eyes.SiglipGaze(),
    ).load({name: entry.kwargs for name, entry in THRESHOLDS.items()}).complete()
