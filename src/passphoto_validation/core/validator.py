"""Passport photo validation business logic."""
import logging
from typing import Callable, Dict, List, Set, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from .checks import background, eyes, features, geometry, lighting, quality, retouching
from .checks import ofiq as ofiq_checks
from . import siglip
from .config import ValidatorConfig
from .crop import FRAMING_CHECKS, plan_crop
from .models import CheckResult, CheckSkipped, Crop, PassportPhotoReport, ValidationIssue, ValidationContext
from .ofiq import PARSING_LABELS, load_engine
from .rules import evaluate_problems, evaluate_rules
from .thresholds import apply_overrides, disabled_in, is_off, level_is_off, load_overrides, parameters, thresholds

logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Checks that cannot measure once another check has failed. When the root (the key) fails, the checks listed
# under it are not run: each ends as skipped with the reason "Not judged: <the root's first reason>", a rule
# only such checks cover ends as skipped with that reason (the photo is not valid anyway, since the root
# fails), and the report's problem for the root lists those rules as not_judged. A root that warns, is
# skipped or crashes stops nothing. The root still runs in list order for the report, but it is measured
# before the first check that depends on it (see _run_checks). Names only: registered_checks() keeps the
# check list and its order, which are part of the API contract.
#
# Covered eyes (dark lenses, hair, a frame) leave the eye checks nothing to measure: gaze and red_eyes would
# read the lenses, ofiq_eyes_open lids it cannot see, glasses_glare and the SigLIP checks a lens surface,
# colour_cast the dark lenses as eye whites, and glasses_tint judges what already fails ofiq_eyes_visible.
CHECK_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    'ofiq_eyes_visible': ('ofiq_eyes_open', 'gaze', 'red_eyes', 'glasses_tint', 'glasses_glare',
                          'siglip_glasses_glare', 'siglip_gaze', 'colour_cast'),
}


def _not_judged(root: CheckResult) -> str:
    """The reason of a check left out because `root` failed."""
    return f'Not judged: {root.reasons[0] if root.reasons else root.name + " failed."}'


class PassportPhotoValidator:
    """Validates passport photos against ICAO/ISO 19794-5 standards."""

    def __init__(self, config: ValidatorConfig = None):
        """Initialize validator with configuration.

        :param config: A ValidatorConfig object. If None, a default config is created,
                       loading values from environment variables.
        """
        self.config = config or ValidatorConfig()
        # The checks and the keyword arguments they run with, resolved once: the thresholds table, then the
        # deployment's thresholds file on top (a bad file stops the validator here, not on the first photo).
        # Checks the file turns off are not run; the rules only they cover end as not_checked
        self._all_checks = self.registered_checks(self.config)
        self.disabled = self.disabled_checks(self.config)
        self._check_list = [check for check in self._all_checks if check[0] not in self.disabled]

        self.face_detection = mp_face_detection.FaceDetection(
            min_detection_confidence=self.config.min_detection_confidence
        )
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        )
        self.selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
        # Optional: without OFIQ its checks report skipped with this reason
        self.ofiq, self.ofiq_unavailable = load_engine(self.config.ofiq_config_dir, self.config.ofiq_config_file)
        self.siglip, self.siglip_unavailable = siglip.load_engine(self.config.siglip_enabled, self.config.siglip_threads)

    def validate(self, image_np: np.ndarray, crop: bool = True) -> PassportPhotoReport:
        """Run all validation checks on the provided image and return a report.

        :param image_np: Image as a numpy array (BGR format for OpenCV)
        :param crop: Also plan a 35x45 crop when a framing check warns or fails, and check the cropped
                     photo (report.crop). Needs config.autocrop too. The verdict never depends on it.
        :return: PassportPhotoReport with validation results
        """
        report = self._validate(image_np)
        if crop and self.config.autocrop:
            report.crop = self._crop(image_np, report)
        return report

    def _crop(self, image_np: np.ndarray, report: PassportPhotoReport) -> Crop:
        """Plan the crop for a photo with framing problems and check the cropped photo.

        Statuses: not_needed (no framing check warns or fails), not_possible (with the reason) or
        proposed (the box, the cropped photo's verdict and how each rule changes).
        """
        statuses = {check.name: check.status for check in report.checks}
        if not any(statuses.get(name) in ('warn', 'fail') for name in FRAMING_CHECKS):
            return Crop('not_needed')
        measurements = report.measurements
        if not report.bbox or 'chin_y' not in measurements or 'crown_y' not in measurements:
            return Crop('not_possible', reason='The head could not be measured, so no crop can be planned.')

        height, width = image_np.shape[:2]
        box, reason = plan_crop(width, height, report.bbox, measurements['chin_y'], measurements['crown_y'])
        if box is None:
            return Crop('not_possible', reason=reason)
        min_width, min_height = self.config.min_res
        if box.width < min_width or box.height < min_height:
            return Crop('not_possible', reason=f'The cropped photo would be {box.width}x{box.height} px; '
                                               f'it needs at least {min_width}x{min_height} px.')

        cropped = np.ascontiguousarray(image_np[box.y:box.y + box.height, box.x:box.x + box.width])
        after = self._validate(cropped)
        before = {rule.id: rule.status for rule in report.rules}
        return Crop(
            'proposed',
            x=box.x, y=box.y, width=box.width, height=box.height,
            is_valid=after.is_valid,
            rules_after={rule.id: rule.status for rule in after.rules},
            # Rules that warn or fail now and pass on the cropped photo
            fixed_rules=[rule.id for rule in after.rules if before[rule.id] in ('warn', 'fail') and rule.status == 'pass'],
            # Rules that keep the cropped photo from being valid; a crop can't fix them, they need a new photo
            remaining_rules=[rule.id for rule in after.rules if rule.status in ('fail', 'skipped')],
        )

    def inspect(self, image_np: np.ndarray, crop: bool = True) -> Tuple[PassportPhotoReport, ValidationContext]:
        """Like validate(), and also the ValidationContext of the photo as sent (its masks, landmarks and
        OFIQ data), for tools that draw what the checks saw. The crop's own context is not kept."""
        report, context = self._run(image_np)
        if crop and self.config.autocrop:
            report.crop = self._crop(image_np, report)
        return report, context

    def _validate(self, image_np: np.ndarray) -> PassportPhotoReport:
        return self._run(image_np)[0]

    def _run(self, image_np: np.ndarray) -> Tuple[PassportPhotoReport, ValidationContext]:
        context = ValidationContext(image_np=image_np)

        # 1. Detect faces and check for exclusion criteria (no faces or multiple faces)
        detection_results = self._detect_faces(context)
        detections = detection_results.detections if detection_results else None

        if not detections:
            gate = ValidationIssue(category='error', message='No face detected', severity=10, check='face_detected')
            return self._generate_report(context, self._gate_failed(context, gate, 'No face detected.')), context

        if len(detections) > 1:
            gate = ValidationIssue(
                category='error',
                message=f'Multiple faces detected ({len(detections)}). Only one person allowed.',
                severity=10,
                check='face_detected'
            )
            # We can still populate face_data for the first face to show its bounding box in the report
            self._populate_face_data(context, detections[0])
            return self._generate_report(context, self._gate_failed(context, gate, 'More than one face detected.')), context

        # Prefill context
        self._populate_face_data(context, detections[0])
        self._populate_segmentation_data(context)
        self._populate_ofiq_data(context)
        self._populate_siglip_data(context)

        # Run checks
        # Note: Checks update the context with issues and measurements in place
        results = [CheckResult(name='face_detected', status='pass')]
        results += self._run_checks(context)
        return self._generate_report(context, results), context

    def _run_checks(self, context: ValidationContext) -> List[CheckResult]:
        """Run every check and return the results in list order.

        A root check (CHECK_DEPENDENCIES) is measured before the first check that depends on it, so that
        check can be left out when the root has failed; the results, the issues and the measurements still
        come out in list order, as if nothing had moved.
        """
        checks = {name: (check, kwargs) for name, check, kwargs in self._checks()}
        root_of = {name: root for root, dependents in CHECK_DEPENDENCIES.items() for name in dependents}
        results: Dict[str, CheckResult] = {}

        def run(name: str) -> CheckResult:
            if name not in results:
                root = run(root_of[name]) if root_of.get(name) in checks else None
                if root is not None and root.status == 'fail':
                    results[name] = CheckResult(name=name, status='skipped', reasons=[_not_judged(root)])
                else:
                    check, kwargs = checks[name]
                    results[name] = self._run_check(context, name, check, kwargs)
            return results[name]

        for name in checks:
            run(name)
        ordered = [results[name] for name in checks]

        # A root measured early put its issues and measurements before its turn; restore the list order
        position = {name: i for i, name in enumerate(checks)}
        context.issues.sort(key=lambda issue: position.get(issue.check, -1))
        measurements = {}
        for result in ordered:
            for key in result.measurements:
                measurements.setdefault(key, context.measurements[key])
        measurements.update({key: value for key, value in context.measurements.items() if key not in measurements})
        context.measurements = measurements
        return ordered

    def _dependencies(self) -> Dict[str, Tuple[str, ...]]:
        """CHECK_DEPENDENCIES plus the face gate, which stops every other check; for the report's problems."""
        return {'face_detected': tuple(name for name, _, _ in self._checks()), **CHECK_DEPENDENCIES}

    def _checks(self) -> List[Tuple[str, Callable, Dict]]:
        """All checks that run once exactly one face is found, as (name, function, keyword arguments)."""
        return self._check_list

    def parameters(self) -> Dict[str, Dict]:
        """Every parameter of every check with the value this validator runs with, by check name, and whether
        the check is enabled. A level that is off reads False, as in the thresholds file; the open end of a
        range reads None (JSON has no infinity)."""
        def shown(name, key, value):
            if level_is_off(name, key, value):
                return False
            if isinstance(value, (tuple, list)):
                return [None if is_off(v) else v for v in value]
            return None if is_off(value) else value

        return {name: dict(enabled=name not in self.disabled,
                           **{key: shown(name, key, value) for key, value in parameters(check, kwargs).items()})
                for name, check, kwargs in self._all_checks}

    @staticmethod
    def disabled_checks(config: ValidatorConfig) -> Set[str]:
        """The checks the thresholds file of `config` turns off (`enabled = false`)."""
        if not config.thresholds_file:
            return set()
        return disabled_in(load_overrides(config.thresholds_file), source=config.thresholds_file)

    @classmethod
    def registered_checks(cls, config: ValidatorConfig) -> List[Tuple[str, Callable, Dict]]:
        """The checks that run once exactly one face is found, as (name, function, keyword arguments), with
        the thresholds file of `config` applied. Needs no models, so the CLI can list the parameters.

        The names are part of the API contract (rules refer to them): add new ones, never rename. The
        thresholds come from core/thresholds.py, which also says where each number came from; a check
        without an entry there runs with its function's defaults. CHECK_DEPENDENCIES says which of these
        checks are left out once another has failed.
        """
        checks = [
            ('resolution', quality.check_resolution,
             dict(min_res=config.min_res, recommended_res=config.recommended_res)),
            ('photo_format', geometry.check_photo_format, {}),
            ('face_size', geometry.check_face_size, {}),
            ('face_centered', geometry.check_face_centered, {}),
            ('face_sharpness', quality.check_face_sharpness, thresholds('face_sharpness')),
            ('compression', quality.check_compression_blocks, thresholds('compression')),
            ('exposure', lighting.check_exposure_level, thresholds('exposure')),
            ('contrast', lighting.check_contrast, thresholds('contrast')),
            ('colour_cast', lighting.check_colour_cast, thresholds('colour_cast')),
            ('background_brightness', background.check_background_brightness, thresholds('background_brightness')),
            ('background_uniformity', background.check_background_uniformity, {}),
            ('background_consistency', background.check_background_consistency, {}),
            ('background_pattern', background.check_background_pattern, thresholds('background_pattern')),
            ('background_gradient', background.check_background_gradient, thresholds('background_gradient')),
            ('background_shadow', background.check_background_shadow, thresholds('background_shadow')),
            ('print_defects', quality.check_print_defects, thresholds('print_defects')),
            ('skin_smoothing', retouching.check_skin_smoothing, thresholds('skin_smoothing')),
            ('skin_colour_patches', retouching.check_skin_colour_patches, thresholds('skin_colour_patches')),
            ('glare', features.check_glasses, {}),
            *cls._ofiq_checks(),
            ('gaze', eyes.check_gaze, thresholds('gaze')),
            ('red_eyes', eyes.check_red_eyes, thresholds('red_eyes')),
            ('glasses_tint', eyes.check_glasses_tint, thresholds('glasses_tint')),
            ('glasses_glare', eyes.check_glasses_glare, thresholds('glasses_glare')),
            # Second opinions, warn only
            ('siglip_glasses_glare', eyes.check_glasses_glare_siglip, thresholds('siglip_glasses_glare')),
            ('siglip_gaze', eyes.check_gaze_siglip, thresholds('siglip_gaze')),
        ]
        if config.thresholds_file:
            checks = apply_overrides(checks, load_overrides(config.thresholds_file), source=config.thresholds_file)
        return checks

    @staticmethod
    def _ofiq_checks() -> List[Tuple[str, Callable, Dict]]:
        """The OFIQ checks. Quality thresholds are on OFIQ's 0-100 scale; pose thresholds in degrees
        (core/thresholds.py).

        Not used, because they failed on the evaluation set: UnderExposurePrevention (scores well-lit dark
        skin as underexposed), Sharpness (drops from 84 to 38 when a photo is shrunk by 3%),
        CompressionArtifacts (blind to JPEG quality 4 at ID-photo resolution) and DynamicRange (doesn't
        separate low-contrast photos).
        """
        def pose(name, axis):
            return name, ofiq_checks.check_pose, dict(axis=axis, **thresholds(name))

        def quality(name, measure_name, problem):
            return name, ofiq_checks.check_quality, dict(measure_name=measure_name, problem=problem, **thresholds(name))

        return [
            pose('ofiq_head_yaw', 'yaw'),
            pose('ofiq_head_pitch', 'pitch'),
            pose('ofiq_head_roll', 'roll'),
            quality('ofiq_expression', 'ExpressionNeutrality', 'Expression is not neutral'),
            quality('ofiq_mouth_closed', 'MouthClosed', 'Mouth is open'),
            quality('ofiq_eyes_open', 'EyesOpen', 'Eyes are not fully open'),
            quality('ofiq_eyes_visible', 'EyesVisible', 'Eyes are covered (hair, frame or tinted lenses)'),
            quality('ofiq_face_occlusion', 'FaceOcclusionPrevention', 'Part of the face is covered'),
            quality('ofiq_mouth_occlusion', 'MouthOcclusionPrevention', 'The mouth is covered'),
            quality('ofiq_head_covering', 'NoHeadCoverings', 'Head covering detected'),
            quality('ofiq_background', 'BackgroundUniformity', 'Background is not uniform'),
            quality('ofiq_illumination', 'IlluminationUniformity', 'Face is unevenly lit (shadows)'),
            quality('ofiq_overexposure', 'OverExposurePrevention', 'Face is overexposed'),
            quality('ofiq_natural_colour', 'NaturalColour', 'Colours look unnatural (colour cast)'),
        ]

    def check_names(self) -> List[str]:
        """Names of every check a report can contain, in report order."""
        return ['face_detected'] + [name for name, _, _ in self._checks()]

    def _gate_failed(self, context: ValidationContext, gate: ValidationIssue, reason: str) -> List[CheckResult]:
        """Record a failed face gate and mark every other check as skipped for the same reason."""
        context.issues.append(gate)
        results = [CheckResult(name='face_detected', status='fail', reasons=[gate.message])]
        results += [CheckResult(name=name, status='skipped', reasons=[reason]) for name, _, _ in self._checks()]
        return results

    @staticmethod
    def _run_check(context: ValidationContext, name: str, check: Callable, kwargs: Dict) -> CheckResult:
        """Run one check and attribute the issues and measurements it adds to it."""
        issues_before = len(context.issues)
        measurements_before = dict(context.measurements)

        try:
            check(context, **kwargs)
        except CheckSkipped as skip:
            return CheckResult(name=name, status='skipped', reasons=[str(skip)])
        except Exception as exc:
            logger.exception('Check %s crashed', name)
            return CheckResult(name=name, status='error', reasons=[f'{type(exc).__name__}: {exc}'])

        new_issues = context.issues[issues_before:]
        for issue in new_issues:
            issue.check = name

        if any(issue.category == 'error' for issue in new_issues):
            status = 'fail'
        elif any(issue.category == 'warning' for issue in new_issues):
            status = 'warn'
        else:
            status = 'pass'

        measurements = {
            key: value for key, value in context.measurements.items()
            if measurements_before.get(key, object()) != value
        }
        return CheckResult(
            name=name,
            status=status,
            reasons=[issue.message for issue in new_issues if issue.category != 'info'],
            measurements=measurements,
        )

    def __call__(self, image_np: np.ndarray, crop: bool = True) -> PassportPhotoReport:
        """Allow the instance to be called directly (i.e., validator(image_np))."""
        return self.validate(image_np, crop=crop)

    def _populate_segmentation_data(self, context: ValidationContext):
        """
        Performs selfie segmentation and stores the masks in the context.
        """
        image_rgb = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2RGB)
        results = self.selfie_segmentation.process(image_rgb)

        if results.segmentation_mask is None:
            return

        context.segmentation_mask = results.segmentation_mask
        # Person is where the mask is > 0.5
        context.person_mask_binary = (results.segmentation_mask > 0.5).astype(np.uint8)

    def _populate_ofiq_data(self, context: ValidationContext):
        """Run OFIQ once on the photo and store its measures, landmarks and masks in the context."""
        if self.ofiq is None:
            context.ofiq_unavailable = self.ofiq_unavailable
            return
        try:
            context.ofiq = self.ofiq.assess(context.image_np)
        except Exception as exc:
            logger.exception('OFIQ failed on this photo')
            context.ofiq_unavailable = f'OFIQ failed on this photo: {exc}'
            return
        # OFIQ's face parsing labels the hair, hats and clothing the selfie segmentation leaves out of the
        # person, which the background checks would otherwise read as objects (two London Set people's hair).
        # Only hair and hats join, and only where they touch the segmented person: the parsing also labels an
        # object or a stain next to the person as clothing, and those must stay background
        parsing = context.ofiq.parsing
        mask = context.person_mask_binary
        if parsing is not None and mask is not None and parsing.shape == mask.shape:
            person = np.isin(parsing, (PARSING_LABELS['hair'], PARSING_LABELS['hat'])).astype(np.uint8)
            count, labels = cv2.connectedComponents(person, connectivity=8)
            touching = np.zeros(count, bool)
            touching[np.unique(labels[mask > 0])] = True
            touching[0] = False
            context.person_mask_binary = np.maximum(mask, touching[labels].astype(np.uint8))

    def _populate_siglip_data(self, context: ValidationContext):
        """Score the eye region once with SigLIP and store the scores in the context."""
        if self.siglip is None:
            context.siglip_unavailable = self.siglip_unavailable
            return
        # SigLIP looks at the eyes OFIQ found
        if context.ofiq is None:
            context.siglip_unavailable = f'SigLIP needs OFIQ\'s eye landmarks: {context.ofiq_unavailable}'
            return
        if context.ofiq.landmarks is None:
            context.siglip_unavailable = 'OFIQ found no eye landmarks for SigLIP.'
            return
        box = siglip.eye_box(context.ofiq.landmarks, context.width, context.height)
        if box is None:
            context.siglip_unavailable = 'The eye region is too small for SigLIP.'
            return
        try:
            eyes_rgb = Image.fromarray(np.ascontiguousarray(context.image_np[..., ::-1])).crop(box)
            context.siglip = self.siglip.scores(eyes_rgb)
        except Exception as exc:
            logger.exception('SigLIP failed on this photo')
            context.siglip_unavailable = f'SigLIP failed on this photo: {exc}'

    def _detect_faces(self, context: ValidationContext):
        """Detect faces in the image and return the detections."""
        image_rgb = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2RGB)
        return self.face_detection.process(image_rgb)

    def _populate_face_data(self, context: ValidationContext, detection: mp_face_detection.FaceDetection):
        """Populates the context with landmarks and bounding box for a given detection."""
        bbox_rel = detection.location_data.relative_bounding_box
        bbox_abs = {
            'x': int(bbox_rel.xmin * context.width),
            'y': int(bbox_rel.ymin * context.height),
            'width': int(bbox_rel.width * context.width),
            'height': int(bbox_rel.height * context.height)
        }

        # Run face mesh only after we confirm a single face
        image_rgb = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2RGB)
        mesh_results = self.face_mesh.process(image_rgb)
        landmarks = mesh_results.multi_face_landmarks[0] if mesh_results.multi_face_landmarks else None

        context.face_data = {
            'bbox': bbox_abs,
            'detection': detection,
            'landmarks': landmarks
        }

    def _generate_report(self, context: ValidationContext, checks: List[CheckResult]) -> PassportPhotoReport:
        """Generate the final validation report from the context."""
        bbox = context.face_data.get('bbox')
        suggested_crop = None
        if bbox:
            # If a face is found, calculate if a crop is needed to meet the target face size.
            face_height_ratio = bbox['height'] / context.height
            if face_height_ratio < self.config.target_face_ratio:
                new_height = int(bbox['height'] / self.config.target_face_ratio)
                new_width = int(new_height * self.config.target_aspect_ratio)

                # Center the new crop box around the detected face.
                crop_x = max(0, bbox['x'] + bbox['width'] // 2 - new_width // 2)
                crop_y = max(0, bbox['y'] + bbox['height'] // 2 - new_height // 2)

                # Ensure the crop box does not go outside the original image boundaries.
                crop_x = max(0, min(crop_x, context.width - new_width))
                crop_y = max(0, min(crop_y, context.height - new_height))
                new_width = min(new_width, context.width - crop_x)
                new_height = min(new_height, context.height - crop_y)

                suggested_crop = {
                    'x': crop_x, 'y': crop_y, 'width': new_width, 'height': new_height,
                    'reason': f'Face is too small. Crop to this region for a face height of ~{int(self.config.target_face_ratio * 100)}%.'
                }

        total_deduction = sum(issue.severity for issue in context.issues)
        score = max(0, 100 - total_deduction)
        rules = evaluate_rules(checks, disabled=self.disabled)
        problems = evaluate_problems(checks, rules, self._dependencies())
        # Valid only if no check found an error, none crashed, and every rule that has checks could be judged
        skipped_rules = [rule.id for rule in rules if rule.status == 'skipped']
        is_valid = (not skipped_rules and not any(issue.category == 'error' for issue in context.issues)
                    and not any(check.status == 'error' for check in checks))

        return PassportPhotoReport(
            is_valid=is_valid,
            overall_score=score,
            issues=context.issues,
            measurements=context.measurements,
            bbox=bbox,
            suggested_crop=suggested_crop,
            checks=checks,
            rules=rules,
            skipped_rules=skipped_rules,
            problems=problems,
        )

    def __del__(self):
        """Cleanup MediaPipe resources."""
        try:
            self.face_detection.close()
            self.face_mesh.close()
            self.selfie_segmentation.close()
        except AttributeError:
            pass
