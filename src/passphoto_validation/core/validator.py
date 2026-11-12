"""Passport photo validation business logic."""
import logging
from typing import Dict, List, Optional, Set, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from . import siglip
from .check import Check, Sequence
from .checks import default_checks
from .config import ValidatorConfig
from .crop import FRAMING_CHECKS, plan_crop
from .models import CheckResult, Crop, PassportPhotoReport, ValidationIssue, ValidationContext
from .ofiq import PARSING_LABELS, load_engine
from .rules import evaluate_problems, evaluate_rules
from .thresholds import apply_overrides, disabled_in, is_off, level_is_off, load_overrides, parameters

logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Which checks a failed check stops, by name: {the check that failed: the checks not run after it}. Each check
# declares it itself (`stops_after`, core/check.py); this is the default sequence's declarations read the
# other way round, for the report's problems and for the tests. A stopped check ends as skipped with the
# reason "Not judged: <the other's first reason>", a rule only such checks cover ends as skipped with that
# reason (the photo is not valid anyway, since the first check fails), and the report's problem for the first
# check lists those rules as not_judged. A check that warns, is skipped or crashes stops nothing.
CHECK_DEPENDENCIES: Dict[str, Tuple[str, ...]] = default_checks().dependencies()


class PassportPhotoValidator:
    """Validates passport photos against ICAO/ISO 19794-5 standards."""

    def __init__(self, config: ValidatorConfig = None, checks: Optional[Sequence] = None):
        """Initialize validator with configuration.

        :param config: A ValidatorConfig object. If None, a default config is created,
                       loading values from environment variables.
        :param checks: The sequence of checks to run instead of the default one (core/checks: default_checks),
                       e.g. `default_checks() + [MyCheck()]`. The configuration and the thresholds file are
                       applied to a copy of it.
        """
        self.config = config or ValidatorConfig()
        # The sequence of checks with the values they run with, resolved once: the defaults of the classes,
        # the thresholds table, the configuration, then the deployment's thresholds file on top (a bad file
        # stops the validator here, not on the first photo). Checks the file turns off stay in the sequence
        # but are not run; the rules only they cover end as not_checked
        self.checks = self.registered_checks(self.config, checks)

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
        results += self.checks.run(context)
        return self._generate_report(context, results), context

    def _dependencies(self) -> Dict[str, Tuple[str, ...]]:
        """Which checks a failed check stops, plus the face gate, which stops every check; for the report's problems."""
        return {'face_detected': tuple(check.name for check in self._checks()), **self.checks.dependencies()}

    def _checks(self) -> List[Check]:
        """The checks that run once exactly one face is found: the sequence without the ones turned off."""
        return self.checks.enabled()

    @property
    def disabled(self) -> Set[str]:
        """The names of the checks the thresholds file turns off."""
        return self.checks.disabled()

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

        return {check.name: dict(enabled=check.enabled,
                                 **{key: shown(check.name, key, value) for key, value in parameters(check).items()})
                for check in self.checks}

    @staticmethod
    def disabled_checks(config: ValidatorConfig) -> Set[str]:
        """The checks the thresholds file of `config` turns off (`enabled = false`)."""
        if not config.thresholds_file:
            return set()
        return disabled_in(load_overrides(config.thresholds_file), source=config.thresholds_file)

    @classmethod
    def registered_checks(cls, config: ValidatorConfig, checks: Optional[Sequence] = None) -> Sequence:
        """The sequence of checks with the values a validator of `config` runs with. Needs no models, so the
        CLI can list the parameters.

        Each layer overrides the one before: the defaults of the check classes and the table in
        core/thresholds.py (both already in `default_checks()`), the resolution limits of `config`, then the
        thresholds file of `config`. `checks` replaces the default sequence and is left as it was given. A
        parameter that has no value after all that stops the validator here, not on the first photo.
        """
        checks = checks.copy() if checks is not None else default_checks()
        if 'resolution' in checks:
            checks.load({'resolution': dict(min_res=config.min_res, recommended_res=config.recommended_res)})
        if config.thresholds_file:
            checks = apply_overrides(checks, load_overrides(config.thresholds_file), source=config.thresholds_file)
        return checks.complete()

    def check_names(self) -> List[str]:
        """Names of every check a report can contain, in report order."""
        return ['face_detected'] + [check.name for check in self._checks()]

    def _gate_failed(self, context: ValidationContext, gate: ValidationIssue, reason: str) -> List[CheckResult]:
        """Record a failed face gate and mark every other check as skipped for the same reason."""
        context.issues.append(gate)
        results = [CheckResult(name='face_detected', status='fail', reasons=[gate.message])]
        results += [CheckResult(name=check.name, status='skipped', reasons=[reason]) for check in self._checks()]
        return results

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
