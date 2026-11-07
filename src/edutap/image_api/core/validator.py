"""Passport photo validation business logic."""
import logging
from typing import Callable, Dict, List, Tuple

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
from .ofiq import load_engine
from .rules import evaluate_rules

logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation


class PassportPhotoValidator:
    """Validates passport photos against ICAO/ISO 19794-5 standards."""

    def __init__(self, config: ValidatorConfig = None):
        """Initialize validator with configuration.

        :param config: A ValidatorConfig object. If None, a default config is created,
                       loading values from environment variables.
        """
        self.config = config or ValidatorConfig()

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

    def _validate(self, image_np: np.ndarray) -> PassportPhotoReport:
        context = ValidationContext(image_np=image_np)

        # 1. Detect faces and check for exclusion criteria (no faces or multiple faces)
        detection_results = self._detect_faces(context)
        detections = detection_results.detections if detection_results else None

        if not detections:
            gate = ValidationIssue(category='error', message='No face detected', severity=10, check='face_detected')
            return self._generate_report(context, self._gate_failed(context, gate, 'No face detected.'))

        if len(detections) > 1:
            gate = ValidationIssue(
                category='error',
                message=f'Multiple faces detected ({len(detections)}). Only one person allowed.',
                severity=10,
                check='face_detected'
            )
            # We can still populate face_data for the first face to show its bounding box in the report
            self._populate_face_data(context, detections[0])
            return self._generate_report(context, self._gate_failed(context, gate, 'More than one face detected.'))

        # Prefill context
        self._populate_face_data(context, detections[0])
        self._populate_segmentation_data(context)
        self._populate_ofiq_data(context)
        self._populate_siglip_data(context)

        # Run checks
        # Note: Checks update the context with issues and measurements in place
        results = [CheckResult(name='face_detected', status='pass')]
        results += [self._run_check(context, name, check, kwargs) for name, check, kwargs in self._checks()]
        return self._generate_report(context, results)

    def _checks(self) -> List[Tuple[str, Callable, Dict]]:
        """All checks that run once exactly one face is found, as (name, function, keyword arguments).

        The names are part of the API contract (rules refer to them): add new ones, never rename.
        """
        distortion = dict(
            error_threshold=self.config.distortion_error_threshold,
            warning_threshold=self.config.distortion_warning_threshold,
        )
        return [
            ('resolution', quality.check_resolution,
             dict(min_res=self.config.min_res, recommended_res=self.config.recommended_res)),
            ('photo_format', geometry.check_photo_format, {}),
            ('face_size', geometry.check_face_size, {}),
            ('face_centered', geometry.check_face_centered, {}),
            ('face_sharpness', quality.check_face_sharpness, dict(warn_below=0.14, fail_below=0.11)),
            ('compression', quality.check_compression_blocks, dict(warn_above=1.6, fail_above=2.5)),
            ('exposure', lighting.check_exposure_level, dict(warn_below=75, fail_below=60)),
            ('contrast', lighting.check_contrast, {}),
            # Fails on dark backgrounds, warns on mid-tones: every Open Images background labelled dark is below 80,
            # none labelled light; the London Set's plain backgrounds are all above 190
            ('background_brightness', background.check_background_brightness, dict(min_brightness=120, fail_below=80)),
            ('background_uniformity', background.check_background_uniformity, {}),
            ('background_consistency', background.check_background_consistency, {}),
            # The London Set's plain backgrounds stay below 7 at 1063 px height (below 14 at 800 px); the faint UN
            # flag in Open Images measures 34 (background labels: openimages-faces/eval/vlm_background_labels.jsonl)
            ('background_pattern', background.check_background_pattern, dict(warn_above=10, fail_above=20)),
            # Calibrated on the London Set's compliant crops and synthetic defects built from them
            # (scripts/build_eval_set.py: bg_gradient, bg_shadow, ink_stain, smoothed). Compliant crops reach a
            # gradient of 16, a shadow depth of 9 (one photo; 98% stay under 5) and 2 marks; the defects start
            # at 41, 3 (90% above 5) and 3 marks. Filtered skin stays under 0.14, natural skin above 0.19
            ('background_gradient', background.check_background_gradient, dict(warn_above=18, fail_above=30)),
            ('background_shadow', background.check_background_shadow, dict(warn_above=5, fail_above=9)),
            ('print_defects', quality.check_print_defects, dict(fail_at=3)),
            ('skin_smoothing', retouching.check_skin_smoothing, dict(warn_below=0.15)),
            ('glare', features.check_glasses, {}),
            *self._ofiq_checks(),
            ('gaze', eyes.check_gaze, dict(warn_above=0.08, fail_above=0.15)),
            ('red_eyes', eyes.check_red_eyes, dict(fail_above=0.3)),
            ('glasses_tint', eyes.check_glasses_tint, dict(min_glasses_share=0.03, warn_below=0.55, fail_below=0.45)),
            ('glasses_glare', eyes.check_glasses_glare, dict(min_glasses_share=0.03, warn_above=0.01, fail_above=0.04)),
            # Second opinions, warn only. Thresholds for SigLIP SO400M: the score 80% of fine photos stay under,
            # on the calibrate half of the Open Images set (notebooks/real_photos.ipynb)
            ('siglip_glasses_glare', eyes.check_glasses_glare_siglip, dict(min_glasses_share=0.03, warn_above=-1.12)),
            ('siglip_gaze', eyes.check_gaze_siglip, dict(warn_above=-0.33)),
        ]

    @staticmethod
    def _ofiq_checks() -> List[Tuple[str, Callable, Dict]]:
        """The OFIQ checks. Quality thresholds are on OFIQ's 0-100 scale; pose thresholds in degrees.

        Calibrated on the London Set (notebooks/evaluation.ipynb); the eye thresholds also on real photos with
        glasses, hats and closed eyes from Open Images (scripts/score_labels.py). Not used, because they failed
        there: UnderExposurePrevention (scores well-lit dark skin as underexposed), Sharpness (drops from
        84 to 38 when a photo is shrunk by 3%), CompressionArtifacts (blind to JPEG quality 4 at ID-photo
        resolution) and DynamicRange (doesn't separate low-contrast photos).
        """
        def quality(measure_name, problem, warn_below, fail_below):
            return ofiq_checks.check_quality, dict(
                measure_name=measure_name, problem=problem, warn_below=warn_below, fail_below=fail_below)

        return [
            ('ofiq_head_yaw', ofiq_checks.check_pose, dict(axis='yaw', warn_above=5, fail_above=10)),
            ('ofiq_head_pitch', ofiq_checks.check_pose, dict(axis='pitch', warn_above=8, fail_above=12)),
            ('ofiq_head_roll', ofiq_checks.check_pose, dict(axis='roll', warn_above=6, fail_above=8)),
            ('ofiq_expression', *quality('ExpressionNeutrality', 'Expression is not neutral', 85, 50)),
            # The London Set's closed mouths never score below 94; 7 in 10 real open mouths score below 90
            ('ofiq_mouth_closed', *quality('MouthClosed', 'Mouth is open', 93, 90)),
            ('ofiq_eyes_open', *quality('EyesOpen', 'Eyes are not fully open', 95, 70)),
            ('ofiq_eyes_visible', *quality('EyesVisible', 'Eyes are covered (hair, frame or tinted lenses)', 60, 25)),
            ('ofiq_face_occlusion', *quality('FaceOcclusionPrevention', 'Part of the face is covered', 90, 70)),
            ('ofiq_mouth_occlusion', *quality('MouthOcclusionPrevention', 'The mouth is covered', 90, 70)),
            ('ofiq_head_covering', *quality('NoHeadCoverings', 'Head covering detected', 90, 50)),
            ('ofiq_background', *quality('BackgroundUniformity', 'Background is not uniform', 40, 25)),
            ('ofiq_illumination', *quality('IlluminationUniformity', 'Face is unevenly lit (shadows)', 72, 60)),
            ('ofiq_overexposure', *quality('OverExposurePrevention', 'Face is overexposed', 90, 50)),
            ('ofiq_natural_colour', *quality('NaturalColour', 'Colours look unnatural (colour cast)', 80, 50)),
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
        rules = evaluate_rules(checks)
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
        )

    def __del__(self):
        """Cleanup MediaPipe resources."""
        try:
            self.face_detection.close()
            self.face_mesh.close()
            self.selfie_segmentation.close()
        except AttributeError:
            pass
