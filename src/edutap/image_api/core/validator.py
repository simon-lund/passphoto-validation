"""Passport photo validation business logic."""

import cv2
import mediapipe as mp
import numpy as np

from .checks import background, features, geometry, lighting, quality
from .config import ValidatorConfig
from .models import PassportPhotoReport, ValidationIssue, ValidationContext

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

        # Allow the instance to be called directly (i.e., validator(image_np))
        self.__call__ = self.validate

    def validate(self, image_np: np.ndarray) -> PassportPhotoReport:
        """Run all validation checks on the provided image and return a report.

        :param image_np: Image as a numpy array (BGR format for OpenCV)
        :return: PassportPhotoReport with validation results
        """
        context = ValidationContext(image_np=image_np)

        # 1. Detect faces and check for exclusion criteria (no faces or multiple faces)
        detection_results = self._detect_faces(context)

        if not detection_results or not detection_results.detections:
            context.issues.append(ValidationIssue(category='error', message='No face detected', severity=10))
            return self._generate_report(context)

        if len(detection_results.detections) > 1:
            context.issues.append(ValidationIssue(
                category='error',
                message=f'Multiple faces detected ({len(detection_results.detections)}). Only one person allowed.',
                severity=10
            ))
            # We can still populate face_data for the first face to show its bounding box in the report
            self._populate_face_data(context, detection_results.detections[0])
            return self._generate_report(context)

        # Prefill context
        self._populate_face_data(context, detection_results.detections[0])
        self._populate_segmentation_data(context)

        # Run checks
        # Note: Checks update the context with issues and measurements in place
        quality.check_resolution(
            context,
            min_res=self.config.min_res,
            recommended_res=self.config.recommended_res
        )
        quality.check_distortions(
            context,
            area='full',
            error_threshold=self.config.distortion_error_threshold,
            warning_threshold=self.config.distortion_warning_threshold
        )
        quality.check_distortions(
            context,
            area='face',
            error_threshold=self.config.distortion_error_threshold,
            warning_threshold=self.config.distortion_warning_threshold
        )

        background.check_background_brightness(context)
        background.check_background_texture(context)
        background.check_background_uniformity(context)
        background.check_background_consistency(context)
        background.check_background_clutter(context)

        lighting.check_lighting(context)
        lighting.check_shadows(context)

        geometry.check_face_position(context)
        geometry.check_head_pose(context)

        features.check_face_features(context)
        features.check_expression(context)
        features.check_hair_obscuring(context)
        features.check_glasses(context)

        return self._generate_report(context)

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

    def _generate_report(self, context: ValidationContext) -> PassportPhotoReport:
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
        is_valid = not any(issue.category == 'error' for issue in context.issues)

        return PassportPhotoReport(
            is_valid=is_valid,
            overall_score=score,
            issues=context.issues,
            measurements=context.measurements,
            bbox=bbox,
            suggested_crop=suggested_crop
        )

    def __del__(self):
        """Cleanup MediaPipe resources."""
        try:
            self.face_detection.close()
            self.face_mesh.close()
            self.selfie_segmentation.close()
        except AttributeError:
            pass
