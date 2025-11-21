"""Configuration for the Passport Photo Validator."""
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Tuple


class ValidatorConfig(BaseSettings):
    """
    Configuration for the PassportPhotoValidator, loadable from environment variables.
    Pydantic automatically matches snake_case fields to UPPER_CASE env vars.
    """
    min_detection_confidence: float = Field(
        default=0.5,
        description="Minimum confidence for face detection (0.0-1.0)."
    )
    target_face_ratio: float = Field(
        default=0.4,
        description="The target face height ratio for crop suggestions (e.g., 0.4 for 40%)."
    )
    target_aspect_ratio: float = Field(
        default=3 / 4,
        description="The target width/height aspect ratio for crop suggestions (e.g., 3/4 for passport)."
    )
    distortion_error_threshold: float = Field(
        default=30.0,
        description="BRISQUE score above this is considered a distortion error."
    )
    distortion_warning_threshold: float = Field(
        default=10.0,
        description="BRISQUE score above this is considered a distortion warning."
    )
    min_res_width: int = Field(
        default=600,
        description="The minimum allowed image width."
    )
    min_res_height: int = Field(
        default=600,
        description="The minimum allowed image height."
    )
    recommended_res_width: int = Field(
        default=900,
        description="The recommended minimum image width."
    )
    recommended_res_height: int = Field(
        default=900,
        description="The recommended minimum image height."
    )

    class Config:
        # Optional: if you want to support a .env file
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False

    @property
    def min_res(self) -> Tuple[int, int]:
        return self.min_res_width, self.min_res_height

    @property
    def recommended_res(self) -> Tuple[int, int]:
        return self.recommended_res_width, self.recommended_res_height

