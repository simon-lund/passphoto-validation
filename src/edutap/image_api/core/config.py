"""Configuration for the Passport Photo Validator."""
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, Tuple


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
        default=413,
        description="The minimum allowed image width (35 mm at 300 dpi)."
    )
    min_res_height: int = Field(
        default=531,
        description="The minimum allowed image height (45 mm at 300 dpi)."
    )
    recommended_res_width: int = Field(
        default=827,
        description="The recommended minimum image width (35 mm at 600 dpi)."
    )
    recommended_res_height: int = Field(
        default=1063,
        description="The recommended minimum image height."
    )

    autocrop: bool = Field(
        default=True,
        description="Plan a 35x45 crop for photos whose framing checks warn or fail, and check the cropped photo "
                    "(runs the checks a second time on those photos). The verdict never depends on it."
    )

    ofiq_config_dir: Optional[str] = Field(
        default=None,
        description="OFIQ's data folder (holds ofiq_config.jaxn and models/). Without it the OFIQ checks are skipped."
    )
    ofiq_config_file: str = Field(
        default='ofiq_config.jaxn',
        description="OFIQ config file name inside ofiq_config_dir (e.g. one that limits OFIQ's threads)."
    )
    siglip_enabled: bool = Field(
        default=False,
        description="Run the SigLIP second opinions on the eyes (needs the siglip extra and OFIQ; see core/siglip.py)."
    )
    siglip_threads: int = Field(
        default=4,
        description="PyTorch threads for SigLIP; with several validators in parallel, keep threads x workers at the core count."
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

