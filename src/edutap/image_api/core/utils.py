import cv2
import numpy as np
from PIL import ImageOps
from PIL.Image import Image


def convert_to_bgr(image: Image) -> np.ndarray:
    """
    Convert PIL Image to BGR color format for OpenCV compatibility.

    OpenCV uses BGR (Blue-Green-Red) color channel ordering by default, while
    PIL/Pillow uses RGB (Red-Green-Blue). This function handles the conversion
    and ensures compatibility between the two libraries.
    """
    image_np = np.array(image)

    # Grayscale image
    if len(image_np.shape) == 2:
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    # RGB image
    elif image_np.shape[2] == 3:
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # RGBA image
    elif image_np.shape[2] == 4:
        return cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    else:
        raise ValueError("Unsupported image format")


def prepare_image(image: Image) -> np.ndarray:
    """
    Turn a freshly opened photo into the BGR array the validator expects.

    Applies the EXIF rotation (phone photos store it there, so the face may otherwise lie on its side)
    and converts palette, CMYK and other modes to RGB first.
    """
    image = ImageOps.exif_transpose(image)
    if image.mode not in ('L', 'RGB', 'RGBA'):
        image = image.convert('RGB')
    return convert_to_bgr(image)
