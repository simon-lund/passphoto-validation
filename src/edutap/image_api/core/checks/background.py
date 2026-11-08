"""Background validation checks for passport photos."""
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from skimage import feature
from skimage.feature import local_binary_pattern

from ..models import CheckSkipped, ValidationIssue, ValidationContext

mp_selfie_segmentation = mp.solutions.selfie_segmentation


def _find_background_rectangles(mask, grid_sizes=(64, 128), erosion_kernel_size: int = 7,  erosion_iterations: int = 3, clean_region_threshold: float = 0.98):
    """
    Sample clean background mask using overlapping rectangular regions for localized analysis.

    This implements a high-overlap grid search to ensure robust statistical sampling
    while avoiding contamination from segmentation edge artifacts (via erosion).

    :param mask: The binary segmentation mask where the background is typically 1 (or high value).
    :param grid_sizes: A Sequence of side lengths (e.g., (64, 128)) to search for clean rectangles.
                       The function will sample rectangles of these sizes.
    :param erosion_kernel_size: The side length of the square kernel used for morphological erosion (e.g., 7).
                                This controls the radius of shrinkage per iteration (r = (k-1)/2, so k=7 => r=3 with 1 center pixel).
    :param erosion_iterations: The number of times the erosion is repeated (e.g., 3).
                               Total shrinkage is (kernel_size - 1) / 2 * iterations.
    :param clean_region_threshold: The minimum percentage of pixels (e.g., 0.98 for 98%) within a sampled
                                   rectangle that must be clean background for the rectangle to be accepted.
    :return: A list of rectangles defined as (x, y, w, h, area).
    """
    # Apply morphological erosion to shrink the subject's boundary.
    # This isolates a clean core of the background, avoiding segmentation contamination.
    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=erosion_iterations)

    rectangles = []
    h, w = eroded_mask.shape

    # Divide the image into a grid and find regions that are fully background
    for size in grid_sizes:
        if size > h or size > w:
            continue

        step = size // 2  # 50% overlap step

        # Iterates through rows and columns using the overlap step
        for y in range(0, h - size + 1, step):
            for x in range(0, w - size + 1, step):
                region = eroded_mask[y:y + size, x:x + size]

                # Ensure the region's percentage of clean background pixels exceeds the threshold
                if np.mean(region) > clean_region_threshold:
                    rectangles.append((x, y, size, size, size * size))

    return rectangles


def _sharpen_image(gray_image):
    """
    Sharpen a image.

    You can use this to enhance texture patterns in advance of edge detection.
    This helps to better distinguish cluttered backgrounds from smooth ones.
    """
    # Sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(gray_image, -1, kernel)
    return sharpened


def _calculate_lbp_variance(gray_image, rectangles):
    """
    Calculate Local Binary Pattern to assess texture across block-wise background regions.
    Higher variance in LBP indicates more texture/patterns.

    https://scikit-image.org/docs/0.25.x/auto_examples/features_detection/plot_local_binary_pattern.html

    :return: Weighted average of LBP variances across rectangles
    """
    lbp_variances = []
    weights = []

    for x, y, w, h, area in rectangles:
        region = gray_image[y:y + h, x:x + w]
        lbp = local_binary_pattern(region, P=8, R=1, method='uniform')

        # For uniform backgrounds, LBP should have low variance
        lbp_var = np.var(lbp)
        lbp_variances.append(lbp_var)
        weights.append(area)

    if lbp_variances:
        return np.average(lbp_variances, weights=weights)
    return 0.0


def _calculate_color_range(image, rectangles):
    """
    Calculate color range and within-rectangle variance.

    Returns (global_color_range, within_rect_variance_mean):
    - global_color_range: Span of colors across all rectangles
    - average_local_instability: Average color variance within rectangles
    """
    mean_colors = []
    instability_scores = []
    weights = []

    for x, y, w, h, area in rectangles:
        region = image[y:y + h, x:x + w]

        # Convert to L*a*b* (BGR is OpenCV's default)
        # L*a*b* is more perceptually uniform, i.e., changes correspond to human vision (L*a*b* = lightness scale + green-red scale + blue-yellow scale)
        lab_region = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)

        mean_color = np.mean(lab_region, axis=(0, 1))
        mean_colors.append(mean_color)

        # Calculate variances for all three channels across the rectangle (axis=(0,1) <=> height,width)
        variances = np.var(lab_region, axis=(0, 1))  # 3-element array: [l_var, a_var, b_var]
        variances_safe = np.nan_to_num(variances, nan=0.0)

        # Local color instability = sum of variances across L*a*b* channels
        instability_scores.append(variances_safe.sum())
        weights.append(area)

    if mean_colors:
        rect_mean_colors = np.array(mean_colors)
        global_color_range = np.ptp(rect_mean_colors, axis=0).sum()

        average_local_instability = np.average(instability_scores, weights=weights)

        return global_color_range, average_local_instability

    return 0.0, 0.0


def _calculate_color_coherence_cov(image, rectangles):
    """
    Calculate coefficient of variation in color differences between neighbors.
    Low CoV = consistent changes (gradient)
    High CoV = random changes (clutter)
    """
    if len(rectangles) < 2:
        return 0.0

    # Calculate mean colors and rectangle centers
    centers = []
    colors = []
    for x, y, w, h, area in rectangles:
        region = image[y:y + h, x:x + w]
        lab_region = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        colors.append(np.mean(lab_region, axis=(0, 1)))
        centers.append((x + w // 2, y + h // 2, max(w, h)))
    centers = np.array(centers, dtype=np.float64)
    colors = np.array(colors, dtype=np.float64)

    # Calculate color differences between neighbors (each pair once, vectorized per rectangle)
    color_diffs = []

    for i in range(len(centers) - 1):
        others = centers[i + 1:]
        distance = np.hypot(others[:, 0] - centers[i, 0], others[:, 1] - centers[i, 1])

        # Only neighbors
        max_neighbor_distance = np.maximum(others[:, 2], centers[i, 2]) * 2
        neighbors = distance < max_neighbor_distance

        # Color difference in L*a*b*, normalized by spatial distance
        color_diff = np.sqrt(np.sum((colors[i + 1:][neighbors] - colors[i]) ** 2, axis=1))
        color_diffs.append(color_diff / (distance[neighbors] + 1))

    color_diffs = np.concatenate(color_diffs) if color_diffs else np.array([])

    if len(color_diffs) > 1:
        # Coefficient of Variation
        mean_diff = np.mean(color_diffs)
        std_diff = np.std(color_diffs)

        # FIX: Only calculate CoV if mean is significant
        # If mean is very small, colors are very similar everywhere (uniform background)
        if mean_diff > 0.1:  # Threshold to avoid division by near-zero
            return std_diff / mean_diff
        else:
            # If mean diff is tiny, background is very uniform
            # Return a low value to indicate consistency
            return 0.0
    return 0.0


def _calculate_edge_density_rect(gray_image, rectangles):
    """
    Calculate edge density on rectangles using sharpened image.
    Sharpening enhances texture patterns for better detection.
    """
    # Sharpen the image first to enhance edges
    sharpened = _sharpen_image(gray_image)

    # Use Canny edge detection on the sharpened image
    edges = feature.canny(sharpened, sigma=2.0)

    edge_densities = []
    weights = []

    for x, y, w, h, area in rectangles:
        edge_region = edges[y:y + h, x:x + w]
        density = np.sum(edge_region) / edge_region.size
        edge_densities.append(density)
        weights.append(area)

    if edge_densities:
        weights = np.array(weights)
        weights = weights / weights.sum()
        return np.average(edge_densities, weights=weights)
    return 0.0


def check_background_brightness(context: ValidationContext, min_brightness: float = 80.0,
                                fail_below: Optional[float] = None, debug: bool = False):
    """
    Check if the background is sufficiently bright.

    :param context: The validation context with segmentation data.
    :param min_brightness: Brightness (0-255) below which the check warns.
    :param fail_below: Brightness below which the check fails; None never fails.
    :param debug: If True, returns visualization data.
    """
    if context.segmentation_mask is None:
        raise CheckSkipped("No segmentation mask present.")

    background_mask_binary = (context.segmentation_mask < 0.5).astype(np.uint8)

    if np.sum(background_mask_binary) == 0:
        raise CheckSkipped("No background visible.")

    gray_full = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray_full[background_mask_binary == 1])
    context.measurements['background_brightness'] = float(mean_brightness)

    if fail_below is not None and mean_brightness < fail_below:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'Background is dark (brightness: {mean_brightness:.2f}). It must be light, ideally grey.',
            severity=7
        ))
    elif mean_brightness < min_brightness:
        context.issues.append(ValidationIssue(
            category='warning',
            message=f'Background is quite dark (brightness: {mean_brightness:.2f}). Consider using a lighter background.',
            severity=4
        ))

    if debug:
        return {'brightness': mean_brightness}
    return None


def check_background_texture(context: ValidationContext, lbp_threshold: float = 50.0, debug: bool = False):
    """
    Check for texture patterns using Local Binary Patterns.

    :param context: The validation context with segmentation data.
    :param lbp_threshold: Maximum acceptable LBP variance.
    :param debug: If True, returns visualization data.
    """
    if context.segmentation_mask is None:
        raise CheckSkipped("No segmentation mask present.")

    background_mask_binary = (context.segmentation_mask < 0.5).astype(np.uint8)
    rectangles = _find_background_rectangles(background_mask_binary)

    if len(rectangles) < 3:
        raise CheckSkipped(f"Not enough clean background regions ({len(rectangles)}) for texture analysis.")

    gray_full = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2GRAY)
    avg_lbp_variance = _calculate_lbp_variance(gray_full, rectangles)
    context.measurements['background_avg_lbp_variance'] = float(avg_lbp_variance)
    context.measurements['background_rect_count'] = len(rectangles)

    if avg_lbp_variance > lbp_threshold:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'Background has texture patterns (LBP variance: {avg_lbp_variance:.2f}). Background must be plain.',
            severity=7
        ))

    if debug:
        kernel = np.ones((7, 7), np.uint8)
        eroded_mask_viz = cv2.erode(background_mask_binary, kernel, iterations=3) * 255
        rect_viz = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)
        for x, y, w, h, area in rectangles:
            cv2.rectangle(rect_viz, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return {
            'avg_lbp_variance': avg_lbp_variance,
            'rectangles': rectangles,
            'rectangles_visualization': rect_viz,
            'eroded_mask': eroded_mask_viz
        }
    return None


def check_background_uniformity(context: ValidationContext, variance_threshold: float = 200.0, debug: bool = False):
    """
    Check for local color uniformity within background regions.
    High within-rectangle variance indicates clutter.

    :param context: The validation context with segmentation data.
    :param variance_threshold: Maximum acceptable within-rectangle color variance.
    :param debug: If True, returns visualization data.
    """
    if context.segmentation_mask is None:
        raise CheckSkipped("No segmentation mask present.")

    background_mask_binary = (context.segmentation_mask < 0.5).astype(np.uint8)
    rectangles = _find_background_rectangles(background_mask_binary)

    if len(rectangles) < 3:
        raise CheckSkipped(f"Not enough clean background regions ({len(rectangles)}) for uniformity analysis.")

    overall_range, within_rect_variance = _calculate_color_range(context.image_np, rectangles)
    context.measurements['background_color_range'] = float(overall_range)
    context.measurements['background_within_rect_variance'] = float(within_rect_variance)
    context.measurements['background_rect_count'] = len(rectangles)

    if within_rect_variance > variance_threshold:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'Background has high local color variation (variance: {within_rect_variance:.2f}), indicating clutter.',
            severity=7
        ))

    if debug:
        return {
            'color_range': overall_range,
            'within_rect_variance': within_rect_variance
        }
    return None


def check_background_pattern(context: ValidationContext, warn_above: float, fail_above: float,
                             min_rectangles: int = 6, blur_sigma: float = 2.0, debug: bool = False):
    """
    Check for a pattern spread across the background (wallpaper, a flag, a printed backdrop).

    Measures the colour variance (sum over L*a*b*) within each clean background rectangle and takes the
    median. A pattern covers most rectangles; hair that strays into the background, or a single object,
    covers only some, so the median ignores them. A light blur first removes film grain and JPEG noise,
    which would otherwise read as texture on a plain wall.

    :param context: The validation context with segmentation data.
    :param warn_above: Median variance above this warns.
    :param fail_above: Median variance above this fails.
    :param min_rectangles: Fewer clean rectangles than this skip the check: the median of a handful of
                           rectangles, all close to the person, is mostly hair and shoulders.
    :param blur_sigma: Gaussian blur (pixels) applied before measuring.
    :param debug: If True, returns visualization data.
    """
    if context.segmentation_mask is None:
        raise CheckSkipped("No segmentation mask present.")

    background_mask_binary = (context.segmentation_mask < 0.5).astype(np.uint8)
    rectangles = _find_background_rectangles(background_mask_binary)

    if len(rectangles) < min_rectangles:
        raise CheckSkipped(f"Not enough clean background regions ({len(rectangles)}) for pattern analysis.")

    lab = cv2.cvtColor(cv2.GaussianBlur(context.image_np, (0, 0), blur_sigma), cv2.COLOR_BGR2LAB).astype(np.float32)
    variances = [lab[y:y + h, x:x + w].reshape(-1, 3).var(axis=0).sum() for x, y, w, h, _ in rectangles]
    median_variance = float(np.median(variances))
    context.measurements['background_median_rect_variance'] = median_variance
    context.measurements['background_rect_count'] = len(rectangles)

    if median_variance > fail_above:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'Background is patterned (median local variance: {median_variance:.1f}). Background must be plain.',
            severity=7
        ))
    elif median_variance > warn_above:
        context.issues.append(ValidationIssue(
            category='warning',
            message=f'Background may be patterned (median local variance: {median_variance:.1f}).',
            severity=4
        ))

    if debug:
        return {'median_rect_variance': median_variance, 'rect_variances': variances, 'rectangles': rectangles}
    return None


def check_background_consistency(context: ValidationContext, cov_threshold: float = 0.5, debug: bool = False):
    """
    Check for consistent color transitions using Coefficient of Variation.
    Smooth gradients have low CoV; cluttered backgrounds have high CoV. So a gradient passes here, and
    check_background_gradient reports it.

    :param context: The validation context with segmentation data.
    :param cov_threshold: Maximum acceptable coefficient of variation.
    :param debug: If True, returns visualization data.
    """
    if context.segmentation_mask is None:
        raise CheckSkipped("No segmentation mask present.")

    background_mask_binary = (context.segmentation_mask < 0.5).astype(np.uint8)
    rectangles = _find_background_rectangles(background_mask_binary)

    if len(rectangles) < 3:
        raise CheckSkipped(f"Not enough clean background regions ({len(rectangles)}) for consistency analysis.")

    color_coherence_cov = _calculate_color_coherence_cov(context.image_np, rectangles)
    context.measurements['background_color_coherence_cov'] = float(color_coherence_cov)
    context.measurements['background_rect_count'] = len(rectangles)

    if color_coherence_cov > cov_threshold:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'Background has inconsistent color transitions (CoV: {color_coherence_cov:.2f}), indicating non-uniform background.',
            severity=6
        ))

    if debug:
        return {'color_coherence_cov': color_coherence_cov}
    return None


def check_background_clutter(context: ValidationContext, edge_density_threshold: float = 0.15, debug: bool = False):
    """
    Check for clutter/patterns using sharpened edge density analysis.

    :param context: The validation context with segmentation data.
    :param edge_density_threshold: Maximum acceptable edge density (0.0-1.0).
    :param debug: If True, returns visualization data.
    """
    if context.segmentation_mask is None:
        raise CheckSkipped("No segmentation mask present.")

    background_mask_binary = (context.segmentation_mask < 0.5).astype(np.uint8)

    if np.sum(background_mask_binary) == 0:
        if context.face_data and 'bbox' in context.face_data:
            face_bbox = context.face_data['bbox']
            face_area = face_bbox['width'] * face_bbox['height']
            image_area = context.width * context.height
            if face_area / image_area > 0.85:
                raise CheckSkipped("No background pixels found, likely because the face covers the entire image.")
        raise CheckSkipped("Could not identify any background pixels for analysis. This indicates a segmentation failure.")

    rectangles = _find_background_rectangles(background_mask_binary)

    if len(rectangles) < 3:
        raise CheckSkipped(f"Not enough clean background regions ({len(rectangles)}) for clutter analysis.")

    gray_full = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2GRAY)
    edge_density = _calculate_edge_density_rect(gray_full, rectangles)
    context.measurements['background_edge_density'] = float(edge_density)
    context.measurements['background_rect_count'] = len(rectangles)

    if edge_density > edge_density_threshold:
        context.issues.append(ValidationIssue(
            category='error',
            message=f'Background has high edge density ({edge_density:.2%}), indicating patterns or objects.',
            severity=7
        ))

    if debug:
        sharpened = _sharpen_image(gray_full)
        edges = feature.canny(sharpened, sigma=2.0)
        edge_viz = (edges * 255).astype(np.uint8)

        return {
            'edge_density': edge_density,
            'sharpened_gray': sharpened,
            'edge_density_map': edge_viz
        }
    return None


def _background_lightness(context: ValidationContext, band_share: float = 0.03):
    """Lightness (L*, 0-100, lightly blurred) of the photo, and each pixel's distance to the person in bands.

    A band is 3% of the short side. Closer than one band to the person the selfie segmentation's edge is
    not reliable (hair, soft shoulders), so no check below looks there.
    """
    mask = context.person_mask_binary
    if mask is None:
        raise CheckSkipped('No segmentation mask present.')
    if not mask.any():
        raise CheckSkipped('The segmentation found no person.')
    band = max(3, int(round(min(context.width, context.height) * band_share)))
    lightness = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) * (100 / 255)
    lightness = cv2.GaussianBlur(lightness, (0, 0), band / 3)
    distance = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5) / band
    return lightness, distance, band


def _fit_plane(values: np.ndarray, where: np.ndarray, max_samples: int = 20000, trim: float = None):
    """Least-squares plane through the pixels in `where`, as (plane over the whole image, (a, b, c)) with
    values = a * x + b * y + c for x, y in 0..1 of the width and height. None with too few pixels.

    With `trim`, the pixels farther than that from the first plane (an object, a shadow, stray hair) are
    left out and the plane fitted again, so it follows the backdrop rather than what sits in front of it.
    """
    ys, xs = np.nonzero(where)
    if len(xs) < 50:
        return None
    step = max(1, len(xs) // max_samples)
    xs, ys = xs[::step], ys[::step]
    h, w = values.shape
    design = np.column_stack([xs / w, ys / h, np.ones(len(xs))])
    samples = values[ys, xs]
    coefficients, *_ = np.linalg.lstsq(design, samples, rcond=None)
    if trim is not None:
        keep = np.abs(samples - design @ coefficients) <= trim
        if keep.sum() >= 50:
            coefficients, *_ = np.linalg.lstsq(design[keep], samples[keep], rcond=None)
    rows, cols = np.mgrid[:h, :w]
    plane = coefficients[0] * cols / w + coefficients[1] * rows / h + coefficients[2]
    return plane.astype(np.float32), coefficients


def check_background_gradient(context: ValidationContext, warn_above: float, fail_above: float):
    """Check that the background doesn't run from light to dark: the lightness change across the photo of a
    plane fitted to the background, in L* (0-100).

    A plain backdrop under even light changes by up to about 16 L* across the photo (London Set); a gradient
    backdrop, or a lamp on one side, by 40 or more. The fit leaves out what sits in front of the backdrop
    (an object, stray hair, a shadow next to the person, which check_background_shadow reports).

    :param warn_above: L* change across the photo above which it warns.
    :param fail_above: L* change above which it fails.
    """
    lightness, distance, _ = _background_lightness(context)
    clear = distance > 1
    if clear.mean() < 0.02:
        raise CheckSkipped('Too little background to judge its gradient.')
    fit = _fit_plane(lightness, clear, trim=6)
    if fit is None:
        raise CheckSkipped('Too little background to judge its gradient.')
    _, (a, b, _) = fit
    span = float(abs(a) + abs(b))
    context.measurements['background_gradient'] = span

    if span > fail_above:
        context.issues.append(ValidationIssue(
            category='error', severity=6,
            message=f'The background runs from light to dark (by {span:.0f} L* across the photo). It must be evenly lit, in one colour.'))
    elif span > warn_above:
        context.issues.append(ValidationIssue(
            category='warning', severity=3,
            message=f'The background is lighter on one side than the other (by {span:.0f} L* across the photo).'))


def _rim_shadow_depth(context: ValidationContext, plane: np.ndarray, residual: np.ndarray, distance: np.ndarray,
                      band: int, sides, confidence: float, texture: float, chroma_tolerance: float = 8.0) -> float:
    """How much darker than the plane the backdrop is in the rim of the person mask, in L*.

    A cast shadow that touches the person is often taken for part of the person by the selfie segmentation,
    so the band outside the mask is clean and the shadow hides in the rim: the pixels the loose mask (above
    0.5) includes and a tight one (above `confidence`) does not, plus the first band outside the loose mask.
    Hair and clothing sit in that rim too, so only pixels that look like the backdrop count: within
    `chroma_tolerance` CIELAB units of the background's chroma, with a local lightness deviation under
    `texture` (a shadow is smooth, hair and folds are not), and no darker than half the plane (a black collar
    is). The depth is the 25th percentile of those pixels' residual on the darker side; 0 when fewer than
    100 such pixels are found on either side.
    """
    segmentation = context.segmentation_mask
    if segmentation is None:
        return 0.0
    loose = context.person_mask_binary.astype(bool)
    rim = (loose & ~(segmentation > confidence)) | ((distance > 0) & (distance <= 1))
    lab = cv2.cvtColor(context.image_np, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = lab[..., 1:] - 128
    far = distance > 3
    reference = np.median(chroma[far].reshape(-1, 2), axis=0)
    backdrop_like = np.linalg.norm(chroma - reference, axis=2) < chroma_tolerance
    lightness = lab[..., 0] * (100 / 255)
    window = max(3, int(band // 2) | 1)
    mean = cv2.blur(lightness, (window, window))
    deviation = np.sqrt(np.maximum(cv2.blur(lightness * lightness, (window, window)) - mean * mean, 0))
    smooth = rim & backdrop_like & (deviation < texture) & (residual > -0.5 * plane)
    depths = [-np.percentile(residual[smooth & side], 25) for side in sides if (smooth & side).sum() >= 100]
    return float(max(depths)) if depths else 0.0


def check_background_shadow(context: ValidationContext, warn_above: float, fail_above: float,
                            rim_confidence: float = 0.85, rim_texture: float = 3.0):
    """Check for a shadow cast on the background next to the person: how much darker the background around
    the person is than the background farther away, in L* (0-100).

    The band runs from one to three bands (3-9% of the short side) from the person. A plane fitted to the
    background beyond it gives the expected lightness, so a gradient across the whole photo doesn't count
    as a shadow. A flash throws the shadow to one side, so the band is judged left and right of the face
    separately: `background_shadow_depth` is how far the darkest quarter of the darker side falls below
    the plane. Plain backdrops stay below 6 (London Set).

    The selfie segmentation often takes a shadow that touches the person for part of the person; the band
    is then clean and the shadow hides in the mask's rim. `background_shadow_rim_depth` measures the rim
    (see _rim_shadow_depth), counting only pixels that look like a darkened backdrop. The verdict follows
    the deeper of the two.

    :param warn_above: Depth in L* above which it warns.
    :param fail_above: Depth above which it fails.
    :param rim_confidence: Segmentation confidence of the tight mask whose rim is searched.
    :param rim_texture: Local lightness deviation (L*) under which a rim pixel counts as smooth backdrop.
    """
    lightness, distance, band = _background_lightness(context)
    near = (distance > 1) & (distance <= 3)
    fit = _fit_plane(lightness, distance > 3, trim=6) if near.sum() >= 200 else None
    if fit is None:
        raise CheckSkipped('Too little background around the person to look for shadows.')
    plane, _ = fit
    residual = lightness - plane

    bbox = context.face_data.get('bbox')
    centre_x = bbox['x'] + bbox['width'] / 2 if bbox else context.width / 2
    columns = np.arange(context.width)[None, :]
    sides = (columns < centre_x, columns >= centre_x)
    depths = []
    for side in sides:
        pixels = residual[near & side]
        if len(pixels) >= 100:
            depths.append(-np.percentile(pixels, 25))
    if not depths:
        raise CheckSkipped('Too little background around the person to look for shadows.')
    band_depth = float(max(depths))
    rim_depth = _rim_shadow_depth(context, plane, residual, distance, band, sides, rim_confidence, rim_texture)
    context.measurements['background_shadow_depth'] = band_depth
    context.measurements['background_shadow_rim_depth'] = rim_depth
    context.measurements['background_shadow_share'] = float((residual[near] < -4).mean())
    depth = max(band_depth, rim_depth)

    if depth > fail_above:
        context.issues.append(ValidationIssue(
            category='error', severity=6,
            message=f'A shadow falls on the background next to the person ({depth:.0f} L* darker). Light the background evenly or move away from it.'))
    elif depth > warn_above:
        context.issues.append(ValidationIssue(
            category='warning', severity=3,
            message=f'The background is darker next to the person ({depth:.0f} L*), which looks like a shadow.'))
