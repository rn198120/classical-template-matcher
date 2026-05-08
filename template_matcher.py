"""
template_matcher.py

A from-scratch implementation of template matching using
Normalized Cross-Correlation (NCC) and an image pyramid for
coarse-to-fine search.

The module exposes one main entry point:
    pyramid_match(image, template, num_levels=3, window_radius=10)

Lower-level building blocks are also exported in case the caller wants
to use them directly (e.g. for visualization or experimentation):
    ncc(patch1, patch2)
    sliding_ncc(main, template)
    shrink_by_half(img)
    build_pyramid(img, num_levels)
    refine_peak(main, template, rough_y, rough_x, window_radius=10)
"""

import numpy as np
import cv2


def ncc(patch1, patch2):
    """
    Normalized Cross-Correlation between two equal-sized patches.

    NCC measures similarity in a way that is invariant to uniform
    brightness shifts: it subtracts each patch's mean before comparing,
    so a patch that is identical in pattern but uniformly brighter
    still scores +1.

    Parameters
    ----------
    patch1, patch2 : numpy.ndarray
        Two 2D arrays of the same shape (single-channel grayscale).

    Returns
    -------
    float
        A score in the range [-1.0, +1.0]. Higher means more similar.
        +1.0 indicates an identical pattern (up to brightness shift).
        Returns 0.0 if either patch has zero variance (uniform).
    """
    p1 = patch1.astype(np.float64)
    p2 = patch2.astype(np.float64)

    p1_centered = p1 - p1.mean()
    p2_centered = p2 - p2.mean()

    numerator = np.sum(p1_centered * p2_centered)
    denominator = np.sqrt(np.sum(p1_centered ** 2) * np.sum(p2_centered ** 2))

    if denominator == 0:
        return 0.0
    return numerator / denominator


def sliding_ncc(main, template):
    """
    Compute NCC at every valid position of the template within the main image.

    This is the brute-force template search. It is O(H * W * h * w) where
    (H, W) is the main image size and (h, w) is the template size, so it
    becomes very slow on large inputs. Use pyramid_match for speed.

    Parameters
    ----------
    main : numpy.ndarray
        2D grayscale image to search within.
    template : numpy.ndarray
        2D grayscale template to find. Must fit inside `main`.

    Returns
    -------
    numpy.ndarray
        A 2D response map of shape (main_h - tmpl_h + 1, main_w - tmpl_w + 1).
        Each entry is the NCC score with the template placed at that
        top-left position.
    """
    main_h, main_w = main.shape
    tmpl_h, tmpl_w = template.shape

    out_h = main_h - tmpl_h + 1
    out_w = main_w - tmpl_w + 1

    response = np.zeros((out_h, out_w), dtype=np.float64)

    for y in range(out_h):
        for x in range(out_w):
            patch = main[y:y + tmpl_h, x:x + tmpl_w]
            response[y, x] = ncc(patch, template)

    return response


def shrink_by_half(img):
    """
    Halve the dimensions of an image using Gaussian-blur + downsample.

    Wraps cv2.pyrDown, which applies a 5x5 Gaussian blur before subsampling
    to avoid aliasing.

    Parameters
    ----------
    img : numpy.ndarray
        2D grayscale image (or 3D color image; cv2.pyrDown handles both).

    Returns
    -------
    numpy.ndarray
        An image with roughly half the height and half the width of the input.
        Odd-sized dimensions are rounded.
    """
    return cv2.pyrDown(img)


def build_pyramid(img, num_levels):
    """
    Build an image pyramid: a list of progressively halved versions.

    Parameters
    ----------
    img : numpy.ndarray
        The base (largest) image.
    num_levels : int
        Total number of levels to produce, including the original.
        Must be >= 1.

    Returns
    -------
    list of numpy.ndarray
        pyramid[0] is the original image.
        pyramid[-1] is the smallest (deepest) level.
        Each level is roughly half the dimensions of the previous.
    """
    pyramid = [img]
    for _ in range(num_levels - 1):
        pyramid.append(shrink_by_half(pyramid[-1]))
    return pyramid


def refine_peak(main, template, rough_y, rough_x, window_radius=10):
    """
    Search a small window around an approximate peak for the best NCC match.

    Used in the pyramid coarse-to-fine refinement step. Given a rough
    estimate of where the template is in `main`, perform an exhaustive
    search in a (2 * window_radius + 1) x (2 * window_radius + 1)
    neighborhood and return the best position found.

    Parameters
    ----------
    main : numpy.ndarray
        2D grayscale image to search within.
    template : numpy.ndarray
        2D grayscale template.
    rough_y, rough_x : int
        Approximate top-left position of the template in `main`.
    window_radius : int, optional
        Half-side of the search window in pixels (default 10).
        The search covers (rough_y +/- radius, rough_x +/- radius),
        clamped to valid positions.

    Returns
    -------
    best_y, best_x : int
        Refined top-left position of the best match.
    best_score : float
        NCC score at that position.
    """
    main_h, main_w = main.shape
    tmpl_h, tmpl_w = template.shape

    y_min = max(0, rough_y - window_radius)
    y_max = min(main_h - tmpl_h, rough_y + window_radius)
    x_min = max(0, rough_x - window_radius)
    x_max = min(main_w - tmpl_w, rough_x + window_radius)

    best_score = -np.inf
    best_y, best_x = rough_y, rough_x

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            patch = main[y:y + tmpl_h, x:x + tmpl_w]
            score = ncc(patch, template)
            if score > best_score:
                best_score = score
                best_y, best_x = y, x

    return best_y, best_x, best_score


def pyramid_match(image, template, num_levels=3, window_radius=10, verbose=False):
    """
    Coarse-to-fine template matching using an image pyramid.

    Builds pyramids of both the image and the template, performs a full
    sliding-window NCC search at the smallest level to find a rough peak,
    then refines that peak by searching small windows at progressively
    larger levels until the original resolution is reached.

    Parameters
    ----------
    image : numpy.ndarray
        2D grayscale image to search within.
    template : numpy.ndarray
        2D grayscale template to find.
    num_levels : int, optional
        Number of pyramid levels (default 3). More levels = faster
        coarse search but risk losing template structure if the smallest
        level becomes too coarse.
    window_radius : int, optional
        Refinement search radius at each non-smallest level (default 10).
        Should be large enough to absorb the rounding errors introduced
        by halving odd-sized dimensions.
    verbose : bool, optional
        If True, print progress at each pyramid level.

    Returns
    -------
    best_y, best_x : int
        Top-left position of the best match in the original image's coordinates.
    best_score : float
        NCC score at that position (in [-1.0, +1.0]).
    """
    image_pyr = build_pyramid(image, num_levels)
    template_pyr = build_pyramid(template, num_levels)

    # Step 1: full search at the smallest level
    smallest_image = image_pyr[-1]
    smallest_template = template_pyr[-1]

    if verbose:
        print(f"Level {num_levels - 1} (smallest): full search "
              f"on {smallest_image.shape} image, {smallest_template.shape} template")

    response = sliding_ncc(smallest_image, smallest_template)
    flat_idx = np.argmax(response)
    rough_y, rough_x = np.unravel_index(flat_idx, response.shape)
    rough_score = response[rough_y, rough_x]

    if verbose:
        print(f"  Peak at level {num_levels - 1}: "
              f"({rough_y}, {rough_x}) score={rough_score:.4f}")

    # Step 2: refine at each larger level
    for level in range(num_levels - 2, -1, -1):
        rough_y *= 2
        rough_x *= 2

        if verbose:
            print(f"Level {level}: refining around ({rough_y}, {rough_x}) "
                  f"with window radius {window_radius}")

        rough_y, rough_x, rough_score = refine_peak(
            image_pyr[level],
            template_pyr[level],
            rough_y, rough_x,
            window_radius=window_radius,
        )

        if verbose:
            print(f"  Refined to ({rough_y}, {rough_x}) score={rough_score:.4f}")

    return rough_y, rough_x, rough_score