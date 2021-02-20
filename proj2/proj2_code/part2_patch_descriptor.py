#!/usr/bin/python3

import numpy as np
from numpy.lib.financial import fv
from numpy.lib.type_check import imag


def compute_normalized_patch_descriptors(
    image_bw: np.ndarray, X: np.ndarray, Y: np.ndarray, feature_width: int
) -> np.ndarray:
    """Create local features using normalized patches.

    Normalize image intensities in a local window centered at keypoint to a
    feature vector with unit norm. This local feature is simple to code and
    works OK.

    Choose the top-left option of the 4 possible choices for center of a square
    window.

    Args:
        image_bw: array of shape (M,N) representing grayscale image
        X: array of shape (K,) representing x-coordinate of keypoints
        Y: array of shape (K,) representing y-coordinate of keypoints
        feature_width: size of the square window

    Returns:
        fvs: array of shape (K,D) representing feature descriptors
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    if feature_width%2 == 0:
        lower = (feature_width - 2)//2
        upper = feature_width//2 + 1
    else:
        upper = feature_width // 2
        lower = feature_width // 2
    fvs = np.empty((X.shape[0], feature_width**2))
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        window = image_bw[y-lower:y+upper, x-lower:x+upper]
        window = (1 / np.linalg.norm(window)) * window
        fvs[i] = window.flatten()

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return fvs
