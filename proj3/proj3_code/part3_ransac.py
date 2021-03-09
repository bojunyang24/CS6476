import numpy as np
import math

from proj3_code.part2_fundamental_matrix import estimate_fundamental_matrix


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: float) -> int:
    """
    Calculates the number of RANSAC iterations needed for a given guarantee of
    success.

    Args:
        prob_success: float representing the desired guarantee of success
        sample_size: int the number of samples included in each RANSAC
            iteration
        ind_prob_success: float representing the probability that each element
            in a sample is correct

    Returns:
        num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    num_samples = np.log(1 - prob_success) / np.log(1 - np.power(ind_prob_correct, sample_size))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)


def ransac_fundamental_matrix(
    matches_a: np.ndarray, matches_b: np.ndarray) -> np.ndarray:
    """
    For this section, use RANSAC to find the best fundamental matrix by
    randomly sampling interest points. You would reuse
    estimate_fundamental_matrix() from part 2 of this assignment and
    calculate_num_ransac_iterations().

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 30 points for either left or
    right images.

    Tips:
        0. You will need to determine your prob_success, sample_size, and
            ind_prob_success values. What is an acceptable rate of success? How
            many points do you want to sample? What is your estimate of the
            correspondence accuracy in your dataset?
        1. A potentially useful function is numpy.random.choice for creating
            your random samples.
        2. You will also need to choose an error threshold to separate your
            inliers from your outliers. We suggest a threshold of 0.1.

    Args:
        matches_a: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image A
        matches_b: A numpy array of shape (N, 2) representing the coordinates
            of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
        best_F: A numpy array of shape (3, 3) representing the best fundamental
            matrix estimation
        inliers_a: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image A that are inliers with respect to
            best_F
        inliers_b: A numpy array of shape (M, 2) representing the subset of
            corresponding points from image B that are inliers with respect to
            best_F
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # define params
    prob_success = 0.99
    sample_size = 8
    ind_prob_correct = 0.4
    inlier_threshold = 0.05

    # num of iterations to run ransac
    iters = calculate_num_ransac_iterations(prob_success, sample_size, ind_prob_correct)
    
    inliers_a = []
    inliers_b = []
    max_inliers = 0
    best_F = []

    # create 3d points for matrix operations
    matches_a_3d = np.hstack((matches_a, np.ones((matches_a.shape[0],1))))
    matches_b_3d = np.hstack((matches_a, np.ones((matches_b.shape[0],1))))

    # ransac
    for i in range(iters):
        # select random sample
        points = np.random.choice(matches_a.shape[0], sample_size)
        points_a = matches_a[points,:]
        points_b = matches_b[points,:]
        F = estimate_fundamental_matrix(points_a, points_b)
        # calculate error using estimated F
        # errors = np.sum(matches_a_3d.dot(F) * matches_b_3d, axis=1)
        errors = np.sum(F.dot(matches_a_3d.T).T * matches_b_3d, axis=1)
        errors = errors / np.linalg.norm(F.dot(matches_b_3d.T))

        inliers = np.sum(np.abs(errors) < inlier_threshold)
        if inliers > max_inliers:
            mask = np.abs(errors) < inlier_threshold
            inliers_a = matches_a[mask]
            inliers_b = matches_b[mask]
            max_inliers = inliers
            best_F = F
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_F, inliers_a, inliers_b
