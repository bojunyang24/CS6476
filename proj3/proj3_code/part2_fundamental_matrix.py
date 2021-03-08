"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    cu, cv = np.mean(points, axis=0)
    su = 1/np.std(points[:,0] - cu)
    sv = 1/np.std(points[:,1] - cv)
    scale = np.array([
        [su, 0, 0],
        [0, sv, 0],
        [0, 0, 1]
    ])
    offset = np.array([
        [1, 0, -cu],
        [0, 1, -cv],
        [0, 0, 1]
    ])
    T = np.dot(scale, offset)
    points_3d = np.hstack((points, np.ones((points.shape[0],1))))
    uv1 = np.dot(T,points_3d.T).T
    points_normalized = uv1[:,:2]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(
    F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    F_orig = T_b.T.dot(F_norm).dot(T_a)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # a[0] = u, a[1] = v
    a, T_a = normalize_points(points_a)
    # b[0] = u', b[1] = v'
    b, T_b = normalize_points(points_b)
    # SVD method
    # A = np.stack(
    #     (
    #         a[:,0]*b[:,0], a[:,1]*b[:,0], b[:,0],
    #         a[:,0]*b[:,1], a[:,1]*b[:,1], b[:,1],
    #         a[:,0], a[:,1], np.ones((a.shape[0]))
    #     ), axis=1
    # )
    # u, s, v = np.linalg.svd(A)
    # F = v[-1,:].reshape((3,3))
    # U,S,V = np.linalg.svd(F)
    # S = np.sort(S)[::-1]
    # S2 = np.array([
    #     [S[0], 0, 0],
    #     [0, S[1], 0],
    #     [0, 0, 0]
    # ])
    # F = U.dot(S2).dot(V)
    # F = unnormalize_F(F, T_a, T_b)

    # least squares method
    A = np.stack(
        (
            a[:,0]*b[:,0], a[:,1]*b[:,0], b[:,0],
            a[:,0]*b[:,1], a[:,1]*b[:,1], b[:,1],
            a[:,0], a[:,1]
        ), axis=1
    )
    F, residuals, rank, s = np.linalg.lstsq(A, -np.ones((A.shape[0])), rcond=None)
    F = np.append(F, 1).reshape(3,3)
    u,s,v = np.linalg.svd(F)
    S = np.array([
        [s[0], 0, 0],
        [0, s[1], 0],
        [0, 0, 0]
    ])
    F = u.dot(S).dot(v)
    F = unnormalize_F(F, T_a, T_b)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F
