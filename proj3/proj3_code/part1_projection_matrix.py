import numpy as np
from numpy.lib.function_base import piecewise


def calculate_projection_matrix(
    points_2d: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
        points_2d: A numpy array of shape (N, 2)
        points_2d: A numpy array of shape (N, 3)

    Returns:
        M: A numpy array of shape (3, 4) representing the projection matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    M = np.zeros((3,4))
    X = points_3d[:,0]
    Y = points_3d[:,1]
    Z = points_3d[:,2]
    U = points_2d[:,0]
    V = points_2d[:,1]
    A_1 = np.stack(
        (
            X, Y, Z, np.ones(X.shape),
            np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape),
            -U*X, -U*Y, -U*Z
        ), axis=1)
    A_2 = np.stack(
        (
            np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape), np.zeros(X.shape),
            X, Y, Z, np.ones(X.shape),
            -V*X, -V*Y, -V*Z
        ), axis=1)
    A = np.empty((A_1.shape[0] * 2, A_1.shape[1]))
    A[::2] = A_1
    A[1::2]= A_2
    b = np.empty((V.shape[0]*2))
    b[::2] = U
    b[1::2] = V
    M, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    M = np.append(M, 1)
    M = M.reshape((3,4))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return M


def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
    Computes projection from [X,Y,Z,1] in homogenous coordinates to
    (x,y) in non-homogenous image coordinates.
    Args:
        P: 3 x 4 projection matrix
        points_3d: n x 4 array of points [X_i,Y_i,Z_i,1] in homogeneous
            coordinates or n x 3 array of points [X_i,Y_i,Z_i]
    Returns:
        projected_points_2d: n x 2 array of points in non-homogenous image
            coordinates
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    U = (points_3d.dot(P[0,:3]) + P[0,3]) / (points_3d.dot(P[2,:3]) + P[2,3])
    V = (points_3d.dot(P[1,:3]) + P[1,3]) / (points_3d.dot(P[2,:3]) + P[2,3])
    projected_points_2d = np.stack((U,V), axis=1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return projected_points_2d


def calculate_camera_center(M: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    Q = M[:3,:3]
    m4 = M[:,3]
    cc = -np.linalg.inv(Q).dot(m4)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return cc
