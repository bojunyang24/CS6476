#!/usr/bin/python3

import numpy as np
import torch

from torch import nn
from typing import Tuple


SOBEL_X_KERNEL = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).astype(np.float32)
SOBEL_Y_KERNEL = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]).astype(np.float32)


def compute_image_gradients(image_bw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Use convolution with Sobel filters to compute the image gradient at each
    pixel.

    Args:
        image_bw: A numpy array of shape (M,N) containing the grayscale image

    Returns:
        Ix: Array of shape (M,N) representing partial derivatives of image
            w.r.t. x-direction
        Iy: Array of shape (M,N) representing partial derivative of image
            w.r.t. y-direction
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    rx, cx = SOBEL_X_KERNEL.shape
    ry, cy = SOBEL_Y_KERNEL.shape
    padHx = rx // 2
    padWx = cx // 2
    padHy = ry // 2
    padWy = cy // 2
    SOBEL_X = torch.tensor(torch_shape(SOBEL_X_KERNEL))
    SOBEL_Y = torch.tensor(torch_shape(SOBEL_Y_KERNEL))
    image = torch.tensor(torch_shape(image_bw))
    Ix = nn.functional.conv2d(image, SOBEL_X, padding=(padHx, padWx)).numpy()[0][0]
    Iy = nn.functional.conv2d(image, SOBEL_Y, padding=(padHy, padWy)).numpy()[0][0]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return Ix, Iy

def torch_shape(arr_2d : np.ndarray) -> np.ndarray:
    return np.tile(arr_2d, (1,1,1,1))

def get_gaussian_kernel_2D_pytorch(ksize: int, sigma: float) -> torch.Tensor:
    """Create a Pytorch Tensor representing a 2d Gaussian kernel

    Args:
        ksize: dimension of square kernel
        sigma: standard deviation of Gaussian

    Returns:
        kernel: Tensor of shape (ksize,ksize) representing 2d Gaussian kernel

    You should be able to reuse your project 1 Code here.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # create gaussian kernel 1d
    mean = ksize // 2
    dist = np.linspace(start=0, stop=ksize, num=ksize, endpoint=False)
    exponent = (-1 / (2 * np.power(sigma, 2))) * np.power((dist - mean), 2)
    kernel = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(exponent)
    kernel = normalize(kernel)
    v = np.array([kernel]).T

    # create gaussian kernel 2d
    kernel = torch.tensor(
        normalize(
            np.outer(v,v)
        )
    ).float()

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return kernel

def normalize(np_array: np.ndarray) -> np.ndarray:
    return np_array / np.sum(np_array)

def second_moments(
    image_bw: np.ndarray,
    ksize: int = 7,
    sigma: float = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Compute second moments from image.

    Compute image gradients Ix and Iy at each pixel, the mixed derivatives,
    then the second moments (sx2, sxsy, sy2) at each pixel, using convolution
    with a Gaussian filter.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        ksize: size of 2d Gaussian filter
        sigma: standard deviation of Gaussian filter

    Returns:
        sx2: array of shape (M,N) containing the second moment in x direction
        sy2: array of shape (M,N) containing the second moment in y direction
        sxsy: array of dim (M,N) containing the second moment in the x then the
            y direction
    """

    sx2, sy2, sxsy = None, None, None
    ###########################################################################
    # TODO: YOUR SECOND MOMENTS CODE HERE                                     #
    ###########################################################################

    Ix, Iy = compute_image_gradients(image_bw)
    kernel = get_gaussian_kernel_2D_pytorch(ksize, sigma)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy
    
    sx2 = apply_kernel(Ix2, kernel)
    sy2 = apply_kernel(Iy2, kernel)
    sxsy = apply_kernel(IxIy, kernel)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return sx2, sy2, sxsy


def apply_kernel(image_bw: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Take greyscale image (2 dimensional) and kernel (2 dimensional) and apply kernel to image and return filtered image
    """
    r, c = kernel.shape
    padH = r // 2
    padW = c // 2
    # shape the kernel into torch specified shape
    torch_kernel = torch.tensor(torch_shape(kernel))
    torch_image = torch.tensor(torch_shape(image_bw))
    return nn.functional.conv2d(torch_image, torch_kernel, padding=(padH, padW)).numpy()[0][0]

def compute_harris_response_map(
    image_bw: np.ndarray,
    ksize: int = 7,
    sigma: float = 5,
    alpha: float = 0.05
) -> np.ndarray:
    """Compute the Harris cornerness score at each pixel (See Szeliski 7.1.1)

    Recall that R = det(M) - alpha * (trace(M))^2
    where M = [S_xx S_xy;
               S_xy  S_yy],
          S_xx = Gk * I_xx
          S_yy = Gk * I_yy
          S_xy  = Gk * I_xy,
    and * is a convolutional operation over a Gaussian kernel of size (k, k).
    (You can verify that this is equivalent to taking a (Gaussian) weighted sum
    over the window of size (k, k), see how convolutional operation works here:
        http://cs231n.github.io/convolutional-networks/)

    Ix, Iy are simply image derivatives in x and y directions, respectively.
    You may find the Pytorch function nn.Conv2d() helpful here.

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        ksize: size of 2d Gaussian filter
        sigma: standard deviation of gaussian filter
        alpha: scalar term in Harris response score

    Returns:
        R: array of shape (M,N), indicating the corner score of each pixel.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    sx2, sy2, sxsy = second_moments(image_bw, ksize, sigma)
    
    det = (sx2 * sy2) - (sxsy ** 2)
    trace = sx2 + sy2
    R = det - alpha * (trace ** 2)

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return R

def normalize(np_array: np.ndarray) -> np.ndarray:
    return np_array / np.sum(np_array)

def maxpool_numpy(R: np.ndarray, ksize: int) -> np.ndarray:
    """ Implement the 2d maxpool operator with (ksize,ksize) kernel size.

    Note: the implementation is identical to my_conv2d_numpy(), except we
    replace the dot product with a max() operator.

    Args:
        R: array of shape (M,N) representing a 2d score/response map

    Returns:
        maxpooled_R: array of shape (M,N) representing the maxpooled 2d
            score/response map
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    M, N = R.shape
    padding = ksize // 2
    maxpooled_R = np.empty(R.shape)
    padded_image = np.pad(R, (padding, padding), mode='constant', constant_values=0).reshape((M + 2 * padding, N + 2 * padding))
    for row in range(M):
        for col in range(N):
            maxpooled_R[row, col] = np.max(padded_image[row:row+ksize, col:col+ksize])

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return maxpooled_R


def nms_maxpool_pytorch(
    R: np.ndarray,
    k: int,
    ksize: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Get top k interest points that are local maxima over (ksize,ksize)
    neighborhood.

    HINT: One simple way to do non-maximum suppression is to simply pick a
    local maximum over some window size (u, v). This can be achieved using
    nn.MaxPool2d. Note that this would give us all local maxima even when they
    have a really low score compare to other local maxima. It might be useful
    to threshold out low value score before doing the pooling (torch.median
    might be useful here).

    You will definitely need to understand how nn.MaxPool2d works in order to
    utilize it, see https://pytorch.org/docs/stable/nn.html#maxpool2d

    Threshold globally everything below the median to zero, and then
    MaxPool over a 7x7 kernel. This will fill every entry in the subgrids
    with the maximum nearby value. Binarize the image according to
    locations that are equal to their maximum. Multiply this binary
    image, multiplied with the cornerness response values. We'll be testing
    only 1 image at a time.

    Args:
        R: score response map of shape (M,N)
        k: number of interest points (take top k by confidence)
        ksize: kernel size of max-pooling operator

    Returns:
        x: array of shape (k,) containing x-coordinates of interest points
        y: array of shape (k,) containing y-coordinates of interest points
        c: array of shape (k,) containing confidences of interest points
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    median = torch.median(torch.tensor(R))
    R[R < median.item()] = 0.0
    pool = nn.MaxPool2d(kernel_size=ksize, stride=1, padding=ksize//2)
    tensor_R = torch.tensor(torch_shape(R))
    pooled_R = pool(tensor_R)
    binary = R == pooled_R.numpy()[0][0]
    interesting = binary * R
    r, c = np.nonzero(interesting)
    confidences = R[r,c]
    ind = np.argsort(-confidences)
    x = np.take_along_axis(c, ind, axis=0)[:k]
    y = np.take_along_axis(r, ind, axis=0)[:k]
    confidences = np.take_along_axis(confidences, ind, axis=0)[:k]

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return x, y, confidences


def remove_border_vals(
    img: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray
) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Remove interest points that are too close to a border to allow SIFT feature
    extraction. Make sure you remove all points where a 16x16 window around
    that point cannot be formed.

    Args:
        img: array of shape (M,N) containing the grayscale image
        x: array of shape (k,) representing x coord of interest points
        y: array of shape (k,) representing y coord of interest points
        c: array of shape (k,) representing confidences of interest points

    Returns:
        x: array of shape (p,), where p <= k (less than or equal after pruning)
        y: array of shape (p,)
        c: array of shape (p,)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    # row is y is M, col is x is N
    M, N = img.shape
    x_mask = (x >= 7) * (x <= N - 9)
    y_mask = (y >= 7) * (y <= M - 9)
    mask = x_mask * y_mask
    x = x[mask]
    y = y[mask]
    c = c[mask]

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return x, y, c


def get_harris_interest_points(
    image_bw: np.ndarray,
    k: int = 2500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Implement the Harris Corner detector. You will find
    compute_harris_response_map(), nms_maxpool_pytorch(), and
    remove_border_vals() useful. Make sure to sort the interest points in
    order of confidence!

    Args:
        image_bw: array of shape (M,N) containing the grayscale image
        k: maximum number of interest points to retrieve

    Returns:
        x: array of shape (p,) containing x-coordinates of interest points
        y: array of shape (p,) containing y-coordinates of interest points
        c: array of dim (p,) containing the strength(confidence) of each
            interest point where p <= k.
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    R = compute_harris_response_map(image_bw)
    x,y,c = nms_maxpool_pytorch(R, k, 7)
    x,y,c = remove_border_vals(image_bw, x, y, c)
    c = c / np.max(c)

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################

    return x, y, c
