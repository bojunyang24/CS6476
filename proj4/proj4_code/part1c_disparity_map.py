#!/usr/bin/python3

"""Calculates disparity maps."""

from typing import Callable, Tuple

import numpy as np
import torch


def calculate_disparity_map(
    left_img: torch.Tensor,
    right_img: torch.Tensor,
    block_size: int,
    sim_measure_function: Callable,
    max_search_bound: int = 50,
) -> torch.Tensor:
    """
    Calculate the disparity value at each pixel by searching a small patch
    around a pixel from the left image in the right image.

    Note:
    1. It is important for this project to follow the convention of search
       input in left image and search target in right image
    2. While searching for disparity value for a patch, it may happen that
       there are multiple disparity values with the minimum value of the
       similarity measure. In that case we need to pick the smallest disparity
       value. Please check the numpy's argmin and pytorch's argmin carefully.
       Example:
       -- diparity_val -- | -- similarity error --
       -- 0               | 5
       -- 1               | 4
       -- 2               | 7
       -- 3               | 4
       -- 4               | 12

       In this case we need the output to be 1 and not 3.
    3. The max_search_bound is defined from the patch center.

    Args:
        left_img: image Tensor of shape (H,W,C) from the left stereo camera.
            C will be >= 1
        right_img: image Tensor of shape (H,W,C) from the right stereo camera
        block_size: the size of the block to be used for searching between the
            left and right image (should be odd)
        sim_measure_function: a function to measure similarity measure between
            two tensors of the same shape; returns the error value
        max_search_bound: the maximum horizontal distance (in terms of pixels)
            to use for searching
    Returns:
        disparity_map: The map of disparity values at each pixel. Tensor of
            shape (H-2*(block_size//2),W-2*(block_size//2))
    """

    assert left_img.shape == right_img.shape

    ###########################################################################
    # Student code begins
    ###########################################################################
    block_half = block_size//2
    disp_map = torch.zeros(left_img.shape[0] - 2*block_half, left_img.shape[1] - 2*block_half)
    leftmost = block_half
    # for row in range(block_half, left_img.shape[0]-block_half):
    #     for col in range(leftmost, left_img.shape[1]-block_half):
    #         pass
    for r in range(0, disp_map.shape[0]):
        for c in range(0, disp_map.shape[1]):
            row = r + block_half # row is indexing for patches, r is indexing for disp_map
            col = c + leftmost # col is indexing for patches, c is indexing for disp_map
            if block_half == 0:
                left_patch = left_img[row, col, :]
                sim_error = [] # indices are disparity, values are similarity error
                for disparity in range(max_search_bound+1):
                    if col - disparity >= 0:
                        right_patch = right_img[row, col-disparity, :]
                        sim_error.append(sim_measure_function(left_patch, right_patch))
                disp_map[r, c] = np.argmin(sim_error)
            else:
                left_patch = left_img[row-block_half:row+block_half+1, col-block_half:col+block_half+1, :]
                sim_error = [] # indices are disparity, values are similarity error
                for disparity in range(max_search_bound+1):
                    if col - block_half - disparity >= 0:
                        right_patch = right_img[row-block_half:row+block_half+1, col-block_half-disparity:col+block_half-disparity+1, :]
                        sim_error.append(sim_measure_function(left_patch, right_patch))
                disp_map[r, c] = np.argmin(sim_error)

        

    ###########################################################################
    # Student code ends
    ###########################################################################
    return disp_map


def calculate_cost_volume(
    left_img: torch.Tensor,
    right_img: torch.Tensor,
    max_disparity: int,
    sim_measure_function: Callable,
    block_size: int = 9,
):
    """
    Calculate the cost volume. Each pixel will have D=max_disparity cost values
    associated with it. Basically for each pixel, we compute the cost of
    different disparities and put them all into a tensor.

    Note:
    1. It is important for this project to follow the convention of search
       input in left image and search target in right image
    2. If the shifted patch in the right image will go out of bounds, it is
       good to set the default cost for that pixel and disparity to be
       something high (we recommend 255) so that when we consider costs, valid
       disparities will have a lower cost.

    Args:
        left_img: image Tensor of shape (H,W,C) from the left stereo camera.
            C will be 1 or 3.
        right_img: image Tensor of shape (H,W,C) from the right stereo camera
        max_disparity: represents the range of disparity values we will
            consider (0 to max_disparity-1)
        sim_measure_function: a function to measure similarity measure between
            two tensors of the same shape; returns the error value
        block_size: the size of the block to be used for searching between the
            left and right image, it should be odd
    Returns:
        cost_volume: The cost volume tensor of shape (H,W,D). H,W are image
            dimensions, and D is max_disparity. cost_volume[x,y,d] represents
            the similarity or cost between a patch around left[x,y] and a patch
            shifted by disparity d in the right image.
    """
    # placeholders

    ###########################################################################
    # Student code begins
    ###########################################################################

    block_half = block_size//2
    H = left_img.shape[0] - 2*block_half
    W = right_img.shape[1] - 2*block_half
    cost_volume = torch.zeros(H, W, max_disparity)
    
    leftmost = block_half
    for r in range(0, cost_volume.shape[0]):
        for c in range(0, cost_volume.shape[1]):
            row = r + block_half # row is indexing for patches, r is indexing for disp_map
            col = c + leftmost # col is indexing for patches, c is indexing for disp_map
            if block_half == 0:
                left_patch = left_img[row, col, :]
                for disparity in range(max_disparity):
                    if col - disparity >= 0:
                        right_patch = right_img[row, col-disparity, :]
                        cost_volume[r, c, disparity] = sim_measure_function(left_patch, right_patch)
            else:
                left_patch = left_img[row-block_half:row+block_half+1, col-block_half:col+block_half+1, :]
                for disparity in range(max_disparity):
                    if col - block_half - disparity >= 0:
                        right_patch = right_img[row-block_half:row+block_half+1, col-block_half-disparity:col+block_half-disparity+1, :]
                        cost_volume[r, c, disparity] = sim_measure_function(left_patch, right_patch)

    ###########################################################################
    # Student code ends
    ###########################################################################

    return cost_volume
