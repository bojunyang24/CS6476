#!/usr/bin/python3

"""Generates random stereogram."""

from typing import Any, List, Tuple

import numpy as np
import torch


def generate_random_stereogram(
    im_size: Tuple[int, int, int] = (51, 51, 3), disparity: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a random stereogram for the given image size.

    The block which is shifted is centered at the image center and is of 0.5
    times the dimension of the input.

    Steps:
    1. Generate the left image to be random 0/1 values
    2. Set the right image as the copy of the left image
    3. Move a block around the center block in the right image by
       'disparity val' to the left
    4. Fill in the 'hole' in the right image with random values

    Note:
    1. The block to be moved is a square of size (H//2,W//2) at the center
       pixel of the image (H,W,C). Note the use of integer division.
    2. The values in the images should be 0 and 1 (at random)
    3. Your code will not be tested with inputs where moving the block with the
       given disparity takes the block out of bounds.
    4. The resulting image should be grayscale, i.e. a pixel value should be
       same in all the channels. image[x,y,0] == image[x,y,1] == ..... and so on
       for all the channels

    Args:
       im_size: The size of the image to be be generated
       disparity: the shift to be induced in the right image

    Returns:
       im_left: tensor of shape (H,W,C) representing the left image
       im_right: tensor of shape (H,W,C) representing the right image
    """
    H, W, C = im_size
    block_size = (H // 2, W // 2)
    im_left = torch.zeros(1)  # placeholder, not actual size
    im_right = torch.zeros(1)  # placeholder, not actual size

    ###########################################################################
    # Student code begins
    ###########################################################################
    left_layer = torch.randint(0,2,(H,W))
    upper_left = (H//2 - block_size[0]//2, W//2 - block_size[1]//2)
    block_range = [(upper_left[0], upper_left[0] + block_size[0]), (upper_left[1], upper_left[1] + block_size[1])]
    block = left_layer[block_range[0][0]:block_range[0][1], block_range[1][0]:block_range[1][1]]
    right_layer = left_layer.clone()
    right_layer[block_range[0][0]:block_range[0][1], (block_range[1][0] - disparity):(block_range[1][1] - disparity)] = block
    right_layer[block_range[0][0]:block_range[0][1], (block_range[1][1] - disparity):block_range[1][1]] = torch.randint(0,2,(block_size[0], disparity))
    left_stack = []
    right_stack = []
    for i in range(C):
        left_stack.append(left_layer)
        right_stack.append(right_layer)
    im_left = torch.stack(left_stack, 2)
    im_right = torch.stack(right_stack, 2)
    im_left = im_left.float()
    im_right = im_right.float()

    ###########################################################################
    # Student code ends
    ###########################################################################
    return im_left, im_right
