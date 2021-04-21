#!/usr/bin/python3

"""Generates a patch from an image."""

import torch


def gen_patch(image: torch.Tensor, x: int, y: int, ws: int = 11) -> torch.Tensor:
    """Returns a patch at a specific location of the image.

    x, y in this case is a top left corner of the patch, for example if (x,y)
    is (0,0) you should return a patch over (0,0) and (ws,ws)

    For corner case, you can pad the output with zeros such that we always have
    (channel, ws, ws) dimension output

    Args:
        image: image of type Tensor with dimension (C, height, width)
        x: x coordinate in the image
        y: y coordinate in the image
        ws: window size or block size of the patch we want

    Returns:
        patch: a patch of size (C, ws, ws) of type Tensor
    """
    ###########################################################################
    # Student code begins
    ###########################################################################
    
    C, H, W = image.shape
    patch = torch.zeros(C, ws, ws)
    if x > H or y > W:
        return patch.float()
    patch_x_lower, patch_x_upper = 0, ws
    image_x_lower, image_x_upper = x, x+ws
    patch_y_lower, patch_y_upper = 0, ws
    image_y_lower, image_y_upper = y, y+ws
    if x + ws > H:
        patch_x_lower = 0
        patch_x_upper = H-x
        image_x_lower = x
        image_x_upper = H
    if y + ws > W:
        patch_y_lower = 0
        patch_y_upper = W-y
        image_y_lower = y
        image_y_upper = W
    if x < 0:
        patch_x_lower = -x
        patch_x_upper = ws
        image_x_lower = 0
        image_x_upper = x+ws
    if y < 0:
        patch_y_lower = -y
        patch_y_upper = ws
        image_y_lower = 0
        image_y_upper = y+ws
    # patch = torch.zeros(C, ws, ws)
    # print(f"imageshape: {image.shape}, x: {x}, y: {y}, ws: {ws}\npatch_y_lower: {patch_y_lower}, patch_y_upper: {patch_y_upper}, patch_x_lower: {patch_x_lower}, patch_x_upper: {patch_x_upper}\nimage_y_lower: {image_y_lower}, image_y_upper: {image_y_upper}, image_x_lower: {image_x_lower}, image_x_upper: {image_x_upper}")
    # patch[:, patch_y_lower:patch_y_upper, patch_x_lower:patch_x_upper] = image[:, image_y_lower:image_y_upper, image_x_lower:image_x_upper]
    patch[:, patch_x_lower:patch_x_upper, patch_y_lower:patch_y_upper] = image[:, image_x_lower:image_x_upper, image_y_lower:image_y_upper]
    # if x + ws > W or y + ws > H: # x is col is W, y is row is H
    #     patch = torch.zeros(C, ws, ws)
    #     patch[:, 0:H-y, 0:W-x] = image[:, y:H, x:W]
    # elif x < 0 or y < 0:
    #     patch = torch.zeros(C, ws, ws)
    #     patch[:, -y:H, -x:H] = image[:, 0:y+ws, 0:x+ws]
    # else:
    # patch = image[:, y:y+ws, x:x+ws]

    ###########################################################################
    # Student code ends
    ###########################################################################
    return patch.float()
