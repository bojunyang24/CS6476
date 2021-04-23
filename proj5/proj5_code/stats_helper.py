import glob
import os
from os.path import dirname
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """
    Computes the mean and the standard deviation of all images present within
    the directory.

    Note: convert the image in grayscale and then in [0,1] before computing the
    mean and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = 1 / Variance

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None
    ############################################################################
    # Student code begin
    ############################################################################

    mean, variance, std, n = 0, 0, 0, 0
    for test_train in os.listdir(dir_name):
        test_train_dir = os.path.join(dir_name, test_train)
        for class_label in os.listdir(test_train_dir):
            class_dir = os.path.join(test_train_dir, class_label)
            for img in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img)
                x = np.array(Image.open(img_path).convert("L")) / 255
                img_size = x.shape[0] * x.shape[1]
                n += img_size
                mean += x.sum()
    mean = mean / n
    num_imgs = 0
    for test_train in os.listdir(dir_name):
        test_train_dir = os.path.join(dir_name, test_train)
        for class_label in os.listdir(test_train_dir):
            class_dir = os.path.join(test_train_dir, class_label)
            for img in os.listdir(class_dir):
                num_imgs+=1
                img_path = os.path.join(class_dir, img)
                x = np.array(Image.open(img_path).convert("L")) / 255
                variance += (1 / (n-1)) * np.sum((x-mean)**2)
    std = np.sqrt(variance)

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
