import numpy as np
from PIL import Image


def read_jpg_gray(path):
    """Read a JPG as a grayscale float32 array, pixel values in [0, 255]."""
    return np.array(Image.open(path).convert("L"), dtype=np.float32)
