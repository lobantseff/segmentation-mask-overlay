import logging
import time
from typing import Optional
from contextlib import contextmanager

import cv2
import numpy as np


def normalize_to_uint8(array: np.ndarray):
    if array.max() == array.min():
        array = array / (array.max() + 1e-6) * 255
    else:
        array = (array - array.min()) / (array.max() - array.min() + 1e-6) * 255
    return array.round().astype(np.uint8)


def check_convert_image(image: np.ndarray, input_dims: str = "HWC") -> np.ndarray:
    if input_dims == "HWC":
        ch_dim = -1
    elif input_dims == "CHW":
        ch_dim = -3
    else:
        raise AssertionError("input_dims should be HWC | CHW")

    if (
        image.dtype == np.uint8
        and image.max() <= 255
        and image.min() >= 0
    ):
        pass
    else:
        image = normalize_to_uint8(image)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif image.ndim == 3:
        if input_dims == "CHW":
            image = image.transpose(1, 2, 0)
        assert image.shape[-1] in [1, 3, 4], "Expected uint8 numpy array of shape HW1, HW3, HW4."
        if image.shape[-1] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[-1] == 3:
            return image
        if image.shape[-1] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    else:
        raise AssertionError("Expected numpy array of shape HW, HW1, HW3, HW4.")


def check_convert_mask(
    mask: np.ndarray, num_classes: Optional[int] = None, input_dims: str = "HWC"
) -> np.ndarray:
    """Checks and converts mask to HWC format if needed.

    Parameters
    ----------
    mask : np.ndarray
        Segmenttaion mask HWC, with binary channel per class
    num_classes : Optional[int], optional
        You may provide this to infer correct number of channels if mask 
        is provided in HW format. By default number of channels inferred
        as max(mask) + 1, by default None
    
    Returns
    -------
    np.ndarray (bool)
        CHW boolean mask array
    """

    assert_message = "The mask is expected to be an HW array with integer per class or HWC with bool mask per channel"
    mask = mask.astype(np.uint8)
    if mask.ndim == 3:
        assert mask.min() >= 0 and mask.max() <= 1, assert_message + f"mask.max: {mask.max()}; max.min: {mask.min()}"
        input_dims = "CHW" if mask.shape[0] < mask.shape[1] and mask.shape[0] < mask.shape[2] else "HWC"
        if input_dims == "CHW":
            mask = mask.transpose(1, 2, 0)
    elif mask.ndim == 2:
        if num_classes is None:
            num_classes = np.max(mask) + 1
        mask = np.eye(num_classes)[mask]  # One-hot encoded array
    else:
        raise AssertionError(assert_message)
    return mask.astype(bool)


@contextmanager
def catchtime(arg: str = "", logger: Optional[logging.Logger] = None) -> float:
    """_summary_

    Returns
    -------
    float
        _description_

    Example
    ------
    >>> with catchtime() as t:
    >>>     time.sleep(1)
    >>> print(f"Execution time: {t():.4f} secs")
    """
    
    start = time.perf_counter()
    yield
    if logger is not None:
        logger.info(f"{arg} exec time: {time.perf_counter() - start:.4f} secs")
    else:
        print(f"{arg} exec time: {time.perf_counter() - start:.4f} secs")
