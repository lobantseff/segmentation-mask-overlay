import logging
import time
from typing import Optional
from contextlib import contextmanager

import cv2
import numpy as np


def check_convert_image(image: np.ndarray, input_format: str = "HWC") -> np.ndarray:
    if input_format == "HWC":
        ch_dim = -1
    elif input_format == "CHW":
        ch_dim = -3
    else:
        raise AssertionError("input_format should be HWC | CHW")

    if (
        image.dtype == np.uint8
        and image.max() <= 255
        and image.min() >= 0
        and image.ndim == 3
        and image.shape[ch_dim] == 3
    ):
        return image

    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    elif image.ndim == 3:
        if input_format == "CHW":
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


def check_convert_mask(mask: np.ndarray, num_classes: int) -> np.ndarray:
    assertion_message = "The mask is expected to be an HW array with integer per class or HWC with bool mask per channel"
    mask = mask.astype(np.uint8)
    if mask.ndim == 3:
        assert mask.max() == 1, assertion_message
        assert mask.shape[-1] == num_classes, "Num mask channels should be equal to len(labels)"
    elif mask.ndim == 2:
        mask = np.eye(num_classes)[mask]  # One-hot encoded array
    else:
        raise AssertionError(assertion_message)
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
