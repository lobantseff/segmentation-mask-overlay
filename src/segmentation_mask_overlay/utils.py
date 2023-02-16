import io
import os
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage


def get_img_from_fig(fig, dpi=180, color_cvt_flag=cv2.COLOR_BGR2RGB) -> np.ndarray:
    """Make numpy array from mpl fig
    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure, usually the result of plt.imshow()
    dpi : int, optional
        Dots per inches of the image to save. Note, that default matplotlib
        figsize is given in inches. Example: px, py = w * dpi, h * dpi  pixels
        6.4 inches * 100 dpi = 640 pixels, by default 180
    color_cvt_flag : int, optional
        OpenCV cvtColor flag. to get grayscale image,
        use `cv2.COLOR_BGR2GRAY`, by default `cv2.COLOR_BGR2RGB`.
    Returns
    -------
    np.ndarray[np.uint8]
        Image array
    """

    with io.BytesIO() as buffer:
        fig.savefig(buffer, format="png", dpi=dpi)
        buffer.seek(0)
        img_arr = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    return cv2.cvtColor(cv2.imdecode(img_arr, 1), color_cvt_flag)


def array_to_uintimage(array: np.ndarray, cvt_flag=cv2.COLOR_BGR2RGB) -> np.ndarray:
    """Converts any array to uint8 data type format using a magic of the matplotlib.
    Parameters
    ----------
    array : np.ndarray
        Image array.
    cvt_flag : int, optional
        OpenCV cvtColor flag. to get grayscale image,
        use `cv2.COLOR_BGR2GRAY`, by default `cv2.COLOR_BGR2RGB`
    Returns
    -------
    np.ndarray[np.uint8]
        Output image array in uint8 datatype
    """

    w, h = array.shape
    dpi = 90
    fig = plt.figure(figsize=(h / dpi, w / dpi))
    plt.imshow(array, cmap="gray")
    plt.axis("off")
    plt.tight_layout(pad=0)
    uint_img = get_img_from_fig(fig, dpi=dpi, color_cvt_flag=cvt_flag)
    plt.close(fig)
    return uint_img


def open_with_PIL(
    image: Union[os.PathLike, PILImage.Image, np.ndarray]
) -> PILImage.Image:
    """Reads image to PIL.Image.
    Parameters
    ----------
    image : Union[os.PathLike, PIL.Image.Image, np.ndarray]
        Image path, PIL.Image or numpy array.
    Returns
    -------
    PIL.Image.Image
        Output image as PIL.Image object
    Raises
    ------
    AttributeError
        In case of inappropriate input.
    """

    if isinstance(image, (str, os.PathLike)):
        pil_image = PILImage.open(image).convert("RGB")
    elif isinstance(image, PILImage.Image):
        pil_image = image.convert("RGB")
    elif isinstance(image, np.ndarray):
        image = image.squeeze()
        assert image.ndim == 2, "Supported only grayscale numpy arrays"
        image_array = array_to_uintimage(image)
        pil_image = PILImage.fromarray(image_array).convert("RGB")
        pil_image = pil_image.resize(image.shape[::-1])
    else:
        raise AttributeError("Unsupported type of image input")
    return pil_image
