from collections.abc import Iterable
from typing import Optional, Tuple, Union, List

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
from matplotlib import colors as C
from matplotlib.patches import Patch
from PIL import Image as PILImage
from segmentation_mask_overlay.utils import check_convert_image, check_convert_mask


def overlay_masks(
    image: np.ndarray,
    masks: np.ndarray,
    labels: List[str],
    colors: Optional[Union[np.ndarray, List[Union[str, List[float]]]]] = None,
    alpha: float = 1.0,
    beta: float = 0.5,
    return_type: str = "numpy",
    mpl_figsize: Tuple[int, int] = (8, 8),
    mpl_dpi: int = 90,
    concat_original: Optional[str] = None,
):
    """Overlays masks on the image.
    Parameters
    ----------
    image : np.ndarray
        Image as a numpy array of shape: HWC or HW.
        The image will be mormalized to [0, 255] and casted to uint8 RGB. Do it manually
        if you want to control this.
    mask : np.ndarray[bool | int]
        Mask should be a numpy array of shape of the image in one of the following forms:
        - H W C, with bool mask per channel, where each channel represents a class.
        - H W, with an pixel integer value representing a class.
    labels : Optional[List[str]], optional
        Names of expected labels. Provide in the same order as the channels in the masks.
        If provided, defines 
        If not provided, will be set as range(mask.shape[-1] | max(mask) + 1), by default None
    colors : Union[np.ndarray, List[Union[str, List[float]]]], optional
        Array of shape (n_labels x 3) or list of matplotlib acceptable colornames.
        Example to get persistent colormap: `plt.cm.tab20(np.arange(NUM_LABELS))[..., :-1]`
    alpha : float, optional
        Image alpha, by default 1.0
    beta : float, optional
        Masks alpha, by default 0.4
    mpl_colormap : str
        Matplotlib colormap name
    return_type : bool, should be numpy | pil | mpl
        Return numpy array, or PIL image, or matpotlib figure.
    mpl_figsize : tuple, optional
        Size in inches of the output matpotlib figure.
        Valid only if return_type="mpl", by default (12, 12)
    mpl_dpi : int, optional
        Resolution of the output matpotlib figure. Note: 'px, py = w * dpi, h * dpi'.
        Valid only if return_type="mpl", by default 120
    concat_original: str, optional
        If provided, should be 'v' or 'h', represention vericat or horisontal concatenation
        of original image to the image with overlayed masks, by default: None

    Returns
    -------
    plt.Figure | PIL.Image | np.ndarray
        Output mpl figure or pillow image with masks.
    """

    assert image.shape[:2] == masks.shape[:2], "Image and mask should be of the same size"

    num_classes = len(labels)

    image = check_convert_image(image)
    masks = check_convert_mask(masks, num_classes)

    if colors is None:
        if num_classes <= 10:
            colors = colormap.get_cmap("tab10", 10).colors[:num_classes, :-1]
        else:
            colors = colormap.get_cmap("rainbow_r", num_classes)(np.arange(num_classes))[:, :-1]

    else:
        assert len(colors) == num_classes, "Number of provided colors != number of labels"
        if all(isinstance(c, str) for c in colors):
            colors = [C.to_rgb(c) for c in colors]

        if isinstance(colors, Iterable):
            colors = np.array(colors)

        assert colors.ndim == 2 and colors.shape[-1] == 3, (
            "Unsupported color format:"
            + " should be list of matplotlib colorname strings for each mask/mask_channel,"
            + " list of RGB arrays or 2-dim numpy array of shape (n_labels x 3)"
        )

    mask_colors = colors.copy()
    mask_colors = (mask_colors * 255).astype(np.uint8)
    cbar = list(zip(mask_colors, colors))

    segmentation_overlay = np.zeros_like(image, dtype=np.uint16)
    segmentation_mask = np.zeros(image.shape[:2], dtype=bool)
    legend_elements = []

    for i in range(num_classes):
        label = labels[i]
        mask = masks[..., i]
        color, legend_color = cbar[i]

        intersection = mask & segmentation_mask
        segmentation_mask = mask | segmentation_mask

        # Paint non-overlapping area
        segmentation_overlay[mask ^ intersection] = color

        # Blend overlapping area
        segmentation_overlay[intersection] = (segmentation_overlay[intersection] + color) // 2

        legend_elements.append(Patch(color=legend_color, label=label))

    overlayed = cv2.addWeighted(image, alpha, segmentation_overlay.astype(np.uint8), beta, 0)

    if concat_original is not None:
        if concat_original == "h":
            overlayed = np.concatenate((image, overlayed), 1)
            mpl_figsize = (mpl_figsize[0] * 2, mpl_figsize[1])
        elif concat_original == "v":
            overlayed = np.concatenate((image, overlayed), 0)
            mpl_figsize = (mpl_figsize[0], mpl_figsize[1] * 2)
        else:
            raise AssertionError("If provided, concat_original should be in v | h")

    if return_type == "numpy":
        return overlayed

    elif return_type == "pil":
        pil_image = PILImage.fromarray(overlayed)
        return pil_image

    elif return_type == "mpl":
        fig = plt.figure(figsize=mpl_figsize, dpi=mpl_dpi)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        plt.imshow(overlayed, vmin=0, vmax=255)
        plt.axis("off")
        mask_legend = plt.legend(
            handles=legend_elements,
            loc="upper left",
            frameon=False,
            bbox_to_anchor=(1.01, 1),
        )
        plt.subplots_adjust(left=0.8)
        plt.tight_layout()
        plt.gca().add_artist(mask_legend)
        return fig

    else:
        raise AssertionError("return_type arg should be numpy | pil | mpl")
