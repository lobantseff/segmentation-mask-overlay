import os
from typing import Optional, Tuple, Union, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image as PILImage

from .label_color import LabelColor
from .utils import open_with_PIL


def overlay_masks(
    image: Union[os.PathLike, PILImage.Image, np.ndarray],
    boolean_masks: Union[np.ndarray, List[np.ndarray]],
    labels: Optional[List[str]] = None,
    colors: np.ndarray = None,
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 90,
    mask_alpha: float = 0.4,
    mpl_colormap: str = "tab20",
) -> plt.Figure:
    """Overlays masks on the image.
    Parameters
    ----------
    image : Union[str, PIL.Image.Image, np.ndarray]
        Image path or PIl.Image or numpy array. If image size inconsistent with
        the masks size, image will be resized.
    boolean_masks : List[np.ndarray[bool]]
        List of segmentation masks or numpy array of shape (height, width, n_classes).
        All masks should be the same size, equal to size of the image.
    labels : Optional[List[str]], optional
        Optional label names. Provide in the same order as the corresponding masks.
        If not provided, will be set as range(len(boolean_masks)), by default None
    colors : np.ndarray
        Array of shape (n_labels x 4).
        Example to get persistent colormap: `plt.cm.tab20(np.arange(NUM_LABELS))`
    figsize : tuple, optional
        Size in inches of the output image, by default (12, 12)
    dpi : int, optional
        Resolution of the output image. Note: 'px, py = w * dpi, h * dpi', by default 120
    mask_alpha : float, optional
        Masks opaque value, by default 0.4
    Returns
    -------
    plt.Figure
        Output figure with masks legend.
    """

    if isinstance(boolean_masks, np.ndarray):
        assert (boolean_masks.ndim == 3 and boolean_masks.dtype == bool), (
            "boolean_masks should be a list boolean numpy"
            + " arrays or 3-dim numpy array with the last dim"
            + " as a channel to store masks of different classes"
        )
        boolean_masks = [boolean_masks[:, :, i] for i in range(boolean_masks.shape[-1])]

    if labels is not None:
        assert len(labels) == len(boolean_masks), (
            "Number of provided labels != number of masks"
        )
    else:
        labels = [f"mask_{_}" for _ in range(len(boolean_masks))]

    if colors is not None:
        assert len(colors) == len(boolean_masks), (
            "Number of provided colors != number of masks"
        )

    image = open_with_PIL(image)
    image_size = tuple(np.array(image.size)[::-1])

    assert all(
        mask.shape == image_size for mask in boolean_masks
    ), "Label mask size is not equal to image size"

    if colors is None:
        cbar = LabelColor(
            num_labels=len(boolean_masks),
            alpha=mask_alpha,
            return_legend_color=True,
            mpl_colormap=mpl_colormap,
        )

    else:
        mask_colors = colors.copy()
        mask_colors[:, -1] *= mask_alpha
        mask_colors = (mask_colors * 255).astype("uint8")
        cbar = zip(mask_colors, colors)

    segmentation_overlay = np.zeros((*image_size, 4), dtype=np.uint16)
    segmentation_mask = np.zeros(image_size, dtype=bool)
    legend_elements = []

    for mask, label, (color, legend_color) in zip(boolean_masks, labels, cbar):

        assert mask.dtype == "bool"

        intersection = mask & segmentation_mask
        segmentation_mask = mask | segmentation_mask

        # Paint non-overlapping area
        segmentation_overlay[mask ^ intersection] = color

        # Blend overlapping area
        segmentation_overlay[intersection] = (
            segmentation_overlay[intersection] + color
        ) / 2

        legend_elements.append(Patch(color=legend_color, label=label))

    segmentation_overlay = PILImage.fromarray(segmentation_overlay.astype("uint8"))
    image.paste(segmentation_overlay, mask=segmentation_overlay)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(image)
    plt.axis("off")
    plt.legend(
        handles=legend_elements,
        loc="upper left",
        frameon=False,
        bbox_to_anchor=(1.01, 1),
    )
    plt.subplots_adjust(left=0.8)
    plt.tight_layout()

    return fig
