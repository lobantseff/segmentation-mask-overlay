import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from segmentation_mask_overlay import overlay_masks
import logging
import time
from typing import Optional
from contextlib import contextmanager


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


# [Example] Load image. If you are sure of you masks
image = Image.open("./examples/cat.jpg").convert("RGB")
image = np.array(image)

image = np.random.randint(0, 1024, (512, 512)) * 255

# [Example] Mimic list of masks
masks = []
for i in np.linspace(0, image.shape[1], 10, dtype="int"):
    mask = np.zeros(image.shape[:2], dtype="bool")
    mask[i : i + 100, i : i + 200] = 1
    masks.append(mask)

# [Optional] prepare labels
mask_labels = [f"Mask_{i}" for i in range(len(masks))]

# [Optional] prepare colors
cmap = plt.cm.tab20(np.arange(len(mask_labels)))[..., :-1]

# Laminate your image!
with catchtime("numpy"):
    array = overlay_masks(image, np.stack(masks, -1), mask_labels, return_type="numpy")

with catchtime("pil"):
    img = overlay_masks(image, np.stack(masks, -1), mask_labels, return_type="pil")
img.save("cat_masked_pil.png")

with catchtime("mpl"):
    fig = overlay_masks(image, np.stack(masks, -1), mask_labels, return_type="mpl")

fig.savefig("cat_masked.png", bbox_inches="tight")
