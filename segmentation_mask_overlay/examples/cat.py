import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from segmentation_mask_overlay import overlay_masks


image = Image.open("./cat.jpg").convert("L")
image = np.array(image)

# Mimic masks
masks = []
for i in np.linspace(0, image.shape[1], 10, dtype="int"):
    mask = np.zeros(image.shape, dtype="bool")
    mask[i : i + 100, i : i + 200] = 1
    masks.append(mask)

mask_labels = [f"Mask_{i}" for i in range(len(masks))]
cmap = plt.cm.tab20(np.arange(len(mask_labels)))

fig = overlay_masks(image, masks, labels=mask_labels, colors=cmap, mask_alpha=0.5)
fig.savefig("./cat_masked.jpg", bbox_inches="tight", dpi=300)
