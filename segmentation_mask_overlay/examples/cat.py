import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from segmentation_mask_overlay import overlay_masks


image = Image.open("./segmentation_mask_overlay/examples/cat.jpg").convert("L")
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

# You can plot anything over the provided image
ax, = fig.get_axes()
ax.plot([0, 100], [0, 100], label="line")
ax.legend(loc=2)

fig.savefig("./segmentation_mask_overlay/examples/cat_masked.jpg", bbox_inches="tight", dpi=300)
