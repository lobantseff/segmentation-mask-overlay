import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union

from segmentation_mask_overlay.utils import catchtime
from segmentation_mask_overlay.utils import check_convert_image


Pathlike = Union[str, Path]

def overlay_masks_video(
    im_sequence: np.ndarray,
    *mask_sequences: np.ndarray,
    output: Union[str, Path] = "numpy",
    array_dims: str = "THWC",
    fps: int = 15,

):
    """Create videos out of sequences of images and masks.

    Parameters
    ----------
    savepath: str "path/to/video.mp4"
    im_sequence : np.ndarray of shape THW or THWC
    mask_sequences : numpy arrays of shape THWC of dtype bool
    output: str | Path
        if output == 'numpy' output is a numpy array. If Path, an mp4 file
        will be saved there, default: 'numpy'
    array_dims: str, 
        THWC | TCHW, default: THWC
    fps: int
        default: 15

    Returns
    -------
    Depends on output.
        - if output is a path, ending with .mp4, saves to a file.
        - if output == 'numpy', returns THWC numpy array
    """

    if array_dims == "TCHW":
        ch_dim = -3
    elif array_dims == "THWC":
        ch_dim = -1
    else:
        raise AssertionError("array_dims should be THWC | TCHW")

    if len(mask_sequences) > 0:
        assert all(
            [im_sequence.shape[0] == m.shape[0] for m in mask_sequences]
        ), "Sequence and masks have different size T"

    mask_cmaps = []
    for mseq in mask_sequences:
        num_classes = mseq.shape[ch_dim]
        if num_classes <= 10:
            mask_cmaps.append(plt.cm.tab10(range(10), bytes=True)[:num_classes, :-1][:, ::-1])
        else:
            mask_cmaps.append(plt.cm.rainbow_r(np.linspace(0, 1, num_classes), bytes=True)[:, :-1][:, ::-1])

    video_frames = []
    for im, *masks in zip(im_sequence, *mask_sequences):

        im = check_convert_image(im, input_dims=array_dims[1:])

        masks_im = []
        if len(masks) > 0:
            masks_im = [np.copy(im) for _ in range(len(masks))]
            segmentation_overlay_list = [np.zeros_like(im, dtype=np.uint16) for _ in range(len(masks))]
            segmentation_mask_list = [np.zeros(im.shape[:2], dtype=bool) for _ in range(len(masks))]

            # Inpaint masks into masks_im
            for i in range(len(masks)):
                mask = masks[i].astype(bool)
                mask_im = masks_im[i]
                mask_colormap = mask_cmaps[i]
                segmentation_overlay = segmentation_overlay_list[i]
                segmentation_mask = segmentation_mask_list[i]

                for class_idx in range(mask.shape[ch_dim]):
                    color = mask_colormap[class_idx]
                    class_mask = mask.take(class_idx, ch_dim)

                    intersection = class_mask & segmentation_mask
                    segmentation_mask = class_mask | segmentation_mask

                    # Paint non-overlapping area
                    segmentation_overlay[class_mask ^ intersection] = color

                    # Blend overlapping area
                    segmentation_overlay[intersection] = (segmentation_overlay[intersection] + color) // 2

                masks_im[i] = cv2.addWeighted(mask_im, 1, segmentation_overlay.astype(np.uint8), 0.5, 0)

        frame = np.concatenate((im, *masks_im), 1)
        video_frames.append(frame)

    if isinstance(output, Path) or output[-4:] == ".mp4":
        height, width, _ = video_frames[-1].shape
        out = cv2.VideoWriter(
            str(output),
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=fps,
            frameSize=(width, height),
        )
        for frame in video_frames:
            out.write(frame)
        out.release()
    
    elif output == "numpy":
        video_frames = np.array(video_frames)
        if array_dims == "TCHW":
            video_frames = video_frames.transpose(0, 3, 1, 2)
        return video_frames


if __name__ == "__main__":
    run = np.random.rand(64, 3, 512, 512)
    mask1 = np.random.randint(0, 1, (64, 4, 512, 512))
    mask1[:, 0, :100, :100] = 1
    mask1[:, 1, 50:150, 50:150] = 1
    mask1[:, 2, 130:180, 130:180] = 1
    mask2 = np.random.randint(0, 2, (64, 4, 512, 512))
    with catchtime("masking video"):
        overlay_masks_video(run, mask1, mask2, array_dims="TCHW")