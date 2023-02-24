from typing import Tuple, Union, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as colormap


RGBAColor = Union[
    Tuple[np.uint8, np.uint8, np.uint8, np.uint8], Tuple[float, float, float, float]
]


class LabelColor:
    def __init__(
        self,
        num_labels: int,
        mpl_colormap: str = "gist_rainbow",
        alpha: float = 0.3,
        color_mode: str = "rgba",
        mode: str = "uint8",
        return_legend_color: bool = False,
    ):
        """Colormap class. Easy-peazy way to get colos for your segmentation labels.
        Parameters
        ----------
        num_labels : int
            Number of labels to map with colors.
        mpl_colormap : str, optional
            Matplotlib colormap name.
            See all at: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
            , by default "gist_rainbow"
        alpha : float, optional
            Opacity, by default 0.3
        mode : str, optional
            Type of image values. Either uint8 or float, by default "uint8"
        return_legend_color: bool, optional
            If true, getitem retirns tuple of mask color
            and same less opaque legend color.
        """
        
        assert color_mode in ["rgb", "rgba"]
        
        self.cmap = colormap.get_cmap(mpl_colormap, num_labels)
        self.alpha = alpha
        self.mode = mode
        self.color_mode = color_mode
        self.num_labels = num_labels
        self.return_legend_color = return_legend_color
    
    def __len__(self):
        return self.num_labels
    
    def __repr__(self):
        return f"Color bar of {len(self)} colors in {cbar.cmap.name} palitre"

    def __getitem__(self, i: int) -> Union[RGBAColor, Tuple[RGBAColor, RGBAColor]]:
        color: RGBAColor
        if (i < 0) or (i >= self.num_labels):
            raise IndexError(
                "Label index out of scope, set larger `num_labels`."
                + f" Current: {self.num_labels}"
            )

        if self.mode == "float":
            color = self.cmap(i, self.alpha)
            color = color[:-1] if self.color_mode == "rgb" else color
        elif self.mode == "uint8":
            color = np.array(self.cmap(i, self.alpha))
            color = color[:-1] if self.color_mode == "rgb" else color
            color = tuple(np.array(color * 255, dtype=np.uint8))
        else:
            raise AttributeError("Unsupported mode. Use 'uint8' or 'float'")

        if self.return_legend_color:
            return color, self.get_legend_color(i)
        else:
            return color
class LabelColorIter:
    def __init__(self, label_color: LabelColor) -> None:
        self.label_color = label_color
        self._i = 0

    def __next__(self) -> Union[RGBAColor, Tuple[RGBAColor, RGBAColor]]:
        if self._i < len(self.label_color):
            color: Union[RGBAColor, Tuple[RGBAColor, RGBAColor]] = self.label_color[
                self._i
            ]
        else:
            raise StopIteration
        self._i += 1
        return color

    def __iter__(self):
        return 
