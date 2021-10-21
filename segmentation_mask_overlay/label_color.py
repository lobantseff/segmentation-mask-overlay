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

        self.cmap = colormap.get_cmap(mpl_colormap, num_labels)
        self.alpha = alpha
        self.mode = mode
        self.num_labels = num_labels
        self.return_legend_color = return_legend_color

    def __getitem__(self, i: int) -> Union[RGBAColor, Tuple[RGBAColor, RGBAColor]]:
        color: RGBAColor
        if (i < 0) or (i >= self.num_labels):
            raise IndexError(
                "Label index out of scope, set larger `num_labels`."
                + f" Current: {self.num_labels}"
            )

        if self.mode == "float":
            color = self.cmap(i, self.alpha)
        elif self.mode == "uint8":
            color = np.array(self.cmap(i, self.alpha))
            color = tuple(np.array(color * 255, dtype=np.uint8))
        else:
            raise AttributeError("Unsupported mode. Use 'uint8' or 'float'")

        if self.return_legend_color:
            return color, self.get_legend_color(i)
        else:
            return color

    def __len__(self):
        return self.num_labels

    def __iter__(self):
        return LabelColorIter(self)

    def get_legend_color(self, i, alpha=0.8):
        """Return more opaque version of the same color
        Parameters
        ----------
        i : int
            color index
        alpha : float, optional
            Opaque coefficient, by default 0.8
        Returns
        -------
        tuple
            RGBA color in tuple.
        """

        return self.cmap(i, alpha)

    @staticmethod
    def blend(*args: RGBAColor) -> RGBAColor:
        """Blends colors.
        Returns
        -------
        RGBAColor
        """

        if isinstance(args[0][0], int):
            return tuple(np.array(args, dtype="uint16").mean(0).astype("uint8"))
        else:
            return tuple(np.array(args).mean(0))


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
        return self
