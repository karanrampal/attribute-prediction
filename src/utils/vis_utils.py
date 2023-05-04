"""Utilities for visualization"""

import io
import math
from typing import List, Tuple

import gcsfs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

plt.style.use("dark_background")


def fetch_data(
    data: pd.DataFrame, num_samples: int, gfs: gcsfs.core.GCSFileSystem
) -> List[np.ndarray]:
    """Fetch num_samples of data given the label
    Args:
        data: Dataframe of paths
        num_samples: Number of samples to fetch
    """
    sample_paths = data.sample(n=num_samples).tolist()

    img_list = []
    for path in sample_paths:
        with Image.open(io.BytesIO(gfs.open(path).read())) as img:
            img_list.append(np.array(img))

    return img_list


def vis_data(
    img_list: List[np.ndarray],
    title: str,
    num_cols: int = 3,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """Visualze the data
    Args:
        img_list: Input image list
        title: Title of the figure
        num_cols: Number of columns in the visuazilation grid
        figsize: Figure size
    """
    num = len(img_list)
    num_rows = math.ceil((num + 1) / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.suptitle(title, fontsize=20)
    axs = axs.flatten()

    for i, axi in enumerate(axs):
        if i < num:
            axi.imshow(img_list[i])
        axi.axis("off")
    plt.tight_layout()
