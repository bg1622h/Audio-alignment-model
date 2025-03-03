import io

import matplotlib.pyplot as plt
import numpy as np
import PIL
from torchvision.transforms import ToTensor

plt.switch_backend("agg")  # fix RuntimeError: main thread is not in main loop

num_log_categories = 4


def plot_images(imgs, config):
    """
    Combine several images into one figure.

    Args:
        imgs (Tensor): array of images (B X C x H x W).
        config (DictConfig): hydra experiment config.
    Returns:
        image (Tensor): a single figure with imgs plotted side-to-side.
    """
    # name of each img in the array
    names = config.writer.names
    # figure size
    figsize = config.writer.figsize
    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    for i in range(len(names)):
        # channels must be in the last dim
        img = imgs[i].permute(1, 2, 0)
        axes[i].imshow(img)
        axes[i].set_title(names[i])
        axes[i].axis("off")  # we do not need axis
    # To create a tensor from matplotlib,
    # we need a buffer to save the figure
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    # convert buffer to Tensor
    image = ToTensor()(PIL.Image.open(buf))

    plt.close()

    return image


def create_log_plot(input, target, predict):
    """
    lengths of all the data are the same
    """
    fig, axes = plt.subplots(len(input), num_log_categories, figsize=(12, 12))
    axes[0, 0].set_title("Input")
    for i in range(len(input)):
        axes[i, 0].imshow(input[i], cmap="viridis", interpolation="nearest")
    axes[0, 1].set_title("Target")
    for i in range(len(target)):
        axes[i, 1].imshow(target[i], cmap="viridis", interpolation="nearest")
    axes[0, 2].set_title("Predict")
    for i in range(len(predict)):
        im = axes[i, 2].imshow(
            np.exp(predict[i]), cmap="viridis", interpolation="nearest"
        )
        fig.colorbar(im, ax=axes[i, 2], label="Value")
    for i in range(len(predict)):
        im = axes[i, 3].imshow(
            (np.exp(predict[i]) > 0.5).int(),
            cmap="viridis",
            interpolation="nearest",
        )
        fig.colorbar(im, ax=axes[i, 3], label="Value")
    fig.suptitle("Input, Target, Out, Predict")
    return fig
