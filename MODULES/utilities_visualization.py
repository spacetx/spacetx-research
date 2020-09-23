import torch
from typing import Tuple, Optional
from torchvision import utils
from matplotlib import pyplot as plt
from neptunecontrib.api import log_chart


def plot_grid(img, figsize=None):
    assert len(img.shape) == 3
    N = img.shape[-3]

    MAX_row = N // 4
    if MAX_row <= 1:
        figure, axes = plt.subplots(ncols=N, figsize=figsize)
        for n in range(N):
            axes[n].imshow(img[n])
    else:
        figure, axes = plt.subplots(ncols=4, nrows=MAX_row, figsize=figsize)
        for n in range(4 * MAX_row):
            row = n // 4
            col = n % 4
            axes[row, col].imshow(img[n])


def show_batch(images: torch.Tensor,
               n_col: int = 4,
               n_padding: int = 10,
               title: Optional[str] = None,
               pad_value: int = 1,
               normalize_range: Optional[tuple] = None,
               figsize: Optional[Tuple[float, float]] = None,
               name_for_neptune: Optional[str] = None):
    """Visualize a torch tensor of shape: (batch x ch x width x height) """
    assert len(images.shape) == 4  # batch, ch, width, height
    if images.device != "cpu":
        images = images.cpu()
    if normalize_range is None:
        grid = utils.make_grid(images, n_col, n_padding, normalize=False, pad_value=pad_value)
    else:
        grid = utils.make_grid(images, n_col, n_padding, normalize=True, range=normalize_range,
                               scale_each=False, pad_value=pad_value)
        
    fig = plt.figure(figsize=figsize)
    plt.imshow(grid.detach().permute(1,2,0).squeeze(-1).numpy())
    if isinstance(title, str):
        plt.title(title)
    plt.close(fig)
    fig.tight_layout()

    if name_for_neptune is not None:
        log_chart(name='name_for_neptune', chart=fig)
    return fig

