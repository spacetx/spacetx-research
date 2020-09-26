import PIL.Image
import PIL.ImageDraw
import torch
import numpy
from typing import Tuple, Optional
from torchvision import utils
from matplotlib import pyplot as plt
import skimage.color

from .namedtuple import BB


def draw_img(prob: torch.tensor,
             bounding_box: BB,
             big_mask: torch.tensor,
             big_img: torch.tensor,
             big_bg: torch.tensor,
             draw_bg: bool,
             draw_boxes: bool) -> torch.tensor:
    assert len(prob.shape) == 2  # boxes, batch
    assert len(big_mask.shape) == len(big_img.shape) == 5  # boxes, batch, ch, w, h

    rec_imgs_no_bb = (prob[..., None, None, None] * big_mask * big_img).sum(dim=-5)  # sum over boxes
    fg_mask = (prob[..., None, None, None] * big_mask).sum(dim=-5)  # sum over boxes
    background = (1 - fg_mask) * big_bg if draw_bg else torch.zeros_like(big_bg)

    width, height = rec_imgs_no_bb.shape[-2:]

    bounding_boxes = draw_bounding_boxes(prob=prob,
                                         bounding_box=bounding_box,
                                         width=width,
                                         height=height) if draw_boxes else torch.zeros_like(rec_imgs_no_bb)
    mask_no_bb = (torch.sum(bounding_boxes, dim=-3, keepdim=True) == 0)

    return mask_no_bb * (rec_imgs_no_bb + background) + ~mask_no_bb * bounding_boxes


def draw_bounding_boxes(prob: Optional[torch.Tensor], bounding_box: BB, width: int, height: int) -> torch.Tensor:
    # set all prob to one if they are not passed as input
    if prob is None:
        prob = torch.ones_like(bounding_box.bx)

    # checks
    assert prob.shape == bounding_box.bx.shape
    assert len(bounding_box.bx.shape) == 2
    n_boxes, batch_size = bounding_box.bx.shape

    # prepare the storage
    batch_bb_np = numpy.zeros((batch_size, width, height, 3))  # numpy storage for bounding box images

    # compute the coordinates of the bounding boxes and the probability of each box
    x1 = bounding_box.bx - 0.5 * bounding_box.bw
    x3 = bounding_box.bx + 0.5 * bounding_box.bw
    y1 = bounding_box.by - 0.5 * bounding_box.bh
    y3 = bounding_box.by + 0.5 * bounding_box.bh
    assert x1.shape == x3.shape == y1.shape == y3.shape  # n_boxes, batch_size
    x1y1x3y3 = torch.stack((x1, y1, x3, y3), dim=-1)

    # draw the bounding boxes
    for batch in range(batch_size):

        # Draw on PIL
        img = PIL.Image.new('RGB', (width, height), color=0)
        draw = PIL.ImageDraw.Draw(img)
        for box in range(n_boxes):
            # if prob[box, batch] > 0.5:
            if prob[box, batch] > -1:
                draw.rectangle(x1y1x3y3[box, batch, :].cpu().numpy(), outline='red', fill=None)
        batch_bb_np[batch, ...] = numpy.array(img.getdata(), numpy.uint8).reshape((width, height, 3))

    # Transform np to torch, rescale from [0,255] to (0,1)
    batch_bb_torch = torch.from_numpy(batch_bb_np).permute(0, 3, 2, 1).float() / 255  # permute(0,3,2,1) is CORRECT
    return batch_bb_torch.to(bounding_box.bx.device)


def plot_grid(img, figsize=None):
    assert len(img.shape) == 3
    n_max = img.shape[-3]

    row_max = n_max // 4
    if row_max <= 1:
        fig, axes = plt.subplots(ncols=n_max, figsize=figsize)
        for n in range(n_max):
            axes[n].imshow(img[n])
    else:
        fig, axes = plt.subplots(ncols=4, nrows=row_max, figsize=figsize)
        for n in range(4 * row_max):
            row = n // 4
            col = n % 4
            axes[row, col].imshow(img[n])

    plt.close(fig)
    fig.tight_layout()
    return fig


def show_batch(images: torch.Tensor,
               n_col: int = 4,
               n_padding: int = 10,
               title: Optional[str] = None,
               pad_value: int = 1,
               normalize_range: Optional[tuple] = None,
               figsize: Optional[Tuple[float, float]] = None):
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
    plt.imshow(grid.detach().permute(1, 2, 0).squeeze(-1).numpy())
    if isinstance(title, str):
        plt.title(title)
    plt.close(fig)
    fig.tight_layout()

    return fig


def plot_tiling(tiling, figsize: tuple = (12, 12)):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=figsize)
    axes[0, 0].imshow(skimage.color.label2rgb(tiling.integer_mask[0, 0].cpu().numpy(),
                                              numpy.zeros_like(tiling.integer_mask[0, 0].cpu().numpy()),
                                              alpha=1.0,
                                              bg_label=0))
    axes[0, 1].imshow(skimage.color.label2rgb(tiling.integer_mask[0, 0].cpu().numpy(),
                                              tiling.raw_image[0, 0].cpu().numpy(),
                                              alpha=0.25,
                                              bg_label=0))
    axes[1, 0].imshow(tiling.fg_prob[0, 0].cpu().numpy(), cmap='gray')
    axes[1, 1].imshow(tiling.raw_image[0].cpu().permute(1, 2, 0).squeeze(-1).numpy(), cmap='gray')

    axes[0, 0].set_title("sample integer mask")
    axes[0, 1].set_title("sample integer mask")
    axes[1, 0].set_title("fg prob")
    axes[1, 1].set_title("raw image")
    fig.tight_layout()
    plt.close(fig)

    return fig


def plot_loss(history_dict: dict, test_frequency: int = 5):

    x = numpy.arange(0, len(history_dict["test_loss"])*test_frequency, test_frequency)

    fig, ax = plt.subplots()
    ax.plot(history_dict["train_loss"], '-', label="train loss")
    ax.plot(x, history_dict["test_loss"], '.--', label="test loss")

    ax.set_xlabel('epoch')
    ax.set_ylabel('LOSS = - ELBO')
    ax.set_title('Training procedure')
    ax.grid()
    ax.legend()
    plt.close()
    fig.tight_layout()
    return fig


def plot_kl(history_dict: dict, train_or_test: str = "test"):
    if train_or_test == "test":
        kl_instance = history_dict["test_kl_instance"]
        kl_where = history_dict["test_kl_where"]
        kl_logit = history_dict["test_kl_logit"]
    elif train_or_test == "train":
        kl_instance = history_dict["train_kl_instance"]
        kl_where = history_dict["train_kl_where"]
        kl_logit = history_dict["train_kl_logit"]
    else:
        raise Exception

    fig, ax = plt.subplots()
    ax.plot(kl_instance, '-', label="kl_instance")
    ax.plot(kl_where, '.-', label="kl_where")
    ax.plot(kl_logit, '.--', label="kl_logit")
    ax.set_xlabel('epoch')
    ax.set_ylabel('kl')
    ax.grid()
    ax.legend()
    plt.close()
    fig.tight_layout()
    return fig


def plot_sparsity(history_dict: dict, train_or_test: str = "test"):
    if train_or_test == "test":
        sparsity_mask = history_dict["test_sparsity_mask"]
        sparsity_box = history_dict["test_sparsity_box"]
        sparsity_prob = history_dict["test_sparsity_prob"]
    elif train_or_test == "train":
        sparsity_mask = history_dict["train_sparsity_mask"]
        sparsity_box = history_dict["train_sparsity_box"]
        sparsity_prob = history_dict["train_sparsity_prob"]
    else:
        raise Exception

    fig, ax = plt.subplots()
    ax.plot(sparsity_mask, '-', label="sparsity_mask")
    ax.plot(sparsity_box, '.-', label="sparsity_box")
    ax.plot(sparsity_prob, '.--', label="sparsity_prob")
    ax.set_xlabel('epoch')
    ax.set_ylabel('sparsity')
    ax.grid()
    ax.legend()
    plt.close()
    fig.tight_layout()
    return fig

def plot_loss_term(history_dict: dict, train_or_test: str = "test"):
    if train_or_test == "test":
        loss = history_dict["test_loss"]
        mse = history_dict["test_mse_tot"]
        reg = history_dict["test_reg_tot"]
        kl = history_dict["test_kl_tot"]
        sparsity = history_dict["test_sparsity_tot"]
    elif train_or_test == "train":
        loss = history_dict["test_loss"]
        mse = history_dict["test_mse_tot"]
        reg = history_dict["test_reg_tot"]
        kl = history_dict["test_kl_tot"]
        sparsity = history_dict["test_sparsity_tot"]
    else:
        raise Exception

    FROM HERE

ax6.plot(loss,'-',label='loss')
ax6.plot(f_geco_sparsity * sparsity_raw,'x-',label='scaled_sparsity')
ax6.plot(f_geco_balance * reg_raw,'x-',label='scaled_reg')
ax6.plot(f_geco_balance * mse_raw,'x-',label='scaled_mse')
ax6.plot((1-f_geco_balance) * kl_raw,'x-',label='scaled_kl')
ax6.set_ylim([0, 1.01*max(loss[epoch_min:epoch_max])])
ax6.set_xlim([epoch_min, epoch_max])
ax6.grid()
ax6.legend()



