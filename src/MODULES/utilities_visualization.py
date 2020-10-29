import PIL.Image
import PIL.ImageDraw
import torch
import numpy
from typing import Tuple, Optional, Union
from torchvision import utils
from matplotlib import pyplot as plt
import skimage.color
import skimage.morphology
import neptune

from MODULES.namedtuple import BB, Output, Segmentation
from MODULES.utilities_neptune import log_img_and_chart, log_img_only


def contours_from_labels(labels: numpy.ndarray,
                         contour_thickness: int = 1) -> numpy.ndarray:
    assert len(labels.shape) == 2
    assert contour_thickness >= 1
    contours = (skimage.morphology.dilation(labels) != labels)

    for i in range(1, contour_thickness):
        contours = skimage.morphology.binary_dilation(contours)
    return contours


def add_red_contours(image: numpy.ndarray, contours: numpy.ndarray) -> numpy.ndarray:
    assert contours.dtype == bool
    if (len(image.shape) == 3) and (image.shape[-1] == 3):
        image_with_contours = image
    elif len(image.shape) == 2:
        image_with_contours = skimage.color.gray2rgb(image)
    else:
        raise Exception
    image_with_contours[contours, 0] = 1
    image_with_contours[contours, 1:] = 0
    return image_with_contours


def plot_label_contours(label: Union[torch.Tensor, numpy.ndarray],
                        image: Union[torch.Tensor, numpy.ndarray],
                        contour_thickness: int = 2,
                        figsize: tuple = (24, 24),
                        experiment: Optional[neptune.experiments.Experiment] = None,
                        neptune_name: Optional[str] = None):

    assert len(label.shape) == 2
    if torch.is_tensor(label):
        label = label.cpu().numpy()

    if torch.is_tensor(image):
        if len(image.shape) == 3:
            image = image.permute(1, 2, 0).cpu().numpy()
        else:
            image = image.cpu().numpy()
    if len(image.shape) == 3 and (image.shape[-1] != 3):
        image = image[..., 0]

    assert image.shape[:2] == label.shape[:2]

    contours = contours_from_labels(label, contour_thickness)
    fig, ax = plt.subplots(ncols=3, figsize=figsize)
    ax[0].imshow(image)
    ax[1].imshow(add_red_contours(image, contours))
    ax[2].imshow(label, cmap='gray')

    fig.tight_layout()
    if neptune_name is not None:
        #log_img_and_chart(name=neptune_name, fig=fig, experiment=experiment)
        log_img_only(name=neptune_name, fig=fig, experiment=experiment)
    plt.close(fig)
    return fig


def draw_img(bounding_box: BB,
             mixing_k: torch.tensor,
             big_img: torch.tensor,
             big_bg: torch.tensor,
             draw_bg: bool,
             draw_boxes: bool,
             c: Optional[torch.tensor] = None) -> torch.tensor:
    if c is None:
        c = torch.ones_like(bounding_box.bx).bool()
    assert len(c.shape) == 2  # boxes, batch
    assert len(mixing_k.shape) == len(big_img.shape) == 5  # boxes, batch, ch, w, h

    rec_imgs_no_bb = (c[..., None, None, None] * mixing_k * big_img).sum(dim=-5)  # sum over boxes
    fg_mask = (c[..., None, None, None] * mixing_k).sum(dim=-5)  # sum over boxes
    background = (1 - fg_mask) * big_bg if draw_bg else torch.zeros_like(big_bg)

    width, height = rec_imgs_no_bb.shape[-2:]

    bounding_boxes = draw_bounding_boxes(bounding_box=bounding_box,
                                         width=width,
                                         height=height,
                                         c=c) if draw_boxes else torch.zeros_like(rec_imgs_no_bb)
    mask_no_bb = (torch.sum(bounding_boxes, dim=-3, keepdim=True) == 0)

    return mask_no_bb * (rec_imgs_no_bb + background) + ~mask_no_bb * bounding_boxes


def draw_bounding_boxes(bounding_box: BB, width: int, height: int, c: Optional[torch.Tensor] = None) -> torch.Tensor:
    # set all prob to one if they are not passed as input
    if c is None:
        c = torch.ones_like(bounding_box.bx).bool()

    # checks
    assert c.shape == bounding_box.bx.shape
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
            if c[box, batch]:
                draw.rectangle(x1y1x3y3[box, batch, :].cpu().numpy(), outline='red', fill=None)
        batch_bb_np[batch, ...] = numpy.array(img.getdata(), numpy.uint8).reshape((width, height, 3))

    # Transform np to torch, rescale from [0,255] to (0,1)
    batch_bb_torch = torch.from_numpy(batch_bb_np).permute(0, 3, 2, 1).float() / 255  # permute(0,3,2,1) is CORRECT
    return batch_bb_torch.to(bounding_box.bx.device)


def plot_grid(img,
              figsize: Optional[Tuple[float, float]] = None,
              experiment: Optional[neptune.experiments.Experiment] = None,
              neptune_name: Optional[str] = None):

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

    fig.tight_layout()
    if neptune_name is not None:
        #log_img_and_chart(name=neptune_name, fig=fig, experiment=experiment)
        log_img_only(name=neptune_name, fig=fig, experiment=experiment)
    plt.close(fig)
    return fig


def show_batch(images: torch.Tensor,
               n_col: int = 4,
               n_padding: int = 10,
               title: Optional[str] = None,
               pad_value: int = 1,
               normalize_range: Optional[tuple] = None,
               figsize: Optional[Tuple[float, float]] = None,
               experiment: Optional[neptune.experiments.Experiment] = None,
               neptune_name: Optional[str] = None):

    """Visualize a torch tensor of shape: (batch x ch x width x height) """
    assert len(images.shape) == 4  # batch, ch, width, height
    if images.device != "cpu":
        images = images.cpu()

    # Always normalize the image in (0,1) either using min_max of tensor or normalize_range
    grid = utils.make_grid(images, n_col, n_padding, normalize=True, range=normalize_range,
                           scale_each=False, pad_value=pad_value)
        
    fig = plt.figure(figsize=figsize)
    plt.imshow(grid.detach().permute(1, 2, 0).squeeze(-1).numpy())
    if isinstance(title, str):
        plt.title(title)
    fig.tight_layout()

    if neptune_name is not None:
        #log_img_and_chart(name=neptune_name, fig=fig, experiment=experiment)
        log_img_only(name=neptune_name, fig=fig, experiment=experiment)

    plt.close(fig)
    return fig


def plot_tiling(tiling,
                figsize: tuple = (12, 12),
                experiment: Optional[neptune.experiments.Experiment] = None,
                neptune_name: Optional[str] = None):

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
    if neptune_name is not None:
        #log_img_and_chart(name=neptune_name, fig=fig, experiment=experiment)
        log_img_only(name=neptune_name, fig=fig, experiment=experiment)
    plt.close(fig)
    return fig


def plot_loss(history_dict: dict,
              test_frequency: int = 5,
              experiment: Optional[neptune.experiments.Experiment] = None,
              neptune_name: Optional[str] = None):

    x = numpy.arange(0, len(history_dict["test_loss"])*test_frequency, test_frequency)
    train_loss = history_dict["train_loss"]
    test_loss = history_dict["test_loss"]

    fig, ax = plt.subplots()
    ax.plot(train_loss, '-', label="train loss")
    ax.plot(x, test_loss, '.--', label="test loss")

    ax.set_xlabel('epoch')
    ax.set_ylabel('LOSS = - ELBO')
    ax.set_title('Training procedure')
    ax.grid()
    ax.legend()
    fig.tight_layout()
    if neptune_name is not None:
        log_img_and_chart(name=neptune_name, fig=fig, experiment=experiment)
    plt.close()
    return fig


def plot_kl(history_dict: dict,
            train_or_test: str = "test",
            experiment: Optional[neptune.experiments.Experiment] = None,
            neptune_name: Optional[str] = None):

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
    fig.tight_layout()
    if neptune_name is not None:
        log_img_and_chart(name=neptune_name, fig=fig, experiment=experiment)
    plt.close()
    return fig


def plot_sparsity(history_dict: dict,
                  train_or_test: str = "test",
                  experiment: Optional[neptune.experiments.Experiment] = None,
                  neptune_name: Optional[str] = None):

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
    fig.tight_layout()
    if neptune_name is not None:
        log_img_and_chart(name=neptune_name, fig=fig, experiment=experiment)
    plt.close()
    return fig


def plot_loss_term(history_dict: dict,
                   train_or_test: str = "test",
                   experiment: Optional[neptune.experiments.Experiment] = None,
                   neptune_name: Optional[str] = None):

    if train_or_test == "test":
        loss = numpy.array(history_dict["test_loss"])
        mse = numpy.array(history_dict["test_mse_tot"])
        reg = numpy.array(history_dict["test_reg_tot"])
        kl = numpy.array(history_dict["test_kl_tot"])
        sparsity = numpy.array(history_dict["test_sparsity_tot"])
        geco_sparsity = numpy.array(history_dict["test_geco_sparsity"])
        geco_balance = numpy.array(history_dict["test_geco_balance"])
    elif train_or_test == "train":
        loss = numpy.array(history_dict["train_loss"])
        mse = numpy.array(history_dict["train_mse_tot"])
        reg = numpy.array(history_dict["train_reg_tot"])
        kl = numpy.array(history_dict["train_kl_tot"])
        sparsity = numpy.array(history_dict["train_sparsity_tot"])
        geco_sparsity = numpy.array(history_dict["train_geco_sparsity"])
        geco_balance = numpy.array(history_dict["train_geco_balance"])
    else:
        raise Exception

    fig, ax = plt.subplots()
    ax.plot(loss, '-', label="loss")
    ax.plot(geco_balance * mse, '.-', label="scaled mse")
    ax.plot(geco_balance * reg, '.--', label="scaled reg")
    ax.plot((1-geco_balance) * kl, '.--', label="scaled kl")
    ax.plot(geco_sparsity * sparsity, '.--', label="scaled sparsity")

    ax.set_xlabel('epoch')
    ax.set_ylabel('loss term')
    ax.grid()
    ax.legend()
    fig.tight_layout()
    if neptune_name is not None:
        log_img_and_chart(name=neptune_name, fig=fig, experiment=experiment)
    plt.close()
    return fig


def plot_trajectory(history_dict: dict,
                    train_or_test: str = "test",
                    experiment: Optional[neptune.experiments.Experiment] = None,
                    neptune_name: Optional[str] = None):

    if train_or_test == "test":
        mse = history_dict["test_mse_tot"]
        kl = history_dict["test_kl_tot"]
        sparsity = history_dict["test_sparsity_tot"]
    elif train_or_test == "train":
        mse = history_dict["train_mse_tot"]
        kl = history_dict["train_kl_tot"]
        sparsity = history_dict["train_sparsity_tot"]
    else:
        raise Exception

    fontsize = 20
    labelsize = 20
    colors = numpy.arange(0.0, len(mse), 1.0)/len(mse)

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection='3d')

    ax1.set_xlabel('MSE', fontsize=fontsize)
    ax1.set_ylabel('KL', fontsize=fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=labelsize)
    ax1.scatter(mse, kl, c=colors)
    ax1.plot(mse, kl, '--')
    ax1.grid()

    ax2.set_xlabel('SPARSITY', fontsize=fontsize)
    ax2.set_ylabel('MSE', fontsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=labelsize)
    ax2.scatter(sparsity, mse, c=colors)
    ax2.plot(sparsity, mse, '--')
    ax2.grid()

    ax3.set_xlabel('SPARSITY', fontsize=fontsize)
    ax3.set_ylabel('KL', fontsize=fontsize)
    ax3.tick_params(axis='both', which='major', labelsize=labelsize)
    ax3.scatter(sparsity, kl, c=colors)
    ax3.plot(sparsity, kl, '--')
    ax3.grid()

    ax4.scatter(kl, sparsity, mse, c=colors)
    ax4.plot(kl, sparsity, mse, '--')
    ax4.set_xlabel('KL', fontsize=fontsize)
    ax4.set_ylabel('SPARSITY', fontsize=fontsize)
    ax4.set_zlabel('MSE', fontsize=fontsize)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if neptune_name is not None:
        log_img_and_chart(name=neptune_name, fig=fig, experiment=experiment)
    plt.close()
    return fig


def plot_geco_parameters(history_dict: dict,
                         params: dict,
                         train_or_test: str = "test",
                         experiment: Optional[neptune.experiments.Experiment] = None,
                         neptune_name: Optional[str] = None):

    if train_or_test == "train":
        fg_fraction = history_dict["train_fg_fraction"]
        geco_sparsity = history_dict["train_geco_sparsity"]
        mse_av = history_dict["train_mse_tot"]
        geco_balance = history_dict["train_geco_balance"]
    elif train_or_test == "test":
        fg_fraction = history_dict["test_fg_fraction"]
        geco_sparsity = history_dict["test_geco_sparsity"]
        mse_av = history_dict["test_mse_tot"]
        geco_balance = history_dict["test_geco_balance"]
    else:
        raise Exception

    fontsize = 20
    labelsize = 20
    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    color = 'tab:red'
    ax1.set_xlabel('epochs', fontsize=fontsize)
    ax1.set_ylabel('fg_fraction', fontsize=fontsize, color=color)
    ax1.tick_params(axis='both', which='major', labelsize=labelsize)
    ax1.plot(fg_fraction, '.--', color=color, label="n_object")
    ymin = min(params["GECO_loss"]['target_fg_fraction'])
    ymax = max(params["GECO_loss"]['target_fg_fraction'])
    ax1.plot(ymin * numpy.ones(len(fg_fraction)), '-', color='black', label="y_min")
    ax1.plot(ymax * numpy.ones(len(fg_fraction)), '-', color='black', label="y_max")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid()

    ax1b = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax1b.set_xlabel('epochs', fontsize=fontsize)
    ax1b.set_ylabel('geco_sparsity', color=color, fontsize=fontsize)
    ax1b.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.plot(geco_sparsity, '-', label="geco_sparsity", color=color)
    ax1b.tick_params(axis='y', labelcolor=color)
    ax1b.grid()

    # ----------------
    color = 'tab:red'
    ax2.set_xlabel('epochs', fontsize=fontsize)
    ax2.set_ylabel('mse av', fontsize=fontsize, color=color)
    ax2.tick_params(axis='both', which='major', labelsize=labelsize)
    ax2.plot(mse_av, '.--', color=color, label="mse av")

    ymin = min(params["GECO_loss"]["target_mse"])
    ymax = max(params["GECO_loss"]["target_mse"])
    ax2.plot(ymin * numpy.ones(len(mse_av)), '-', color='black', label="y_min")
    ax2.plot(ymax * numpy.ones(len(mse_av)), '-', color='black', label="y_max")
    ax2.tick_params(axis='y', labelcolor=color)

    ax2.grid()
    ax2b = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax2b.set_xlabel('epochs', fontsize=fontsize)
    ax2b.set_ylabel('geco_balance', fontsize=fontsize, color=color)
    plt.plot(geco_balance, '-', label="geco_balance", color=color)
    ax2b.tick_params(axis='both', which='major', labelsize=labelsize)
    ax2b.tick_params(axis='y', labelcolor=color)
    ax2b.grid()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if neptune_name is not None:
        log_img_only(name=neptune_name, fig=fig, experiment=experiment)
    plt.close()
    return fig


def plot_all_from_dictionary(history_dict: dict,
                             params: dict,
                             test_frequency: int = 5,
                             train_or_test: str = "test",
                             experiment: Optional[neptune.experiments.Experiment] = None,
                             verbose: bool = False):
    if verbose:
        print("in plot_all_from_dictionary ->"+train_or_test)

    plot_loss(history_dict, test_frequency=test_frequency, experiment=experiment, neptune_name="loss_history_"+train_or_test)
    plot_kl(history_dict, train_or_test=train_or_test, experiment=experiment, neptune_name="kl_history_"+train_or_test)
    plot_sparsity(history_dict, train_or_test=train_or_test, experiment=experiment, neptune_name="sparsity_history_"+train_or_test)
    plot_loss_term(history_dict, train_or_test=train_or_test, experiment=experiment, neptune_name="loss_terms_"+train_or_test)
    plot_trajectory(history_dict, train_or_test=train_or_test, experiment=experiment, neptune_name="trajectory_"+train_or_test)
    plot_geco_parameters(history_dict, params, train_or_test=train_or_test, experiment=experiment, 
            neptune_name="geco_params_trajectory_"+train_or_test)

    if verbose:
        print("leaving plot_all_from_dictionary ->"+train_or_test)


def plot_generation(output: Output,
                    epoch: int,
                    prefix: str = "",
                    postfix: str = "",
                    verbose: bool = False):
    if verbose:
        print("in plot_reconstruction_and_inference")

    _ = show_batch(output.imgs[:8],
                   n_col=4,
                   n_padding=4,
                   normalize_range=(0.0, 1.0),
                   title='imgs, epoch= {0:6d}'.format(epoch),
                   neptune_name=prefix+"imgs"+postfix)
    _ = show_batch(output.inference.sample_c_map[:8].float(),
                   n_col=4,
                   n_padding=4,
                   normalize_range=(0.0, 1.0),
                   title='c_map, epoch= {0:6d}'.format(epoch),
                   neptune_name=prefix+"c_map"+postfix)
    _ = show_batch(output.inference.big_bg[:8],
                   n_col=4,
                   n_padding=4,
                   normalize_range=(0.0, 1.0),
                   title='background, epoch= {0:6d}'.format(epoch),
                   neptune_name=prefix+"bg"+postfix)

    if verbose:
        print("leaving plot_generation")


def plot_reconstruction_and_inference(output: Output,
                                      epoch: int,
                                      prefix: str = "",
                                      postfix: str = "",
                                      verbose: bool = False):
    if verbose:
        print("in plot_reconstruction_and_inference")

    _ = show_batch(output.imgs[:8],
                   n_col=4,
                   n_padding=4,
                   normalize_range=(0.0, 1.0),
                   title='imgs, epoch= {0:6d}'.format(epoch),
                   neptune_name=prefix+"imgs"+postfix)
    _ = show_batch(output.inference.sample_c_map[:8].float(),
                   n_col=4,
                   n_padding=4,
                   normalize_range=(0.0, 1.0),
                   title='c_map, epoch= {0:6d}'.format(epoch),
                   neptune_name=prefix+"c_map"+postfix)
    _ = show_batch(output.inference.prob_map[:8],
                   n_col=4,
                   n_padding=4,
                   normalize_range=(0.0, 1.0),
                   title='p_map, epoch= {0:6d}'.format(epoch),
                   neptune_name=prefix+"p_map"+postfix)
    _ = show_batch(output.inference.big_bg[:8],
                   n_col=4,
                   n_padding=4,
                   normalize_range=(0.0, 1.0),
                   title='background, epoch= {0:6d}'.format(epoch),
                   neptune_name=prefix+"bg"+postfix)
    _ = show_batch(output.inference.area_map[:8],
                   n_col=4,
                   n_padding=4,
                   normalize_range=None,  # use min_max of tensor
                   title='area_map, epoch= {0:6d}'.format(epoch),
                   neptune_name=prefix + "area_map" + postfix)

    if verbose:
        print("leaving plot_reconstruction_and_inference")


def plot_segmentation(segmentation: Segmentation,
                      epoch: Union[int, str] = "",
                      prefix: str = "",
                      postfix: str = "",
                      verbose: bool = False):
    if verbose:
        print("in plot_segmentation")

    if isinstance(epoch, int):
        title_postfix = 'epoch= {0:6d}'.format(epoch)
    elif isinstance(epoch, str):
        title_postfix = epoch
    else:
        raise Exception

    _ = show_batch(segmentation.integer_mask.float(),
                   n_padding=4,
                   normalize_range=None,  # use min_max of tensor
                   figsize=(12, 12),
                   title='integer_mask, '+title_postfix,
                   neptune_name=prefix+"integer_mask"+postfix)
    _ = show_batch(segmentation.fg_prob,
                   n_padding=4,
                   normalize_range=(0.0, 1.0),
                   figsize=(12, 12),
                   title='fg_prob, '+title_postfix,
                   neptune_name=prefix+"fg_prob"+postfix)

    if verbose:
        print("leaving plot_segmentation")


def plot_concordance(concordance,
                     figsize: tuple = (12, 12),
                     experiment: Optional[neptune.experiments.Experiment] = None,
                     neptune_name: Optional[str] = None):
    fig, axes = plt.subplots(figsize=figsize)
    axes.imshow(concordance.intersection_mask, cmap='gray')
    axes.set_title("intersection mask, iou=" + str(concordance.iou))

    fig.tight_layout()
    if neptune_name is not None:
        log_img_only(name=neptune_name, fig=fig, experiment=experiment)
    plt.close(fig)
    return fig

