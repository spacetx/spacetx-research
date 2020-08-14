import skimage.exposure
import numpy as np
from .namedtuple import ImageBbox
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None
from typing import Optional


def pil_to_numpy(pilfile, mode: str = 'L', reduction_factor: int = 1):
    """ Open file using pillow, and return numpy array with shape:
        w,h,channel if channel > 1
        w,h         if channel ==1.
        Mode can be: 'L', 'RGB', 'I', 'F'
        See https://pillow.readthedocs.io/en/3.0.x/handbook/concepts.html
    """
    assert (mode == 'L' or mode == 'RGB' or mode == 'I' or mode == 'F')

    pilobject = PIL.Image.open(pilfile).convert(mode)

    # Use PIL.Image.LANCZOS pto revent antialiasing when downsampling
    if reduction_factor != 1:
        w_raw, h_raw = pilobject.size
        w_new = int(w_raw / reduction_factor)
        h_new = int(h_raw / reduction_factor)
        pilobject = pilobject.resize((w_new, h_new), resample=PIL.Image.BILINEAR)

    return skimage.exposure.rescale_intensity(image=np.array(pilobject), 
                                              in_range='image', 
                                              out_range='dtype')


def rescale_intensity(img_reference: np.ndarray, 
                      histo_mask: Optional[np.ndarray] = None, 
                      q: tuple = (0.01, 99.99), 
                      gamma: float = 1.0):
    """ Rescale intensity inside the histo_mask so that the intensities occupy the full range
        For gamma greater than 1, the histogram will shift towards left and
        the output image will be darker than the input image.
        For gamma less than 1, the histogram will shift towards right and
        the output image will be brighter than the input image."""

    # Here I am just computing ql,qr
    if histo_mask is None:
        ql, qr = np.percentile(img_reference.reshape(-1), q=q)
    else:
        ql, qr = np.percentile(img_reference[histo_mask].reshape(-1), q=q)
    
    # Define the pre-processing which uses: ql, qr
    def f(x):
        x1 = skimage.exposure.rescale_intensity(x,
                                                in_range=(ql, qr),
                                                out_range='dtype')
        if gamma == 1:
            return x1
        else:
            x2 = skimage.exposure.adjust_gamma(x1, gamma=gamma, gain=1)
            return x2

    img_out = f(img_reference)
    return img_out, f


def compute_sqrt_mse(predictions: np.ndarray, 
                     roi_mask: Optional[np.ndarray] = None):
    if roi_mask is None:
        w,h = predictions.shape[-2:]
        roi_mask = np.ones((w,h), dtype=predictions.dtype)
        
    assert len(roi_mask.shape) == 2
    if len(predictions.shape) == 3:
        # no channel
        diff = (predictions - predictions[0]) * roi_mask
        mse = np.sum(diff ** 2, axis=(-1, -2)) / np.sum(roi_mask)
    elif len(predictions.shape) == 4:
        # yes channel
        c = predictions.shape[-1]
        diff = (predictions - predictions[0]) * roi_mask[..., None]
        mse = np.sum(diff ** 2, axis=(-1, -2, -3)) / (c * np.sum(roi_mask))
    else:
        raise Exception("Yopu should never be here")
    sigma = np.sqrt(mse)
    return sigma


def getLargestCC(mask):
    labels = skimage.measure.label(mask)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def find_bbox(mask):
    assert len(mask.shape) == 2
    row = np.sum(mask, axis=-1) > 0
    col = np.sum(mask, axis=-2) > 0
    max_row = max(np.arange(row.shape[0]) * row) + 1
    min_row = row.shape[0] - max(np.arange(start=row.shape[0], stop=0, step=-1) * row)
    max_col = max(np.arange(col.shape[0]) * col) + 1
    min_col = col.shape[0] - max(np.arange(start=col.shape[0], stop=0, step=-1) * col)
    return ImageBbox(min_row=min_row,
                     min_col=min_col,
                     max_row=max_row,
                     max_col=max_col)


#####def normalize_tensor(image, scale_each_image=False, scale_each_channel=False, in_place=False):
#####    """ Normalize a batch of images to the range 0,1 """
#####
#####    assert len(image.shape) == 4  # batch, ch, w,h
#####
#####    if (not scale_each_image) and (not scale_each_channel):
#####        max_tmp = torch.max(image)
#####        min_tmp = torch.min(image)
#####    elif scale_each_image and (not scale_each_channel):
#####        max_tmp = torch.max(image, dim=-4, keepdim=True)[0]
#####        min_tmp = torch.min(image, dim=-4, keepdim=True)[0]
#####    elif (not scale_each_image) and scale_each_channel:
#####        max_tmp = torch.max(image, dim=-3, keepdim=True)[0]
#####        min_tmp = torch.min(image, dim=-3, keepdim=True)[0]
#####    elif scale_each_image and scale_each_channel:
#####        max_tmp = torch.max(image, dim=-4, keepdim=True)[0].max(dim=-3, keepdim=True)[0]
#####        min_tmp = torch.min(image, dim=-4, keepdim=True)[0].min(dim=-3, keepdim=True)[0]
#####
#####    if in_place:
#####        data = image.clone().clamp_(min=min_tmp, max=max_tmp)  # avoid modifying tensor in-place
#####    else:
#####        data = image.clamp_(min=min_tmp, max=max_tmp)
#####    return data.add_(-min_tmp).div_(max_tmp - min_tmp + 1e-5)
#####
#####
#####def img_pre_processing(pilfile, reduction_factor=1, remove_background=True):
#####    """ Resize and rescale intensities in (0,1) """
#####
#####    # Open and resize using bilinear interpolation
#####    w_raw, h_raw = pilfile.size
#####    w_new = int(w_raw/reduction_factor)
#####    h_new = int(h_raw/reduction_factor)
#####    pilresize = pilfile.convert("F").resize((w_new, h_new), resample=PIL.Image.BILINEAR)
#####    img_np = numpy.array(pilresize)
#####
#####    debug = img_np.flatten()
#####    print(numpy.min(debug), numpy.max(debug))
#####
#####    # Rescale foreground intensity in (0,1)
#####    if remove_background:
#####        image_thresh = skimage.filters.threshold_otsu(img_np)
#####        print(image_thresh)
#####        ql, qr = numpy.percentile(img_np[img_np > image_thresh].flatten(), q=(0, 100))  # note that the statistics are compute on the foreground only
#####        img_tmp = skimage.exposure.rescale_intensity(img_np, in_range=(ql, qr), out_range=(0.0, 1.0))
#####    else:
#####        img_tmp = skimage.exposure.rescale_intensity(img_np, in_range="image", out_range=(0.0, 1.0))
#####
#####    return img_tmp
#####
#####
########def sum_in_windows(img, window_size: int=80):
########    """ returns the sum inside a square of size=window_size with center located at (i,j) """
########    w, h = img.shape[-2:]
########    img_pad = numpy.pad(img, pad_width=window_size//2, mode='constant', constant_values=0)
########    assert (img == img_pad[window_size//2:window_size//2+w,window_size//2:window_size//2+h]).all()
########
########    cum = numpy.cumsum(numpy.cumsum(img_pad, axis=0), axis=1)
########
########    # roll
########    px = numpy.roll(cum, +window_size//2, axis=0)
########    mx = numpy.roll(cum, -window_size//2, axis=0)
########    px_py = numpy.roll(px, +window_size//2, axis=1)
########    px_my = numpy.roll(px, -window_size//2, axis=1)
########    mx_py = numpy.roll(mx, +window_size//2, axis=1)
########    mx_my = numpy.roll(mx, -window_size//2, axis=1)
########
########    # compute sum in square
########    tmp = (px_py - px_my - mx_py + mx_my)
########    return tmp[window_size//2: window_size//2+w, window_size//2: window_size//2+h]
########
########
########def estimate_noise(img: torch.Tensor, radius_nn: int=2):
########    # Compute average first
########    avg = torch.zeros_like(img)
########    n = 0
########    for dx in range(-radius_nn,radius_nn+1):
########        y_tmp = torch.roll(img, dx, dims=-2)
########        for dy in range(-radius_nn,radius_nn+1):
########            y = torch.roll(y_tmp, dy, dims=-1)
########            avg += y
########            n +=1
########    avg = avg.float()/n
########    # print("avg ->",torch.min(avg), torch.max(avg))
########
########    # Compute variance later
########    var = torch.zeros_like(avg)
########    n = 0
########    for dx in range(-radius_nn,radius_nn+1):
########        y_tmp = torch.roll(img, dx, dims=-2)
########        for dy in range(-radius_nn,radius_nn+1):
########            y = torch.roll(y_tmp, dy, dims=-1)
########            var += (y-avg)**2
########            n +=1
########    var = var / (n-1)
########    # print("var ->",torch.min(var), torch.max(var))
########
########    # remove boundaries
########    avg = avg[...,radius_nn+1:-radius_nn-1]
########    var = var[...,radius_nn+1:-radius_nn-1]
########
########    y = torch.sqrt(var[avg>0]).view(-1)
########    x = avg[avg>0].view(-1)
########    return x,y
########
########
########def index_for_binning(input, bins=100, min=0, max=0):
########    if (min == 0) and (max == 0):
########        min = torch.min(input)
########        max = torch.max(input)
########    index = (bins * (input - min).float()/(max-min)).int()
########    return index
########
########
########def compute_average_in_each_bin(x,y,bins=100, x_min=0, x_max=0):
########    assert x.shape == y.shape
########    index = index_for_binning(x, bins=bins, min=x_min, max=x_max)
########    x_stratified = torch.zeros(bins, dtype=x.dtype, device=x.device)
########    y_stratified = torch.zeros(bins, dtype=x.dtype, device=x.device)
########    for i in range(0, bins):
########        x_stratified[i] = x[index == i].mean()
########        y_stratified[i] = y[index == i].mean()
########    return x_stratified, y_stratified
########
########
