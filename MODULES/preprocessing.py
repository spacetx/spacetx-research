import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

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
    array = skimage.exposure.rescale_intensity(image=np.array(pilobject), 
                                               in_range='image', 
                                               out_range='dtype')
    
    return array


def gaussian_smoothing(img: np.ndarray, sigma_gaussian_filter: np.ndarray = np.arange(0,10)):
    
    # sigma=0 has no smoothing, i.e. original image
    smoothed = np.zeros([sigma_gaussian_filter.shape[0]]+list(img.shape))
    multichannel = (len(img.shape)==3)
    for n,s in enumerate(sigma_gaussian_filter):
        smoothed[n] = skimage.filters.gaussian(img, sigma=s, multichannel=multichannel)
    return smoothed


def compute_sqrt_mse(smoothed: np.ndarray, 
                     roi_mask: Optional[np.ndarray] = None):
    if roi_mask is None:
        w,h = smoothed.shape[-2:]
        roi_mask = np.ones((w,h), dtype=smoothed.dtype)
        
    assert len(roi_mask.shape) == 2
    if len(smoothed.shape) == 3:
        # no channel
        diff = (smoothed - smoothed[0]) * roi_mask
        mse = np.sum(diff ** 2, axis=(-1, -2)) / np.sum(roi_mask)
    elif len(smoothed.shape) == 4:
        # yes channel
        c = smoothed.shape[-1]
        diff = (smoothed - smoothed[0]) * roi_mask[..., None]
        mse = np.sum(diff ** 2, axis=(-1, -2, -3)) / (c * np.sum(roi_mask))
    else:
        raise Exception("Yopu should never be here")
    return np.sqrt(mse)


def getLargestCC(mask):
    labels = skimage.measure.label(mask)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = (labels == np.argmax(np.bincount(labels.flat)[1:]) + 1)
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


#-----------------------------------------
#-- VISUALIZATION HELPER FUNCTIONS -------
#-----------------------------------------

def show_random_examples(img: np.ndarray,
                         nexamples: int = 9, 
                         ncols: int = 3, 
                         size: int = 200,
                         figsize: tuple = (12,12)):
    
    nrows=int(np.ceil(nexamples/ncols))
    iw_array = np.random.randint(low=0, high=img.shape[-2]-size, size=nexamples, dtype=int)
    ih_array = np.random.randint(low=0, high=img.shape[-1]-size, size=nexamples, dtype=int)

    if nrows == 1:
        figure, axes = plt.subplots(ncols=ncols, figsize=figsize)
        for n in range(nexamples):
            axes[n].imshow(img[iw_array[n]:iw_array[n]+size,
                               ih_array[n]:ih_array[n]+size], cmap='gray')
    else:
        figure, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
        for n in range(nexamples):
            r = int(n / ncols)
            c = np.mod(n,ncols)
            axes[r,c].imshow(img[iw_array[n]:iw_array[n]+size,
                                 ih_array[n]:ih_array[n]+size], cmap='gray')
        
def plot_img_and_histo(img: np.ndarray, 
                       img_gray: Optional[np.ndarray] = None,
                       histo_mask: Optional[np.ndarray] = None, 
                       figsize=(24, 12)):
    if histo_mask is None:
        histo_mask = np.ones(img.shape[:2], dtype=bool)
    
    if img_gray is None:
        figure, axes = plt.subplots(ncols=2, figsize=figsize)
        a0 = axes[0]
        a1 = axes[1]
      
    else:
        figure, axes = plt.subplots(ncols=2, nrows=2, figsize=figsize)
        a0 = axes[0,0]
        a1 = axes[0,1]
        a2 = axes[1,0]
        a3 = axes[1,1]
        _ = a2.imshow(img_gray, cmap='gray')
        _ = a3.hist(img_gray[histo_mask].reshape(-1), density=True, bins=100)
        _ = a2.set_title("image_GRAY")
        _ = a3.set_title("histogram image_GRAY")
        
    _ = a0.imshow(img)
    _ = a1.hist(img[histo_mask].reshape(-1), density=True, bins=100)
    _ = a0.set_title("image")
    _ = a1.set_title("histogram image")
    
def plot_img_and_zoom(window: tuple,
                      img: Optional[np.ndarray] = None,
                      img_gray: Optional[np.ndarray] = None, 
                      figsize: tuple = (24, 12)):
    
    w0,h0,w1,h1 = window
    
    if img is None and img_gray is None:
        raise Exception("Nothing to plot. Either img or img_gray should be not zero")
    elif (img is not None) and (img_gray is not None): 
        figure, axes = plt.subplots(ncols=2, nrows=2, figsize=figsize)
        a0 = axes[0,0]
        a2 = axes[0,1]
        a1 = axes[1,0]
        a3 = axes[1,1]
        _ = a0.imshow(img)
        _ = a1.imshow(img[w0:w1,h0:h1])
        _ = a0.set_title("image")
        _ = a1.set_title("image, zoom")
        _ = a2.imshow(img_gray, cmap="gray")
        _ = a3.imshow(img_gray[w0:w1,h0:h1], cmap="gray")
        _ = a2.set_title("image_GRAY")
        _ = a3.set_title("image_GRAY, zoom")
    elif (img is not None):
        figure, axes = plt.subplots(ncols=2, figsize=figsize)
        a0 = axes[0]
        a1 = axes[1]
        _ = a0.imshow(img)
        _ = a1.imshow(img[w0:w1,h0:h1])
        _ = a0.set_title("image")
        _ = a1.set_title("image, zoom")
    else:
        figure, axes = plt.subplots(ncols=2, figsize=figsize)
        a0 = axes[0]
        a1 = axes[1]
        _ = a0.imshow(img_gray, cmap='gray')
        _ = a1.imshow(img_gray[w0:w1,h0:h1], cmap='gray')
        _ = a0.set_title("image_GRAY")
        _ = a1.set_title("image_GRAY, zoom")

        
def plot_img_and_nuclei(img_gray, NUCLEI_mask, window, figsize=(24, 12)):

    w0,h0,w1,h1 = window 
    figure, axes = plt.subplots(ncols=3, nrows=2, figsize=figsize)
    gray_img = axes[0,0]
    instances = axes[0,1]
    overlay = axes[0,2]

    gray_img_zoom = axes[1,0]
    instances_zoom = axes[1,1]
    overlay_zoom = axes[1,2]

    gray_img.imshow(img_gray, cmap='gray')
    instances.imshow(NUCLEI_mask, cmap='gray')
    overlay.imshow(img_gray)
    overlay.imshow(NUCLEI_mask, alpha=0.5, cmap='gray')

    gray_img_zoom.imshow(img_gray[w0:w1,h0:h1], cmap='gray')
    instances_zoom.imshow(NUCLEI_mask[w0:w1,h0:h1], cmap='gray')
    overlay_zoom.imshow(img_gray[w0:w1,h0:h1])
    overlay_zoom.imshow(NUCLEI_mask[w0:w1,h0:h1], alpha=0.5)#, cmap='gray')

    _ = gray_img.set_title("RAW GRAYSCALE image")
    _ = gray_img_zoom.set_title("RAW GRAYSCALE image, zoom")
    _ = instances.set_title("NUCLEI only")
    _ = instances_zoom.set_title("NUCLEI only, zoom")
    _ = overlay.set_title("overlay image and nuclei")
    _ = overlay_zoom.set_title("overlay image and nuclei, zoom")
    
    
def show_video(frames: np.ndarray,
               ref_image: np.ndarray,
               mse: np.ndarray,
               figsize: tuple = (8, 4),
               interval: int = 50):
    """
    :param video: an ndarray with shape (n_frames, height, width, 3)
    """
    fig, axs = plt.subplots(ncols=2, figsize=figsize)
    ax_image = axs[0]
    ax_video = axs[1]
    
    # image
    ax_image.axis('off')
    ax_image.imshow(ref_image)
    ax_image.set_title("ref image")
    
    ax_video.axis('off')
    ax_video.imshow(frames[0, ...])
    ax_video.set_title("title")
        
    plt.tight_layout()
    plt.close()
    
    def init():
        ax_video.imshow(frames[0, ...])
        ax_video.set_title("title")
        
    def animate(i):
        title = 'corrupted prediction, frame={0:3d}, sigma={1:.3f}'.format(i,mse[i])
        ax_video.imshow(frames[i, ...])
        ax_video.set_title(title)
        
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=frames.shape[0],
        interval=interval)
    
    return HTML(anim.to_html5_video())


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
