import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import skimage.exposure
import numpy as np
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None
from typing import Optional

from .namedtuple import ImageBbox


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

