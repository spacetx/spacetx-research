import torch
import skimage.filters
import skimage.exposure
import skimage.transform
import numpy as np
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None


def foreground_mask(image_):
    image_thresh = skimage.filters.threshold_otsu(image_)
    fg_mask = (image_ > image_thresh).float()  # True = bright, False = dark
    return fg_mask 


def img_pre_processing(pilfile, reduction_factor=1, remove_background=True):
    """ Resize and rescale intensities in (0,1) """
    
    # Open and resize using bilinear interpolation
    w_raw, h_raw = pilfile.size
    w_new = int(w_raw/reduction_factor)
    h_new = int(h_raw/reduction_factor)
    pilresize = pilfile.convert("F").resize((w_new, h_new), resample=PIL.Image.BILINEAR)
        
    # compute OTSU threshold
    img_np = np.array(pilresize)
    # mask = (img_np != 0)
    image_thresh = skimage.filters.threshold_otsu(img_np)
        
    # Rescale foreground intensity in (0,1)
    if remove_background:
        ql, qr = np.percentile(img_np[img_np > image_thresh].flatten(), q=(0, 100))  # note that the statistics are compute on the foreground only
        img_tmp = skimage.exposure.rescale_intensity(img_np, in_range=(ql, qr), out_range=(0.0, 1.0))
    else:
        img_tmp = skimage.exposure.rescale_intensity(img_np, in_range="image", out_range=(0.0, 1.0))
                
    return img_tmp


def sum_in_windows(img, window_size: int=80):
    """ returns the sum inside a square of size=window_size with center located at (i,j) """
    w, h = img.shape[-2:]
    img_pad = np.pad(img, pad_width=window_size//2, mode='constant', constant_values=0)
    assert (img == img_pad[window_size//2:window_size//2+w,window_size//2:window_size//2+h]).all()

    cum = np.cumsum(np.cumsum(img_pad, axis=0), axis=1)
    
    # roll
    px = np.roll(cum, +window_size//2, axis=0)
    mx = np.roll(cum, -window_size//2, axis=0)
    px_py = np.roll(px, +window_size//2, axis=1)
    px_my = np.roll(px, -window_size//2, axis=1)
    mx_py = np.roll(mx, +window_size//2, axis=1)
    mx_my = np.roll(mx, -window_size//2, axis=1)
    
    # compute sum in square
    tmp = (px_py - px_my - mx_py + mx_my)
    return tmp[window_size//2: window_size//2+w, window_size//2: window_size//2+h]


def estimate_noise(img: torch.Tensor, radius_nn: int=2):
    # Compute average first
    avg = torch.zeros_like(img)
    n = 0
    for dx in range(-radius_nn,radius_nn+1):
        y_tmp = torch.roll(img, dx, dims=-2)
        for dy in range(-radius_nn,radius_nn+1):
            y = torch.roll(y_tmp, dy, dims=-1)
            avg += y
            n +=1
    avg = avg.float()/n
    # print("avg ->",torch.min(avg), torch.max(avg))
                    
    # Compute variance later
    var = torch.zeros_like(avg)
    n = 0
    for dx in range(-radius_nn,radius_nn+1):
        y_tmp = torch.roll(img, dx, dims=-2)
        for dy in range(-radius_nn,radius_nn+1):
            y = torch.roll(y_tmp, dy, dims=-1)
            var += (y-avg)**2
            n +=1
    var = var / (n-1)
    # print("var ->",torch.min(var), torch.max(var))
                        
    # remove boundaries
    avg = avg[...,radius_nn+1:-radius_nn-1] 
    var = var[...,radius_nn+1:-radius_nn-1] 
                    
    y = torch.sqrt(var[avg>0]).view(-1)
    x = avg[avg>0].view(-1)
    return x,y


def index_for_binning(input, bins=100, min=0, max=0):
    if (min == 0) and (max == 0):
        min = torch.min(input)
        max = torch.max(input)
    index = (bins * (input - min).float()/(max-min)).int()
    return index


def compute_average_in_each_bin(x,y,bins=100, x_min=0, x_max=0):
    assert x.shape == y.shape
    index = index_for_binning(x, bins=bins, min=x_min, max=x_max)
    x_stratified = torch.zeros(bins, dtype=x.dtype, device=x.device)
    y_stratified = torch.zeros(bins, dtype=x.dtype, device=x.device)
    for i in range(0, bins):
        x_stratified[i] = x[index == i].mean()
        y_stratified[i] = y[index == i].mean()
    return x_stratified, y_stratified


def normalize_tensor(image, scale_each_image=False, scale_each_channel=False, in_place=False):
    """ Normalize a batch of images to the range 0,1 """
            
    assert len(image.shape) == 4  # batch, ch, w,h
    
    if (not scale_each_image) and (not scale_each_channel):
        max_tmp = torch.max(image)
        min_tmp = torch.min(image)
    elif scale_each_image and (not scale_each_channel):
        max_tmp = torch.max(image, dim=-4, keepdim=True)[0]
        min_tmp = torch.min(image, dim=-4, keepdim=True)[0]
    elif (not scale_each_image) and scale_each_channel:
        max_tmp = torch.max(image, dim=-3, keepdim=True)[0]
        min_tmp = torch.min(image, dim=-3, keepdim=True)[0]
    elif scale_each_image and scale_each_channel:
        max_tmp = torch.max(image, dim=-4, keepdim=True)[0].max(dim=-3, keepdim=True)[0]
        min_tmp = torch.min(image, dim=-4, keepdim=True)[0].min(dim=-3, keepdim=True)[0]

    if in_place:
        data = image.clone().clamp_(min=min_tmp, max=max_tmp)  # avoid modifying tensor in-place
    else:
        data = image.clamp_(min=min_tmp, max=max_tmp)
    return data.add_(-min_tmp).div_(max_tmp - min_tmp + 1e-5)
