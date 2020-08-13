import torch
import pickle
import json
import numpy as np
from torchvision import utils
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset
import pyro
from collections import namedtuple
from torch.distributions.utils import broadcast_all


def show_batch(images, n_col=4, n_padding=10, title=None, pad_value=1):
    """Visualize a torch tensor of shape: (batch x ch x width x height) """
    assert len(images.shape) == 4  # batch, ch, width, height
    if images.device != "cpu":
        images = images.cpu()
    grid = utils.make_grid(images, n_col, n_padding, normalize=True, range=(0.0, 1.0),
                           scale_each=False, pad_value=pad_value)

    fig = plt.figure()
    plt.imshow(grid.detach().numpy().transpose((1, 2, 0)))
    if isinstance(title, str):
        plt.title(title)
    plt.close(fig)
    return fig


def flatten_list(my_list):
    flat_list = []
    for sublist in my_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list
        
        
def save_obj(obj, root_dir, name):
    with open(root_dir + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(root_dir, name):
    with open(root_dir + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def load_json_as_dict(root_dir, name):
    with open(root_dir + name + '.json', 'rb') as f:
        return json.load(f)


def save_dict_as_json(my_dict, root_dir, name):
    with open(root_dir + name + '.json', 'w') as f:
        return json.dump(my_dict, f)


def reset_parameters(parent_module, verbose):
    for m in parent_module.modules():
        try:
            m.reset_parameters()
            if verbose:
                print("reset -> ", m)
        except AttributeError:
            pass


def inverse_sigmoid(p):
    """ p = sigmoid(x) = 1.0 / (1.0+exp(-x))
        a = 2*p-1 = (1-exp(-x))/(1+exp(-x))
        x = log(1+a) - log(1-a)
        If p=0,1 then a = -1,+1 and x=-Inf, +Inf
    """    
    a = 2*p-1 #problem when a = +/- 1
    x = torch.log1p(a)-torch.log1p(-a)
    return x


def sample_normal(mu, std, noisy_sampling):
    new_mu, new_std = broadcast_all(mu, std)
    if noisy_sampling:
        return new_mu + new_std * torch.randn_like(new_mu)
    else:
        return new_mu


def kl_normal0_normal1(mu0, mu1, std0, std1):
    tmp = (std0 + std1) * (std0 - std1) + (mu0 - mu1).pow(2)
    return tmp / (2 * std1 * std1) - torch.log(std0 / std1)


def kl_bernoulli0_bernoulli1(p0, p1):
    return p0 * (torch.log(p0) - torch.log(p1)) + (1-p0) * (torch.log1p(-p0) - torch.log1p(-p1))


def compute_ranking(x):
    """ Given a vector of shape: n, batch_size
        For each batch dimension it ranks the n elements"""
    assert len(x.shape) == 2
    n, batch_size = x.shape
    _, order = torch.sort(x, dim=-2, descending=False)

    # this is the fast way which uses indexing on the left
    rank = torch.zeros_like(order)
    batch_index = torch.arange(batch_size, dtype=order.dtype, device=order.device).view(1, -1).expand(n, batch_size)
    rank[order, batch_index] = torch.arange(n, dtype=order.dtype, device=order.device).view(-1, 1).expand(n, batch_size)
    return rank


def compute_average_intensity_in_box(imgs=None, bounding_box=None):
    """ Input batch of images: batch_size x ch x w x h
        z_where collections of [bx,by,bw,bh]
        bx.shape = batch x n_box
        similarly for by,bw,bh

        Output:
        av_intensity = n_box x batch_size
    """
    # cumulative sum in width and height, standard sum in channels
    cum = torch.sum(torch.cumsum(torch.cumsum(imgs, dim=-1), dim=-2), dim=-3)
    assert len(cum.shape) == 3
    batch_size, w, h = cum.shape

    # compute the x1,y1,x3,y3
    x1 = torch.clamp((bounding_box.bx - 0.5 * bounding_box.bw).long(), min=0, max=w - 1)
    x3 = torch.clamp((bounding_box.bx + 0.5 * bounding_box.bw).long(), min=0, max=w - 1)
    y1 = torch.clamp((bounding_box.by - 0.5 * bounding_box.bh).long(), min=0, max=h - 1)
    y3 = torch.clamp((bounding_box.by + 0.5 * bounding_box.bh).long(), min=0, max=h - 1)
    assert x1.shape == x3.shape == y1.shape == y3.shape

    # compute the area
    # Note that this way penalizes boxes that go out-of-bound
    # This is in contrast to area = (x3-x1)*(y3-y1) which does NOT penalize boxes out of bound
    area = bounding_box.bw * bounding_box.bh
    assert area.shape == x1.shape == x3.shape == y1.shape == y3.shape
    n_boxes, batch_size = area.shape

    # compute the total intensity in each box
    batch_index = torch.arange(start=0, end=batch_size, step=1, device=x1.device,
                               dtype=x1.dtype).view(1, -1).expand(n_boxes, -1)
    assert batch_index.shape == x1.shape

    tot_intensity = cum[batch_index, x3, y3] \
                    + cum[batch_index, x1, y1] \
                    - cum[batch_index, x1, y3] \
                    - cum[batch_index, x3, y1]

    # return the average intensity
    assert tot_intensity.shape == x1.shape
    return tot_intensity / area


def linear_interpolation(t, values: tuple, times: tuple):
    """ Makes an interpolation between (t_in,v_in) and (t_fin,v_fin)
        For time t>t_fin and t<t_in the value of v is clamped to either v_in or v_fin
        Usage:
        epoch = np.arange(0,100,1)
        v = linear_interpolation(epoch, values=[0.0,0.5], times=[20,40])
        plt.plot(epoch,v)
    """
    v_in, v_fin = values  # initial and final values
    t_in, t_fin = times   # initial and final times

    if t_fin >= t_in:
        den = max(t_fin-t_in, 1E-8)
        m = (v_fin-v_in)/den
        v = v_in + m*(t-t_in)
    else:
        sys.exit("t_fin should be greater than t_in")

    v_min = min(v_in, v_fin)
    v_max = max(v_in, v_fin)
    return np.clip(v, v_min, v_max)


def test_model(model, guide, loss):
    pyro.clear_param_store()
    loss.loss(model, guide)


def train_one_epoch(svi, dataset, batch_size=64, verbose=False, weight_clipper=None):
    epoch_loss = 0.
    n_term = 0
    batch_iterator = dataset.generate_batch_iterator(batch_size)

    for i, indices in enumerate(batch_iterator):  # get the indeces

        # get the data from the indices. Note that everything is already loaded into memory
        loss = svi.step(imgs=dataset[indices][0])
        if verbose:
            print("i = %3d train_loss=%.5f" % (i, loss))

        # apply the weight clipper
        if weight_clipper is not None:
            svi.model.__self__.apply(weight_clipper)

        epoch_loss += loss
        n_term += len(indices)
    return epoch_loss / n_term

        

def evaluate_one_epoch(svi, dataset, batch_size=64, verbose=False, weight_clipper=None):
    epoch_loss = 0.
    n_term = 0
    batch_iterator = dataset.generate_batch_iterator(batch_size)

    for i, indices in enumerate(batch_iterator):  # get the indices

        # get the data from the indices. Note that everything is already loaded into memory
        loss = svi.evaluate_loss(imgs=dataset[indices][0])
        if verbose:
            print("i = %3d test_loss=%.5f" % (i, loss))

        # apply the weight clipper
        if weight_clipper is not None:
            svi.model.__self__.apply(weight_clipper)

        epoch_loss += loss
        n_term += len(indices)
    return epoch_loss / n_term


class DatasetInMemory(Dataset):
    """ Typical usage:

        synthetic_data_test  = dataset_in_memory(root_dir,"synthetic_data_DISK_test_v1",use_cuda=False)
        for epoch in range(5):
            print("EPOCH")
            batch_iterator = synthetic_data_test.generate_batch_iterator(batch_size)
            for i, x in enumerate(batch_iterator):
                print(x)
                blablabla
                ......
    """

    def __init__(self, root_dir, name, use_cuda=False):
        self.histo = None
        self.otsu = None
        self.cumulative = None
        self.use_cuda = use_cuda

        with torch.no_grad():
            try:
                # this is when I have both imgs and labels
                imgs, labels = load_obj(root_dir, name)
                self.n = imgs.shape[0]
            except ValueError:
                # this is where the labels are missing, Create fake labels = -1
                imgs = load_obj(root_dir, name)
                self.n = imgs.shape[0]
                labels = -1 * imgs.new_ones(self.n)

            if self.use_cuda:
                self.data = imgs.cuda()
                self.labels = labels.cuda()
            else:
                self.data = imgs.cpu().detach()
                self.labels = labels.cpu().detach()

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.data[index, ...], self.labels[index]

    def load(self, batch_size=8, indices=None):
        if indices is None:
            indices = torch.randint(low=0, high=self.n, size=(batch_size,)).long()
        return self.__getitem__(indices[:batch_size])

    def generate_batch_iterator(self, batch_size):
        indices = torch.randperm(self.n).numpy()
        remainder = len(indices) % batch_size
        n_max = len(indices) - remainder # Note the trick so that all the minibatches have the same size
        n_max = len(indices)
        assert (n_max >= batch_size), "batch_size is too big for this dataset of size."
        batch_iterator = (indices[pos:pos + batch_size] for pos in range(0, n_max, batch_size))
        return batch_iterator

    def check(self):
        print("Dataset lenght:", self.__len__())

        imgs, labels = self.load(batch_size=8)
        title = "# labels =" + str(labels.cpu().numpy().tolist())
        tmp_img = show_batch(imgs[:8], n_col=4, n_padding=4, title=title)

        print("imgs.shape", imgs.shape)
        print("type(imgs)", type(imgs))
        print("imgs.device", imgs.device)
        print("torch.max(imgs)", torch.max(imgs))
        print("torch.min(imgs)", torch.min(imgs))
        return tmp_img

    def compute_brightness_distribution_window(self, window_size=28, stride=6):
        """ For each image analyze the distribution of brightness of window_size=size"""
        if self.cumulative is None:
            self.cumulative = torch.sum(torch.cumsum(torch.cumsum(self.data, dim=-1), dim=-2), dim=-3)

        assert len(self.cumulative.shape) == 3
        batch_size, w, h = self.cumulative.shape

        l_tot = len(range(0, w - window_size, stride)) * len(range(0, h - window_size, stride))
        result = torch.zeros(batch_size, l_tot)

        k = 0
        for ix in range(0, w - window_size, stride):
            for iy in range(0, h - window_size, stride):
                tmp = self.cumulative[:, ix + window_size, iy + window_size] + \
                      self.cumulative[:, ix, iy] - \
                      self.cumulative[:, ix + window_size, iy] - \
                      self.cumulative[:, ix, iy + window_size]
                result[:, k] = tmp
                k = k + 1
        return result / (window_size * window_size)

    def compute_otsu_parameters(self, left=-0.1, right=1.1, nbins=50):
        """ For each channel, use the Otsu method to compute:
            1. histogram
            2. inter class variance
            3. threshold value
        """

        # preparation
        channels = self.data[0].shape[-3]
        delta = (right - left) / nbins
        x = torch.arange(start=left + 0.5 * delta,
                         end=right + 0.5 * delta,
                         step=delta,
                         dtype=self.data.dtype,
                         device=self.data.device).float()

        # Prepare variable to save
        bg_fraction = torch.zeros(channels, dtype=self.data.dtype, device=self.data.device)
        threshold = torch.zeros(channels, dtype=self.data.dtype, device=self.data.device)
        fg_mu = torch.zeros(channels, dtype=self.data.dtype, device=self.data.device)
        bg_mu = torch.zeros(channels, dtype=self.data.dtype, device=self.data.device)
        fg_std = torch.zeros(channels, dtype=self.data.dtype, device=self.data.device)
        bg_std = torch.zeros(channels, dtype=self.data.dtype, device=self.data.device)
        histogram_x = torch.zeros(channels, nbins, dtype=self.data.dtype, device=self.data.device)
        histogram_y = torch.zeros(channels, nbins, dtype=self.data.dtype, device=self.data.device)

        for c in range(channels):
            # Compute histogram and normalize
            hist_count = torch.histc(self.data[:, c].view(-1), bins=nbins, min=left, max=right)
            p = hist_count / (delta * torch.sum(hist_count))  # normalize histogram
            histogram_x[c] = x  # normalized histogram
            histogram_y[c] = p  # normalized histogram

            # Otsu
            w0 = torch.cumsum(p, dim=-1)
            w1 = w0[-1] - w0
            m0 = torch.cumsum(p * x, dim=-1)
            m1 = m0[-1] - m0
            m0 = m0 / w0  # this might be division by zero
            m1 = m1 / w1  # this might be division by zero
            m0[torch.isnan(m0)] = 0  # take care, in case I divided by zero
            m1[torch.isnan(m1)] = 0  # take care, in case I divided by zero

            # Maximize the Inter-class variance
            inter_var = w0 * w1 * (m0 - m1).pow(2)
            n_th = torch.argmax(inter_var)
            threshold[c] = x[n_th] + 0.5 * delta
            bg_mu[c] = m0[n_th]
            fg_mu[c] = m1[n_th]
            bg_fraction[c] = w0[n_th] / w0[-1]

            # Compute the two separate sigmas
            tmp0 = p * (x - m0[n_th]).pow(2)
            tmp1 = p * (x - m1[n_th]).pow(2)
            bg_std[c] = torch.sqrt(torch.sum(tmp0[:n_th + 1]) / w0[n_th])  # +1 is to recover the uniform dist results
            fg_std[c] = torch.sqrt(torch.sum(tmp1[n_th + 1:]) / w1[n_th])  # +1 is to recover the uniform dist results

        Histo = namedtuple("histogram", "x y")
        Otsu = namedtuple('Otsu', 'threshold bg_mu bg_std fg_mu fg_std bg_fraction')
        self.otsu = Otsu(threshold=threshold,
                         bg_mu=bg_mu,
                         bg_std=bg_std,
                         fg_mu=fg_mu,
                         fg_std=fg_std,
                         bg_fraction=bg_fraction)

        self.histo = Histo(x=histogram_x, y=histogram_y)

        return self.histo, self.otsu


### def my_max(a, dim, keepdim=False):
###     """ SImilar to torch.max but works with dim = Tuple"""
###
###     if isinstance(dim, int):
###
###         return torch.max(a, dim, keepdim=keepdim)[0]
###
###     elif isinstance(dim, tuple):
###
###         list_shape = list(a.shape)
###         K = len(list_shape)
###         pos_dim = []
###         tmp = a
###         for n in dim:
###             tmp = torch.max(tmp, dim=n, keepdim=True)[0]
###             j = n if n >= 0 else K + n
###             pos_dim.append(j)
###
###         if keepdim:
###             return tmp
###         else:
###             for n in sorted(pos_dim, reverse=True):
###                 del list_shape[n]
###             return tmp.view(list_shape)
###
###
### def sample_uniform(z_av=None, z_d=None, noisy_sampling=True):
###     """ Sample from uniform distribution given av and delta"""
###     if noisy_sampling:
###         return z_av + (torch.rand_like(z_av)-0.5) * z_d
###     else:
###         return z_av
###
### def kl_uniform0_uniform1(d0=None, d1=None):
###     """ KL divergence between Uniform_0 and Uniform_1
###         KL(U0 || U1) = int dz U0(z) log[ U0(z)/U1(1) ]
###         Note that KL > 0 because the support of the posterior,
###         i.e. U0, is a subset of the support of the prior, i.e. U1
###     """
###     return torch.log(d1) - torch.log(d0)
###
### LN_1_M_EXP_THRESHOLD = -np.log(2.)
### def get_log_prob_compl(log_prob):
###     """ Compute log(1-p) from log(p) """
###     return torch.where(
###         log_prob >= LN_1_M_EXP_THRESHOLD,
###         torch.log(-torch.expm1(log_prob)),
###         torch.log1p(-torch.exp(log_prob)))
###
###
### def Log_Add_Exp(log_pa,log_pb,log_w,log_1mw,verbose=False):
###     """ Compute log(w*pa+(1-w)*pb) in a numerically stable way
###
###         Test usage:
###
###         log_pa  = torch.tensor([-20.0])
###         log_pb  = torch.tensor([-50.0])
###         log_w   = torch.arange(-100,1,5.0).float()
###         log_1mw = get_log_prob_compl(log_w)
###
###         logp_v1 = Log_Add_Exp(log_pa,log_pb,log_w,log_1mw)
###         logp_v2 = Log_Add_Exp(log_pb,log_pa,log_1mw,log_w)
###
###         plt.plot(log_w.numpy(),logp_v1.numpy(),'.')
###         plt.plot(log_w.numpy(),logp_v2.numpy(),'-')
###     """
###     assert log_pa.shape == log_pb.shape
###     log_pa_and_log_pb = torch.stack((log_pa,log_pb),dim=len(log_pa.shape))
###
###     assert log_w.shape == log_1mw.shape
###     log_w_and_log_1mw = torch.stack((log_w,log_1mw),dim=len(log_w.shape))
###
###     if(verbose):
###         print("log_w_and_log_1mw.shape",log_w_and_log_1mw.shape)
###         print("log_pa_and_log_pb.shape",log_pa_and_log_pb.shape)
###
###     return torch.logsumexp(log_pa_and_log_pb+log_w_and_log_1mw,dim=-1)
###
###
###def Normal(x,mu,sigma):
###    """ Return the value of N(mu,sigma) at the locations x 
###        
###        Typical usage:
###        x = np.arange(0.0,1.0,0.01)
###        y1 = Normal(x,0.1,0.1)
###        plt.plot(x,y1,'-')
###    """
###    tmp=-0.5*((x-mu)/sigma)**2
###    c = 1.0/(np.sqrt(2*np.pi)*sigma)
###    return np.exp(tmp)*c
###
###def Cauchy(x,loc,scale):
###    """ Return the value of Cauchy(mu,sigma) at the locations x 
###        
###        Typical usage:
###        x = np.arange(0.0,1.0,0.01)
###        y1 = Cauchy(x,0.1,0.1)
###        plt.plot(x,y1,'-')
###    """
###    tmp  = ((x-loc)/scale)**2
###    tmp2 = np.pi*scale*(1.0+tmp)
###    return 1.0/tmp2
###
###def Exp_shift_scale(x,loc,scale):
###    """ Return the value of (0.5/scale)*Exp(-||x-mu||/scale) at the locations x 
###        
###        Typical usage:
###        x = np.arange(0.0,1.0,0.01)
###        y1 = Exp_shift_scale(x,0.1,0.1)
###        plt.plot(x,y1,'-')
###    """
###    tmp = np.abs(x-loc)/scale
###    return 0.5*np.exp(-tmp)/scale
###
###
###    
###
###def corners_from_bxby_bwbh_cos_sin(bxby,bwbh,cos_sin):
###
###    cos_bw_half = 0.5*cos_sin[...,0]*bwbh[...,0]
###    sin_bw_half = 0.5*cos_sin[...,1]*bwbh[...,0]
###    cos_bh_half = 0.5*cos_sin[...,0]*bwbh[...,1]
###    sin_bh_half = 0.5*cos_sin[...,1]*bwbh[...,1]
###
###    x1 = ( bxby[...,0] -cos_bw_half + sin_bh_half ).unsqueeze(2)
###    x2 = ( bxby[...,0] +cos_bw_half + sin_bh_half ).unsqueeze(2)
###    x3 = ( bxby[...,0] +cos_bw_half - sin_bh_half ).unsqueeze(2)
###    x4 = ( bxby[...,0] -cos_bw_half - sin_bh_half ).unsqueeze(2)
###
###    y1 = ( bxby[...,1] -sin_bw_half - cos_bh_half ).unsqueeze(2)
###    y2 = ( bxby[...,1] +sin_bw_half - cos_bh_half ).unsqueeze(2)
###    y3 = ( bxby[...,1] +sin_bw_half + cos_bh_half ).unsqueeze(2)
###    y4 = ( bxby[...,1] -sin_bw_half + cos_bh_half ).unsqueeze(2)
###
###    return torch.cat((x1,y1,x2,y2,x3,y3,x4,y4),dim=2)
###
###
###def logit_to_p(x):
###    """ This is low accuracy.
###        Use logit_to_dp for higher accuracy
###        Note that p=1.0-dp ~ 1.0 when dp is small due to rounding off error 
###    """
###    dp = logit_to_dp(x)
###    return np.where(dp>0,dp,1.0-dp)
###
###def p_to_logit(p):
###    """ This is low accuracy.
###        Use dp_to_logit for higher accuracy 
###        Note that p can NOT be arbitrarely close to 1.0 due to rounding error
###    """
###    dp = np.where(p<0.5,p,p-1.0)
###    return dp_to_logit(dp)
###    
###def logit_to_dp(x):
###    """ This is high precision
###        
###        Analytical expression: p = exp(x)/(1.0+exp(x)) 
###        becomes unstable for very large x. 
###                
###        dp = min(p,1-p) = exp(-|x|)/(1.0+exp(-|x|))
###        Is always stable
###        
###        I return +dp if x<0, i.e. p=dp
###        I return -dp if x>0, i.e. p=1-dp
###    """    
###    tmp = np.exp(-np.abs(x))
###    dp  = tmp/(1.0+tmp)
###    # I can not use torch.sign since torch.sign(0)=0
###    return np.where(x<0,dp,-dp) 
###
###def dp_to_logit(dp):
###    """ This is high precision
###        logit = torch.log(p/(1-p)) = torch.log(p)-torch.log(1-p) = torch.log(|dp|)-torch.log1p(-|dp|) is stable  
###    """
###    abs_dp = np.abs(dp)
###    r1 = np.log(abs_dp)-np.log1p(-abs_dp)
###    print(dp,abs_dp,r1)
###    return r1*np.sign(dp)
###
###
###def Log_Add_Exp(log_pa,log_pb,log_w,verbose=False):
###    """ Compute log(w*pa+(1-w)*pb) in a numerically stable way 
###        Note that w is in (0,1) and logits_w = log (w/(1-w)) is in (-Infinity,Infinity)
###        The inversion is: 
###        w = exp(logits_w)/(1+exp(logits_w))
###        which is remarkably similar to the softmax 
###        
###                
###        Test usage:
###        log_pa = torch.tensor([-20.0])
###        log_pb = torch.tensor([-50.0])
###        logits = torch.arange(-100,100,5.0).float()
###
###        logp_v1 = Log_Add_Exp(log_pa,log_pb,logits)
###        logp_v2 = Log_Add_Exp(log_pb,log_pa,-logits)
###
###        plt.plot(logits.numpy(),logp_v1.numpy(),'.')
###        plt.plot(logits.numpy(),logp_v2.numpy(),'-')
###    """
###    assert log_pa.shape == log_pb.shape 
###    log_pa_and_log_pb = torch.stack((log_pa,log_pb),dim=len(log_pa.shape))
###    
###    zeros = torch.zeros_like(logits_weights)
###    logits_weight_and_zero = torch.stack((logits_weights,zeros),dim=len(logits_weights.shape))
###    log_w_and_log_1mw = F.log_softmax(logits_weight_and_zero,dim=-1)
###    
###    if(verbose):
###        print("log_w_and_log_1mw.shape",log_w_and_log_1mw.shape)
###        print("log_pa_and_log_pb.shape",log_pa_and_log_pb.shape)
###    
###    return torch.logsumexp(log_pa_and_log_pb+log_w_and_log_1mw,dim=-1)
