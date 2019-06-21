#import os
import torch
import pickle
import numpy as np
from torchvision import utils
import matplotlib.pyplot as plt

import torchvision.transforms as tvt
from torch.utils.data.dataset import Dataset
import pyro


def linear_decay_p_factor(epoch,decay_lenght_scale):
    f = 1.0/decay_lenght_scale
    tmp = 0.5*(1.0-epoch*f)
    #make sure the correction factor is larger than zero
    return np.clip(tmp,np.zeros_like(tmp),tmp) 

def test_model(model, guide, loss):
    pyro.clear_param_store()
    loss.loss(model, guide)

def train_one_epoch(svi, dataset, epoch=None, batch_size=64, verbose=False):
    
    epoch_loss = 0.
    batch_iterator = dataset.generate_batch_iterator(batch_size)
    for i, indeces in enumerate(batch_iterator): #get the indeces

        # get the data from the indeces. Note that everything is already loaded into memory
        loss = svi.step(imgs=dataset[indeces][0], epoch=epoch)
        if(verbose):
            print("i= %3d train_loss=%.5f" %(i,loss))
        epoch_loss += loss

    return epoch_loss / ((i+1)*batch_size)

def evaluate_one_epoch(svi, dataset, epoch=None, batch_size=64, verbose=False):
    
    epoch_loss = 0.
    batch_iterator = dataset.generate_batch_iterator(batch_size)
    for i, indeces in enumerate(batch_iterator): #get the indeces

        # get the data from the indeces. Note that everything is already loaded into memory
        loss = svi.evaluate_loss(imgs=dataset[indeces][0],epoch=epoch)
        if(verbose):
            print("i= %3d test_loss=%.5f" %(i,loss))
        epoch_loss += loss

    return epoch_loss / ((i+1)*batch_size)

def show_batch(images,nrow=4,npadding=10,title=None):
    """Visualize a torch tensor of shape: (batch x ch x width x height) """
    batch, ch, width, height = images.shape
    if(images.device != "cpu"):
        images=images.cpu()
    grid = utils.make_grid(images,nrow, npadding, normalize=True, range=None, scale_each=True, pad_value=1)       
    fig = plt.imshow(grid.detach().numpy().transpose((1, 2, 0))) 
    if(isinstance(title, str)):
        plt.title(title)
    return fig


class dataset_in_memory(Dataset):
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
    def __init__(self,root_dir,name,use_cuda=False):
        self.use_cuda = use_cuda
        try:
            # this is when I have both imgs and labels
            imgs, labels = load_obj(root_dir,name)
            self.n = imgs.shape[0]

            if(self.use_cuda):
                self.data   = imgs.cuda().detach()
                self.labels = labels.cuda().detach()
            else:
                self.data   = imgs.cpu().detach()
                self.labels = labels.cpu().detach()

        except ValueError:
            # this is where the labels are missing, Create fake labels = -1
            if(self.use_cuda):
                self.data   = load_obj(root_dir,name).cuda().detach()
            else:
                self.data   = load_obj(root_dir,name).cpu().detach()
            self.n = self.data.shape[0]
            self.labels = -1*self.data.new_ones(self.n)

                
        
    def __len__(self):
        return self.n
    
    def __getitem__(self,index):
        return self.data[index,...],self.labels[index]
    
    def load(self,batch_size=8):
        indeces = torch.randint(low=0,high=self.n,size=(batch_size,)).long()
        return self.__getitem__(indeces)
    
    def generate_batch_iterator(self,batch_size):
        # Note the trick so that all the minibatches have the same size
        indeces = torch.randperm(self.n).numpy()
        remainder = len(indeces)%batch_size
        n_max = len(indeces)-remainder
        assert (n_max > batch_size), "batch_size is too big for this dataset of size."
        batch_iterator = (indeces[pos:pos + batch_size] for pos in range(0, n_max, batch_size))
        return batch_iterator
    
    def analyze_brightness_distribution(self,size=28,stride=6):
        """ Analyze the distribution of brtightness of window of size=size"""
        if not hasattr(self,'cumulative'):
            self.cumulative = torch.cumsum(torch.cumsum(self.data, dim=-1),dim=-2)[:,0,:,:]
        
        batch_size,w,h = self.cumulative.shape
        
        L = len(range(0,w-size,stride))*len(range(0,h-size,stride))
        result = torch.zeros(batch_size,L)
    
        k = 0
        for ix in range(0,w-size,stride):
            for iy in range(0,h-size,stride):
                tmp = self.cumulative[:,ix+size,iy+size] - self.cumulative[:,ix+size,iy]- \
                      self.cumulative[:,ix,iy+size] + self.cumulative[:,ix,iy]
                result[:,k]=tmp
                k=k+1
        return result/(size*size)
    

def check_datasets(dataset):
    print("Dataset lenght:",dataset.__len__())

    imgs, labels =dataset.load(8)
    title = "# labels ="+str(labels.cpu().numpy().tolist())          
    show_batch(imgs[:8],nrow=4,npadding=4,title=title)

    print("imgs.shape",imgs.shape)
    print("type(imgs)",type(imgs))
    print("imgs.device",imgs.device)
    print("torch.max(imgs)",torch.max(imgs))
    print("torch.min(imgs)",torch.min(imgs))

def save_obj(obj,root_dir,name):
    with open(root_dir + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(root_dir,name):
    with open(root_dir + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def reset_parameters(parent_module):
    for m in parent_module.modules():
        try:
            m.reset_parameters()
            print("reset -> ",m)
        except:
            pass
        

LN_1_M_EXP_THRESHOLD = -np.log(2.)
def get_log_prob_compl(log_prob):
    """ Compute log(1-p) from log(p) """
    return torch.where(
        log_prob >= LN_1_M_EXP_THRESHOLD,
        torch.log(-torch.expm1(log_prob)),
        torch.log1p(-torch.exp(log_prob)))


def Log_Add_Exp(log_pa,log_pb,log_w,log_1mw,verbose=False):
    """ Compute log(w*pa+(1-w)*pb) in a numerically stable way 
                
        Test usage:

        log_pa  = torch.tensor([-20.0])
        log_pb  = torch.tensor([-50.0])
        log_w   = torch.arange(-100,1,5.0).float()
        log_1mw = get_log_prob_compl(log_w)

        logp_v1 = Log_Add_Exp(log_pa,log_pb,log_w,log_1mw)
        logp_v2 = Log_Add_Exp(log_pb,log_pa,log_1mw,log_w)

        plt.plot(log_w.numpy(),logp_v1.numpy(),'.')
        plt.plot(log_w.numpy(),logp_v2.numpy(),'-')
    """
    assert log_pa.shape == log_pb.shape 
    log_pa_and_log_pb = torch.stack((log_pa,log_pb),dim=-1)
    
    assert log_w.shape == log_1mw.shape 
    log_w_and_log_1mw = torch.stack((log_w,log_1mw),dim=-1)
    
    if(verbose):
        print("log_w_and_log_1mw.shape",log_w_and_log_1mw.shape)
        print("log_pa_and_log_pb.shape",log_pa_and_log_pb.shape)
    
    return torch.logsumexp(log_pa_and_log_pb+log_w_and_log_1mw,dim=-1)


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
