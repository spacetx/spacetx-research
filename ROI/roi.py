# sub-parts of the Unet

import torch
import collections

import torch
import collections
from collections import deque

def convert_to_box_list(x):
    """ takes x of shape: (batch x ch x width x height) 
        and returns a list: batch x n_list x ch
        where n_list = width x height
    """
    batch_size, ch, width, height = x.shape
    return x.permute(0,2,3,1).view(batch_size,width*height,ch)

def concatenate_deque_of_tuples(deque_of_tuples,concat_dim):
    """ Concatenating deque of the same tuple """
    L = len(deque_of_tuples)
    
    if( L == 1):
        return deque_of_tuples[0]
    else:
        
        # check that merging is possible
        for i in range(L-1):
            assert deque_of_tuples[i]._fields == deque_of_tuples[i+1]._fields
        
        # extract title and fields
        title = deque_of_tuples[0].__class__.__name__
        names = deque_of_tuples[0]._fields
        
        # concatenate the torch tensor
        output = dict()
        for name in names:
            output[name] = torch.cat([ getattr(deque_of_tuples[i],name) for i in range(L)],dim=concat_dim)
        
        # create new tuple and return
        return collections.namedtuple(title, names)._make([output[name] for name in names])


class predict_Zwhere(torch.nn.Module):
    """ Input  shape: batch x ch x width x height
        Output: namedtuple with all dimless stuff: prob,bx,by,bw,bh
        Each one of them has shape: batch x n_boxes x 1
    """
    def __init__(self, channel_in,params: dict):
        super().__init__()
        self.ch_in = channel_in
        self.max_size= params['PRIOR.max_object_size']
        self.min_size= params['PRIOR.min_object_size']

        
        self.comp_p  = torch.nn.Conv2d(channel_in,1,kernel_size=1, stride=1, padding=0, bias=True)
        self.comp_tx = torch.nn.Conv2d(channel_in,1,kernel_size=1, stride=1, padding=0, bias=True)
        self.comp_ty = torch.nn.Conv2d(channel_in,1,kernel_size=1, stride=1, padding=0, bias=True)
        self.comp_tw = torch.nn.Conv2d(channel_in,1,kernel_size=1, stride=1, padding=0, bias=True)
        self.comp_th = torch.nn.Conv2d(channel_in,1,kernel_size=1, stride=1, padding=0, bias=True)

        # Here I am initializing the bias with large value so that p also has large value
        # This in turns helps the model not to get stuck in the empty configuration which is a local minimum
        self.comp_p.bias.data += 1.0


    def forward(self,features,width_raw_image,height_raw_image):

        batch, ch, n_width, n_height = features.shape

        with torch.no_grad():
            ix = torch.arange(0,n_width,  dtype=features.dtype, device=features.device).view(1,1,-1,1) #between 0 and n_width-1
            iy = torch.arange(0,n_height, dtype=features.dtype, device=features.device).view(1,1,1,-1) #between 0 and n_height-1

        # probability
        p = torch.sigmoid(self.comp_p(features))

        # center of bounding bofeatures.from dimfull to dimless and reshaping
        bx = torch.sigmoid(self.comp_tx(features)) + ix #-- in (0,n_width)
        by = torch.sigmoid(self.comp_ty(features)) + iy #-- in (0,n_height)
        bx_dimfull = width_raw_image*bx/n_width   # in (0,width_raw_image)
        by_dimfull = height_raw_image*by/n_height # in (0,height_raw_image)  

        # size of the bounding box
        bw_dimless = torch.sigmoid(self.comp_tw(features)) # between 0 and 1
        bh_dimless = torch.sigmoid(self.comp_th(features)) # between 0 and 1
        bw_dimfull = self.min_size + (self.max_size-self.min_size)*bw_dimless # in (min_size,mafeatures.size)
        bh_dimfull = self.min_size + (self.max_size-self.min_size)*bh_dimless # in (min_size,mafeatures.size)

        return collections.namedtuple('z_where_dimfull', 'prob bx_dimfull by_dimfull bw_dimfull bh_dimfull')._make(
            [convert_to_box_list(p),
             convert_to_box_list(bx_dimfull),
             convert_to_box_list(by_dimfull),
             convert_to_box_list(bw_dimfull),
             convert_to_box_list(bh_dimfull)])

class Roi(torch.nn.Module):
    """ Input: deque with all the feature maps.
        Each feature map has different ch,width,height  
        Of course the batch dimension is the same for all feature maps
        OUTPUT: Each one of them has shape: batch x n_boxes x 1
    """ 
    def __init__(self, ch_feature_maps,params):
        super().__init__()
        self.predict_roi = torch.nn.ModuleList()
        for n in range(len(ch_feature_maps)):
            self.predict_roi.append(predict_Zwhere(ch_feature_maps[n],params))

    def forward(self,feature_maps,width_raw_image,height_raw_image):
                
        z_where_list = deque() 
        for n, module in enumerate(self.predict_roi):
            z_where_list.append(module(feature_maps[n],width_raw_image,height_raw_image))
            
        # concatenate a list of tuple along the channel dimension
        return concatenate_deque_of_tuples(z_where_list,concat_dim=1)
