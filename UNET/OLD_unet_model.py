import torch
from .unet_parts import *
import numpy as np
from collections import deque


### AFTER YOU FIND SWEEP OF HYPERPARAMETERS
### FROm https://github.com/milesial/Pytorch-UNet
######class UNet(nn.Module):
######    def __init__(self, n_channels, n_classes):
######        super(UNet, self).__init__()
######        self.inc = inconv(n_channels, 64)
######        self.down1 = down(64, 128)
######        self.down2 = down(128, 256)
######        self.down3 = down(256, 512)
######        self.down4 = down(512, 512)
######        self.up1 = up(1024, 256)
######        self.up2 = up(512, 128)
######        self.up3 = up(256, 64)
######        self.up4 = up(128, 64)
######        self.outc = outconv(64, n_classes)
######
######    def forward(self, x):
######        x1 = self.inc(x)
######        x2 = self.down1(x1)
######        x3 = self.down2(x2)
######        x4 = self.down3(x3)
######        x5 = self.down4(x4)
######        x = self.up1(x5, x4)
######        x = self.up2(x, x3)
######        x = self.up3(x, x2)
######        x = self.up4(x, x1)
######        x = self.outc(x)
######        return F.sigmoid(x)







class UNet(torch.nn.Module):
    """ Wrapper around partial_unet or cnn_net """
    def __init__(self, params: dict):
        super().__init__() 
        n_up_conv  = params['UNET.N_up_conv']
        if(n_up_conv <= 0 ):
            self.label = "CNN_NET"
            self.net = cnn_net(params)
        elif(n_up_conv >= 1):
            self.label = "partial_UNET"
            self.net = partial_unet(params)
    
    def forward(self,x,verbose=False):
        return self.net.forward(x,verbose) 
    
    def show_grid(self,ref_image):
       
        assert( len(ref_image.shape) == 4 )
        batch,ch,w_raw,h_raw = ref_image.shape
        
        j_list = self.net.j_list
        l = len(j_list)        
        check_board = ref_image.new_zeros((l,1,w_raw,h_raw))
        counter_w = torch.arange(w_raw)
        index_w = torch.zeros(w_raw)
        counter_h = torch.arange(h_raw)
        index_h = torch.zeros(h_raw)

        for k in range(l):
            j = j_list[k]
            index_w = 1+((counter_w/j)%2) #either 1 or 2
            dx=index_w.float().view(w_raw,1)
            index_h = 1+((counter_h/j)%2) #either 1 or 2
            dy=index_h.float().view(1,h_raw)
            check_board[k,0,:,:]=0.25*(dy*dx) # dx*dy=1,2,4 multiply by 0.25 to have (0,1)
   
        assert( check_board.shape == (l,1,w_raw,h_raw) )

        # I need to sum:
        # check_board of shape: --> levels,1,w_raw,h_raw
        # ref_image of shape -----> batch,ch,w_raw,h_raw
        return ref_image.unsqueeze(1) + check_board 
        
    
    def __print_info_layer__(self,l,current_layer):
        print("At layer l= ",l," we have w_h_j_rf_wloc_hloc= ",l,current_layer)
    
    def describe_receptive_field(self,image):
        """ Show the value of ch_w_h_j_rf_loc as the tensor moves thorugh the net.
            Here:
            a. w,h are the width and height
            b. j is grid spacing
            c. rf is the maximum theoretical receptive field
            d. wloc,hloc are the location of the center of the first cell
        """
        w,h = image.shape[-2:]
        j =1
        rf =1
        w_loc =0.5
        h_loc =0.5
        current_layer = (w,h,j,rf,w_loc,h_loc)
        self.__print_info_layer__(0,current_layer)
        for i in range(len(self.net.s_p_k)):
            current_layer = self.__outFromIn__(self.net.s_p_k[i], current_layer)
            self.__print_info_layer__(i+1,current_layer)
            
    
    def __outFromIn__(self,s_p_k,layerIN):
        w_in,h_in,j_in,rf_in,wloc_in,hloc_in = layerIN
        s = s_p_k[0]
        p = s_p_k[1]
        k = s_p_k[2]
    
        w_out = np.floor((w_in - k + 2*p)/s) + 1
        h_out = np.floor((h_in - k + 2*p)/s) + 1
        
        pad_w = ((w_out-1)*s - w_in + k)/2 
        pad_h = ((h_out-1)*s - h_in + k)/2 
  
        j_out = j_in * s
        rf_out = rf_in + (k - 1)*j_in
        wloc_out = wloc_in + ((k-1)/2 - pad_w)*j_in
        hloc_out = hloc_in + ((k-1)/2 - pad_h)*j_in
        return int(w_out),int(h_out),j_out,int(rf_out),wloc_out,hloc_out

##########################################################################################################   
##########################################################################################################   
##########################################################################################################   
    
class cnn_net(torch.nn.Module):
    """ CNN_Net with geometry specified by the parameters """
    def __init__(self, params: dict):
        super().__init__()
              
        # Parameters for cnn 
        self.n_up_conv    = params['UNET.N_up_conv']
        self.n_max_pool   = params['UNET.N_max_pool']   
        self.n_pred_maps  = params['UNET.N_prediction_maps'] 
        self.ch_after_first_two_conv = params['UNET.CH_after_first_two_conv']
        self.ch_raw_image = len(params['IMG.ch_in_description'])
        
        # Initializations
        ch = self.ch_after_first_two_conv
        j = 1
        self.j_list  = [ j ]
        self.ch_list = [ ch ] 
       
        # Down path to center
        self.cnn = torch.nn.ModuleList([double_convolution_block(self.ch_raw_image,self.ch_list[-1])])
        for i in range(self.n_max_pool):  
            j=j*2
            ch=ch*2
            self.ch_list.append(ch)
            self.j_list.append(j)
            self.cnn.append(down_block(self.ch_list[-2],self.ch_list[-1]))   
        
        # compute s_p_k
        self.s_p_k = list()
        for module in self.cnn:
            self.s_p_k = module.__add_to_spk_list__(self.s_p_k)
        
        # Prediction maps
        self.pred_maps = torch.nn.ModuleList()
        for i in range(0,self.n_pred_maps):
            k=i+1
            self.pred_maps.append(prediction_Zwhere_Zwhat_Zmask(self.ch_list[-k],params))
            
               
    def forward(self, x, verbose):
        if(verbose):
            print("INPUT ---> shape ",x.shape)
        
        predictions = deque()
        n_layers = len(self.ch_list)
        for i, down in enumerate(self.cnn):
            dist_to_end_of_net = n_layers-i
            if( dist_to_end_of_net < self.n_pred_maps):
                predictions.append(self.pred_maps[dist_to_end_of_net](x))
                        
            x = down(x)
            
            if(verbose):
                print("level  ",i+1," shape ",x.shape)
        
        # always add a pred_map to the rightmost layer (which had distance 0 from the end of the net)
        predictions.append(self.pred_maps[0](x))
                
        return predictions

##########################################################################################################   
##########################################################################################################   
##########################################################################################################  


class partial_unet(torch.nn.Module):
    """ PARTIAL UNET with geometry specified by the parameters """
    def __init__(self, params: dict):
        super().__init__()

        # Parameters UNet
        self.n_up_conv    = params['UNET.N_up_conv']
        self.n_max_pool   = params['UNET.N_max_pool']
        self.n_pred_maps  = params['UNET.N_prediction_maps']
        self.ch_after_first_two_conv = params['UNET.CH_after_first_two_conv']
        self.ch_raw_image = len(params['IMG.ch_in_description'])

        # Initializations
        ch = self.ch_after_first_two_conv
        j = 1
        self.j_list  = [ j ]
        self.ch_list = [ ch ]

        # Down path to center
        self.down_path = torch.nn.ModuleList([double_convolution_block(self.ch_raw_image,self.ch_list[-1])])
        for i in range(0,self.n_max_pool):
            j=j*2
            ch=ch*2
            self.ch_list.append(ch)
            self.j_list.append(j)
            self.down_path.append(down_block(self.ch_list[-2],self.ch_list[-1]))


        # Up path
        self.up_path = torch.nn.ModuleList()
        for i in range(0,self.n_up_conv):
            j=int(j/2)
            ch=int(ch/2)
            self.ch_list.append(ch)
            self.j_list.append(j)
            self.up_path.append(up_block(self.ch_list[-2],self.ch_list[-1]))

        # Compute s_p_k
        self.s_p_k = list()
        for module in self.down_path:
            self.s_p_k = module.__add_to_spk_list__(self.s_p_k)
        for module in self.up_path:
            self.s_p_k = module.__add_to_spk_list__(self.s_p_k)

        # Prediction maps
        self.pred_maps = torch.nn.ModuleList()
        for i in range(0,self.n_pred_maps):
            k=i+1
            self.pred_maps.append(prediction_Zwhere(self.ch_list[-k],params))
            #self.pred_maps.append(prediction_Zwhere_Zwhat_Zmask(self.ch_list[-k],params))              

    
    def concatenete_stack_of_identical_tuple(self,stack,new_name,torch_cat_dim):
        """ takes 2 tuple with the same field and crete new tuple with the same fields.
            Here is is assumed that the entries of the tuples are torch tensors """
        
        z = stack.pop()
        while 1<2:
            try:
                z_new = stack.pop()
                z = self.concatenete_2_tuple(z,z_new,new_name,torch_cat_dim=1)
            except IndexError: 
                # If I am here is b/c I am popping from an empty deque. It means I have to stop
                break     
        return z       
            
    def concatenete_2_tuple(self,tuple1,tuple2,name_new_tuple,torch_cat_dim):
        """ takes 2 tuple with the same field and crete new tuple with the same fields.
            Here is is assumed that the entries of the tuples are torch tensors """
        assert(tuple1._fields ==  tuple2._fields)
        list_subnames = tuple1._fields
        return collections.namedtuple(name_new_tuple, list_subnames)._make( [torch.cat((getattr(tuple1,subname),getattr(tuple2,subname)),dim=torch_cat_dim) for subname in list_subnames])
           
    def concatenate_tuple_of_tuple(self,predictions):
        """ Concatenating the different predictions maps along the n_box dimension """
        multiple_zs = predictions.pop()
        names_of_zs = multiple_zs._fields   
        output = dict()
        for name in names_of_zs:
            output[name] = getattr(multiple_zs,name)
           
        while 1<2:
            try:
                multiple_zs = predictions.pop()   
                for name in names_of_zs:
                    output[name] = self.concatenete_2_tuple(output[name],getattr(multiple_zs,name),name,torch_cat_dim=1)
            except IndexError: 
                # If I am here is b/c I am popping from an empty deque. It means I have to stop
                break

        # return     
        return collections.namedtuple('z_prediction', names_of_zs)._make([output[name] for name in names_of_zs])

    def forward(self, x, verbose):
        width_raw_image,height_raw_image = x.shape[-2:]
        
        if(verbose):
            print("INPUT ---> shape ",x.shape)

        # Down path and save the tensor which will need to be concatenated
        to_be_concatenated = deque()
        
        for i, down in enumerate(self.down_path):
            x = down(x)

            if(verbose):
                print("down   ",i+1," shape ",x.shape)

            dist_from_center = self.n_max_pool - i
            if(dist_from_center>0 and dist_from_center <= self.n_up_conv):
                to_be_concatenated.append(x)


        # During up path I need to concatenate with the tensor obtained during the down path
        # If distance is < self.n_prediction_maps I need to export a prediction map
        prediction_stack = deque()
        for i, up in enumerate(self.up_path):
            dist_to_end_of_net = self.n_up_conv - i
            if( dist_to_end_of_net < self.n_pred_maps):
                prediction_stack.append(self.pred_maps[dist_to_end_of_net](x,width_raw_image,height_raw_image))

            x=up(to_be_concatenated.pop(),x)
            if(verbose):
                print("up     ",i," shape ",x.shape)

        # always add a pred_map to the rightmost layer (which had distance 0 from the end of the net)
        prediction_stack.append(self.pred_maps[0](x,width_raw_image,height_raw_image))
        
        
        # Decide what to return 
        if(len(prediction_stack)==1):
            return prediction_stack[0]
        elif(len(prediction_stack)>1):
            return self.concatenete_stack_of_identical_tuple(prediction_stack,"z_where",torch_cat_dim=1)
        elif(len(prediction_stack)<1):
            raise Exception('There are no predictions at the end of the UNET!')
            return None
            

        

