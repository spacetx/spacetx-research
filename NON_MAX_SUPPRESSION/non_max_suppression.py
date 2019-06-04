# %load NON_MAX_SUPPRESSION/non_max_suppression.py
import torch
import collections

class Non_Max_Suppression(torch.nn.Module):
    """ Use Intersection_over_Union criteria to put most of the entries to zero while leaving few detection unchanged.
        INPUT  has shape: BATCH x N_BOXES x .... 
        OUTPUT has shape: BATCH x K_MAX   x ....
                
        The non_max_suppression algorithm is as follows:
        1. Discard all voxels with p<p_threshold
        2. While there are any remaining voxels:
            - Pick the voxel with largest p and output it as a prediction
            - Discard any voxels which has a IoU> IoU_threshold with the box just exported.
    """

    def __init__(self,params: dict):
        super().__init__()
        self.p_threshold       = params['NMS.p_threshold']
        self.overlap_threshold = params['NMS.overlap_threshold']
        self.n_max_object      = params['PRIOR.n_max_objects'] 
        
    def unroll_and_compare(self,x,label):
        """ Given a vector of size: batch x n_boxes 
            it creates a matrix of size: batch x n_boxes x n_boxes 
            obtained by comparing all vecotr entries with all other vector entries 
            The comparison is either: MIN,MAX,GE,GT,LE,LT,EQ  """
        assert len(x.shape) == 2
        batch_size,nbox = x.shape
        tmp_A = x.view(batch_size,1,nbox).expand(-1,nbox,nbox)
        tmp_B = x.view(batch_size,nbox,1).expand(-1,nbox,nbox)
        if(label=="MAX"):
            return torch.max(tmp_A,tmp_B)
        elif(label=="MIN"):
            return torch.min(tmp_A,tmp_B)
        elif(label=="GE"):
            return (tmp_A >= tmp_B)
        elif(label=="GT"):
            return (tmp_A > tmp_B)
        elif(label=="LE"):
            return (tmp_A <= tmp_B)
        elif(label=="LT"):
            return (tmp_A < tmp_B)
        elif(label=="EQ"):
            return (tmp_A == tmp_B)
        else:
            raise Exception("label is unknown. It is ",label)

    def compute_intersection_over_min_area(self,x1,x3,y1,y3,area):
        """ compute the matrix of shape: batch x n_boxes x n_boxes with the Intersection Over Unions """
       
        min_area = self.unroll_and_compare(area,"MIN") #min of area between box1 and box2
        xi1 = self.unroll_and_compare(x1,"MAX") #max of x1 between box1 and box2
        yi1 = self.unroll_and_compare(y1,"MAX") #max of y1 between box1 and box2
        xi3 = self.unroll_and_compare(x3,"MIN") #min of x3 between box1 and box2
        yi3 = self.unroll_and_compare(y3,"MIN") #min of y3 between box1 and box2
        
        intersection_area = torch.clamp(xi3-xi1,min=0)*torch.clamp(yi3-yi1,min=0) 
        return intersection_area/min_area
    
    
    
    def compute_nms_mask(self,z_where):
        """ Compute NMS mask """
        
        # # compute x1,x3,y1,y3 and p_raw
        p_raw = z_where.prob.squeeze(-1)
        x1    = (z_where.bx_dimfull - 0.5*z_where.bw_dimfull).squeeze(-1)
        x3    = (z_where.bx_dimfull + 0.5*z_where.bw_dimfull).squeeze(-1)
        y1    = (z_where.by_dimfull - 0.5*z_where.bh_dimfull).squeeze(-1)
        y3    = (z_where.by_dimfull + 0.5*z_where.bh_dimfull).squeeze(-1)
        area  = (z_where.bw_dimfull * z_where.bh_dimfull).squeeze(-1)
        
        # computes the overlap measure, this is O(N^2) algorithm
        # Note that cluster_mask is of size: (batch x n_boxes x n_boxes) ans has entry 0.0 or 1.0
        overlap_measure = self.compute_intersection_over_min_area(x1,x3,y1,y3,area) # shape: batch x n_box x n_box
        cluster_mask = (overlap_measure > self.overlap_threshold).float()      # shape: batch x n_box x n_box
            
        # This is the NON-MAX-SUPPRESSION algorithm:
        # Preparation
        batch_size, n_boxes = p_raw.shape
        possible  = (p_raw > self.p_threshold).float() # chosen objects must have p > p_threshold
        #possible  = (p_raw > max(self.p_threshold,1E-10)).float() # chosen objects must have p > p_threshold
        nms_mask = torch.zeros_like(p_raw)
        idx = torch.arange(n_boxes).unsqueeze(0).expand(batch_size,-1).to(p_raw.device)
        
        # Loop
        for l in range(self.n_max_object): # I never need more that this since downstream I select the top few box by probability
            p_mask = ((p_raw*possible).view(batch_size,1,n_boxes))*(cluster_mask)
            index = torch.max(p_mask,dim=-1)[1]            
            nms_mask += possible*(idx == index).float()
            impossible = torch.matmul(cluster_mask,nms_mask.unsqueeze(-1)).squeeze(-1)
            possible = 1.0 - torch.clamp(impossible,max=1)
            #tmp_cazzo = torch.sum(nms_mask,dim=-1)
            #print("l, nms_mask",l,torch.min(tmp_cazzo),torch.max(tmp_cazzo))
            #if(l>=10):
            #    print("nms",l)
            if( (possible == 0.0).all() ):
                break
                
        return nms_mask.int()
    

        
    def forward(self,z_where):
                
        with torch.no_grad():
            # compute yolo mask
            nms_mask = self.compute_nms_mask(z_where) 
            
        # mask the probability according to the NMS
        p_masked = (z_where.prob.squeeze(-1))*(nms_mask.float()) 
        
        
        with torch.no_grad():
            # select the top_k boxes by probability
            batch_size,n_boxes = p_masked.shape
            p_top_k, top_k_indeces = torch.topk(p_masked, k=min(self.n_max_object,n_boxes), dim=-1, largest=True, sorted=True)
            batch_size, k = top_k_indeces.shape 
            batch_indeces = torch.arange(batch_size).unsqueeze(-1).expand(-1,k).to(top_k_indeces.device)
            
            # Next two lines are just to check that I did not mess up the indeces resampling algebra
            p_top_k_v3 = p_masked[batch_indeces,top_k_indeces]
            assert((p_top_k == p_top_k_v3).all()) 
            
        # package the output
        return collections.namedtuple('z_where', 'prob bx_dimfull by_dimfull bw_dimfull bh_dimfull')._make(
                [p_masked.unsqueeze(-1)[batch_indeces,top_k_indeces],
                 z_where.bx_dimfull[batch_indeces,top_k_indeces],
                 z_where.by_dimfull[batch_indeces,top_k_indeces],
                 z_where.bw_dimfull[batch_indeces,top_k_indeces],
                 z_where.bh_dimfull[batch_indeces,top_k_indeces]])
        
        
        #### for debug here I am just selecting the first k-boxes
        ###with torch.no_grad():
        ###    batch_size,n_boxes = z_where.prob.shape[:2]
        ###    k=min(self.n_max_object,n_boxes)
        ###    top_k_indeces = torch.randint(low=0,high=n_boxes,size=(batch_size,k)).to(z_where.prob.device)
        ###    batch_indeces = torch.arange(batch_size).unsqueeze(-1).expand(-1,k).to(top_k_indeces.device)
        ###
        #### package the output
        ###return collections.namedtuple('z_where', 'prob bx_dimfull by_dimfull bw_dimfull bh_dimfull')._make(
        ###    [z_where.prob[batch_indeces,top_k_indeces],
        ###     z_where.bx_dimfull[batch_indeces,top_k_indeces],
        ###     z_where.by_dimfull[batch_indeces,top_k_indeces],
        ###     z_where.bw_dimfull[batch_indeces,top_k_indeces],
        ###     z_where.bh_dimfull[batch_indeces,top_k_indeces]])
            
