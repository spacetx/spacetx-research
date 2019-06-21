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
            - Discard any voxels which has a IoU> IoU_threshold with the box just chosen.
    """

    def __init__(self,params: dict):
        super().__init__()
        self.p_threshold       = params['NMS.p_threshold']
        self.overlap_threshold = params['NMS.overlap_threshold']
        self.n_max_objects     = params['PRIOR.n_max_objects'] 
        
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
    
    
    
    def compute_nms_mask(self,z_where,randomize_score_nms):
        """ Compute NMS mask """
        
        # # compute x1,x3,y1,y3
        x1    = (z_where.bx_dimfull - 0.5*z_where.bw_dimfull).squeeze(-1)
        x3    = (z_where.bx_dimfull + 0.5*z_where.bw_dimfull).squeeze(-1)
        y1    = (z_where.by_dimfull - 0.5*z_where.bh_dimfull).squeeze(-1)
        y3    = (z_where.by_dimfull + 0.5*z_where.bh_dimfull).squeeze(-1)
        area  = (z_where.bw_dimfull * z_where.bh_dimfull).squeeze(-1)
        batch_size, n_boxes = x1.shape
        
        # computes the overlap measure, this is O(N^2) algorithm
        # Note that cluster_mask is of size: (batch x n_boxes x n_boxes) ans has entry 0.0 or 1.0
        overlap_measure = self.compute_intersection_over_min_area(x1,x3,y1,y3,area) # shape: batch x n_box x n_box
        cluster_mask = (overlap_measure > self.overlap_threshold).float()           # shape: batch x n_box x n_box
            
        # This is the NON-MAX-SUPPRESSION algorithm:
        # Preparation
        score = z_where.prob.permute(0,2,1) #shape batch x 1 x n_box
        if(randomize_score_nms):
            score = torch.rand_like(score)
        assert (batch_size, 1, n_boxes) == score.shape
        possible  = (score > self.p_threshold).float() # shape: batch x 1 x n_box, objects must have score > p_threshold
        idx = torch.arange(start=0,end=n_boxes,step=1,device=p_raw.device).view(1,n_boxes,1).long()
        chosen = torch.zeros((batch_size,n_boxes,1),device=p_raw.device).float() # shape: batch x n_box x 1
    
        # Loop
        for l in range(self.n_max_objects):     
        #while (possible != 0.0).any():
            #l=l+1
            #print("v3",l)
            score_mask = cluster_mask*(score*possible)                      # shape: batch x n_box x n_box
            index = torch.max(score_mask,keepdim=True,dim=-1)[1]            # shape: batch x n_box x 1
            chosen += possible.permute(0,2,1)*(idx == index).float()    # shape: batch x n_box x 1
            blocks = torch.sum(cluster_mask*chosen,keepdim=True,dim=-2) # shape: batch x 1 x n_box 
            possible *= (blocks==0).float()                             # shape: batch x 1 x n_box
        return chosen # shape: batch x n_box x 1
        
    def forward(self,z_where,randomize_score_nms):
                
        with torch.no_grad():
            # compute yolo mask
            nms_mask = self.compute_nms_mask(z_where,randomize_score_nms) 
            
        # mask the probability according to the NMS
        p_masked = z_where.prob*nms_mask #shape batch_size x n_boxes x 1 
        
        
        with torch.no_grad():
            # select the top_k boxes by probability
            batch_size,n_boxes = p_masked.shape[:2]
            p_top_k, top_k_indeces = torch.topk(p_masked.squeeze(-1), k=min(self.n_max_objects,n_boxes), dim=-1, largest=True, sorted=True)
            batch_size, k = top_k_indeces.shape 
            batch_indeces = torch.arange(start=0,end=batch_size,step=1,
                                         dtype=top_k_indeces.dtype,device=top_k_indeces.device).view(-1,1).expand(-1,k)
            # Next two lines are just to check that I did not mess up the indeces resampling algebra
            #p_top_k_v3 = p_masked[batch_indeces,top_k_indeces]
            #assert((p_top_k == p_top_k_v3).all()) 
            
        # package the output
        return collections.namedtuple('z_where', 'prob bx_dimfull by_dimfull bw_dimfull bh_dimfull')._make(
                [p_masked[batch_indeces,top_k_indeces],
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
            
