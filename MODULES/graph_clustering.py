import torch
import numpy
import skimage.color
import torch.nn.functional as F
import leidenalg as la
import igraph as ig
import functools 
from MODULES.namedtuple import COMMUNITY, Adjacency, TILING
from MODULES.utilities import roller_2d, plot_grid
from typing import Optional
from matplotlib import pyplot as plt


class GraphSegmentation(object):
    """ Produce a consensus segmentation mask by finding comminities on a graph.
        Each node is a foreground pixel and each edge is the probabilty that 
        spatially nearby pixels belong to the same object.
        
        
        Typical usage:
        g = GraphSegmentation(edges=integer_segmentation_masks)
        mask = consensus.mask()
    """
    
    def __init__(self, tiling: TILING) -> None:
        super().__init__()
        
        # size = (2*r+1)*(2*r+1), w, h. Each channel contains the edges between pixel_i and pixel_j
        assert len(tiling.co_object.shape) == 3
        
        self.tiling = tiling
        self.device = self.tiling.raw_img.device
        ch, self.nx, self.ny = self.tiling.co_object.shape
        self.radius_nn = int((numpy.sqrt(ch) - 1) // 2)
        print("radius_nn ->", self.radius_nn)
        
        self.ch_edge_ii = (ch - 1)//2
        print("ch_e_ii -->", self.ch_edge_ii)
        
        self.fg_mask = self.tiling.co_object[self.ch_edge_ii] > 0.25
        
        ix_matrix, iy_matrix = torch.meshgrid([torch.arange(self.nx, dtype=torch.long, device=self.device),
                                               torch.arange(self.ny, dtype=torch.long, device=self.device)])
        self.x_coordinate_fg_pixel = ix_matrix[self.fg_mask]
        self.y_coordinate_fg_pixel = iy_matrix[self.fg_mask]
        self.n_fg_pixel = self.x_coordinate_fg_pixel.shape[0]
        self.index_array = torch.arange(self.n_fg_pixel, dtype=torch.long, device=self.device)
        self.index_matrix = -1*torch.ones_like(ix_matrix)
        self.index_matrix[self.x_coordinate_fg_pixel, self.y_coordinate_fg_pixel] = self.index_array
        print("n_fg_pixel -->", self.n_fg_pixel)
                
        self.graph = self._build_graph(adj=self._build_adjacency())

    def _build_adjacency(self):
        w_list, i_list, j_list = [], [], []
        
        pad_list = [self.radius_nn+1, self.radius_nn+1, self.radius_nn+1, self.radius_nn+1]
        pad_index_matrix = F.pad(self.index_matrix, pad=pad_list, mode="constant", value=-1)
        pad_fg_mask = F.pad(self.fg_mask, pad=pad_list, mode="constant", value=False)
        pad_weight = F.pad(self.tiling.co_object, pad=pad_list, mode="constant", value=0.0)
        
        for ch, pad_index_matrix_shifted in enumerate(roller_2d(pad_index_matrix, radius_nn=self.radius_nn)):
            
            w = pad_weight[ch][pad_fg_mask]
            i = pad_index_matrix[pad_fg_mask]
            j = pad_index_matrix_shifted[pad_fg_mask]
            
            w_tmp = w[w > 0.01]
            i_tmp = i[w > 0.01]
            j_tmp = j[w > 0.01]
            
            w_list += w_tmp[j_tmp >= 0].cpu().numpy().tolist()
            i_list += i_tmp[j_tmp >= 0].cpu().numpy().tolist()
            j_list += j_tmp[j_tmp >= 0].cpu().numpy().tolist()
                
        return Adjacency(edge_weight=w_list, source=i_list, destination=j_list)

    def _build_graph(self, adj: Adjacency):
        
        vertex_list = [n for n in range(self.n_fg_pixel)]
        edgelist = list(zip(adj.source, adj.destination))
        
        graph = ig.Graph(vertex_attrs={"label": vertex_list}, edges=edgelist, directed=False)
        graph.es['weight'] = adj.edge_weight
        return graph
    
    @functools.lru_cache(maxsize=10)
    def find_profile(self, resolution_range=(0.01, 0.1)):
        optimiser = la.Optimiser()
        profile = optimiser.resolution_profile(self.graph, la.CPMVertexPartition,
                                               resolution_range=resolution_range,
                                               weights=self.graph.es['weight'])
        return profile

    @functools.lru_cache(maxsize=10)
    def find_partition(self, resolution: int = 0.03):
        partition = la.find_partition(self.graph, la.CPMVertexPartition, 
                                      resolution_parameter=resolution,
                                      weights=self.graph.es['weight'])
        return partition

    def profile_2_communities(self, profile, size_threshold: int = 10):
        communities = [self.partition_2_community(partition, size_threshold) for partition in profile]
        return communities

    def partition_2_community(self, partition, size_threshold: int = 10):
        instance_ids = torch.tensor(partition.membership, device=self.device) + 1  # +1 b/c label_bg=0, label_fg=1,2,...
        
        n_instance = 0
        for n, size in enumerate(partition.sizes()):
            if size < size_threshold:
                instance_ids[instance_ids == n+1] = 0   # small community are set to bg value
            else:
                n_instance += 1
                
        mask = torch.zeros_like(self.index_matrix)
        mask[self.x_coordinate_fg_pixel, self.y_coordinate_fg_pixel] = instance_ids
        return COMMUNITY(mask=mask.cpu().numpy(),
                         n=n_instance,
                         modularity=partition.modularity, 
                         resolution=getattr(partition, "resolution_parameter", 0))
    
    def plot_community(self, community, figsize: Optional[tuple] = None):
        assert isinstance(community, COMMUNITY)

        figure, axes = plt.subplots(ncols=3, figsize=figsize)
        axes[0].imshow(skimage.color.label2rgb(community.mask, numpy.zeros_like(community.mask), alpha=1.0, bg_label=0))
        axes[1].imshow(skimage.color.label2rgb(community.mask, self.tiling.raw_img[0].cpu(), alpha=0.25, bg_label=0))
        axes[2].imshow(self.tiling.raw_img[0].cpu(), cmap='gray')
        axes[0].set_title("N = "+str(community.n))
        axes[1].set_title("resolution = "+str(community.resolution))
        axes[2].set_title("raw image")

    def plot_seg_masks(self, masks, ncols: int = 4, figsize: tuple = (20, 20)):
        assert isinstance(masks, list)
        max_row = int(numpy.ceil(len(masks)/ncols))
    
        figure, axes = plt.subplots(ncols=ncols, nrows=max_row, figsize=figsize)
        for n in range(len(masks)):
            row = n//ncols
            col = n % ncols
            axes[row, col].imshow(skimage.color.label2rgb(masks[n], numpy.zeros_like(masks[n]), alpha=1.0, bg_label=0))
