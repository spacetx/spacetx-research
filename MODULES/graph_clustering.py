import torch
import numpy
import skimage.color
import torch.nn.functional as F
import leidenalg as la
import igraph as ig
import functools 
from MODULES.namedtuple import COMMUNITY, Adjacency, TILING, SimplifiedPartition
from MODULES.utilities import roller_2d_first_quadrant
from typing import Optional
from matplotlib import pyplot as plt
import time


class GraphSegmentation(object):
    """ Produce a consensus segmentation mask by finding communities on a graph.
        Each node is a foreground pixel and each edge is the probabiliity that
        spatially nearby pixels belong to the same object.
        
        
        Typical usage:
        g = GraphSegmentation(tiling)

        disconnected_components = g.graph.clusters(mode="STRONG")
        community1=g.partition_2_community(disconnected_components)
        g.plot_community(community1, figsize=(20,20))


        partition = g.find_partition(resolution = 0.03)
        community2 = g.partition_2_community(partition, size_threshold=10)
        g.plot_community(community2, figsize=(20,20))
    """
    
    def __init__(self, tiling: TILING) -> None:
        super().__init__()
        
        # size = (2*r+1)*(2*r+1), w, h. Each channel contains the edges between pixel_i and pixel_j
        assert len(tiling.co_object.shape) == 3
        
        self.tiling = tiling
        self.device = self.tiling.raw_img.device
        ch, self.nx, self.ny = self.tiling.co_object.shape
        self.radius_nn = int(numpy.sqrt(ch) - 1)
        self.ch_edge_ii = 0
        print("radius_nn ->", self.radius_nn)
        print("ch_e_ii -->", self.ch_edge_ii)
        
        self.fg_mask = self.tiling.co_object[self.ch_edge_ii] > 0.1
        
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
        self._clusters = None

    @property
    def clusters(self):
        if self._clusters is None:
            print("computing connected clusters")
            start_time = time.time()
            self._clusters = self.graph.clusters(mode="STRONG")
            print("--- clustering time %s ---" % (time.time() - start_time))
        return self._clusters

    def _build_adjacency(self):
        w_list, i_list, j_list = [], [], []

        pad_list = [self.radius_nn+1, self.radius_nn+1, self.radius_nn+1, self.radius_nn+1]
        pad_index_matrix = F.pad(self.index_matrix, pad=pad_list, mode="constant", value=-1)
        pad_fg_mask = F.pad(self.fg_mask, pad=pad_list, mode="constant", value=False)
        pad_weight = F.pad(self.tiling.co_object, pad=pad_list, mode="constant", value=0.0)

        for ch, rolled in enumerate(roller_2d_first_quadrant(pad_index_matrix, radius_nn=self.radius_nn)):
            pad_index_matrix_shifted, dx, dy = rolled

            w = pad_weight[ch][pad_fg_mask]
            i = pad_index_matrix[pad_fg_mask]
            j = pad_index_matrix_shifted[pad_fg_mask]

            # Do not add loops.
            my_filter = (i >= 0) * (j >= 0) * (w > 0.01) * (i != j)  # avoid self loop

            w_list += w[my_filter].cpu().numpy().tolist()
            i_list += i[my_filter].cpu().numpy().tolist()
            j_list += j[my_filter].cpu().numpy().tolist()
                
        return Adjacency(edge_weight=w_list, source=i_list, destination=j_list)

    def _build_graph(self, adj: Adjacency):

        vertex_ids = [n for n in range(self.n_fg_pixel)]
        edgelist = list(zip(adj.source, adj.destination))
        graph = ig.Graph(vertex_attrs={"label": vertex_ids, "size": [1] * self.n_fg_pixel},
                         edges=edgelist,
                         edge_attrs={"weight": adj.weight},
                         directed=False)

        # Sometimes people normalize e_ij -> e_ij / sqrt(d_i d_j) where d_i = sum_j e_ij
        # After normalization sum_j e_ij ~ 1 for all vertex.
        # It is unclear if I need this or not

#         # Compute the sum of the edges connected to each vertex
#         i_array = torch.tensor(adj.source, dtype=torch.long, device=self.device)
#         j_array = torch.tensor(adj.destination, dtype=torch.long, device=self.device)
#         w_array = torch.tensor(adj.edge_weight, dtype=torch.float, device=self.device)
#         d = torch.zeros(self.n_fg_pixel, dtype=torch.float, device=self.device)
#         for v in vertex_list:
#             mask = ((i_array == v) + (j_array == v)) > 0  # check either source or destination involve vertex v
#             d[v] = w_array[mask].sum()
#         w_array /= torch.sqrt(d[i_array]*d[j_array])
#         graph.es['weight'] = list(w_array.cpu().numpy())
        return graph
    
    @functools.lru_cache(maxsize=10)
    def find_partition(self, resolution: int = 0.03, each_connected_component_separately: bool = True):

        if each_connected_component_separately:

            # find community in each connected cluster and then put the partition back together
            sizes = []
            membership = numpy.zeros(self.n_fg_pixel)
            n_clusters = len(self.clusters.sizes())
            for n, g in enumerate(self.clusters.subgraphs()):
                if (n % 100) == 0:
                    print("cluster %s out of %s" % (n, n_clusters))
                p = la.find_partition(graph=g,
                                      partition_type=la.CPMVertexPartition, #la.RBConfigurationVertexPartition, #la.RBERVertexPartition, #la.CPMVertexPartition,
                                      initial_membership=None,  # start from singleton
                                      weights=g.es['weight'],
                                      node_sizes=g.vs['size'],
                                      n_iterations=2,
                                      resolution_parameter=resolution)

                membership[g.vs['label']] = numpy.array(p.membership) + len(sizes)
                sizes += p.sizes()
            return SimplifiedPartition(sizes=sizes,
                                       membership=list(membership),
                                       resolution_parameter=resolution,
                                       modularity=0.0)

        else:
            partition = la.find_partition(graph=self.graph,
                                          partition_type=la.CPMVertexPartition, #la.RBConfigurationVertexPartition, #la.RBERVertexPartition, #la.CPMVertexPartition,
                                          initial_membership=None,  # start from singleton
                                          weights=self.graph.es['weight'],
                                          node_sizes=self.graph.vs['size'],
                                          n_iterations=2,
                                          resolution_parameter=resolution)
            return partition

    def partition_2_community(self, partition, size_threshold: int = 10):

        if not isinstance(partition, SimplifiedPartition):
            partition = SimplifiedPartition(sizes=partition.sizes(),
                                            membership=partition.membership,
                                            resolution_parameter=getattr(partition, "resolution_parameter", 0.0),
                                            modularity=partition.modularity)

        cell_ids = torch.arange(start=1, end=len(partition.sizes) + 1, step=1, device=self.device)  # +1 b/c bg=0, fg=1,2,...
        my_filter = torch.tensor(partition.sizes, device=self.device) > size_threshold
        cell_ids *= my_filter  # this makes the small cells having id=0
        pixel_membership = cell_ids[partition.membership]  # label for each pixel, pixel in small cells have label=0

        mask = torch.zeros_like(self.index_matrix, device=self.device)
        mask[self.x_coordinate_fg_pixel, self.y_coordinate_fg_pixel] = pixel_membership
        return COMMUNITY(mask=mask.cpu().numpy(),
                         n=torch.sum(my_filter).cpu().numpy(),
                         modularity=partition.modularity, 
                         resolution=getattr(partition, "resolution_parameter", 0))
    
    def plot_community(self, community, figsize: Optional[tuple] = None, windows: Optional[tuple] = None, **kargs):
        """ kargs can include:
            density=True, bins=50, range=(10,100), ...
        """
        assert isinstance(community, COMMUNITY)
        w = [0, community.mask.shape[-2], 0, community.mask.shape[-1]] if windows is None else windows
        sizes_fg = numpy.bincount(community.mask.flatten())[1:]

        figure, axes = plt.subplots(ncols=2, nrows=3, figsize=figsize)
        axes[0, 0].imshow(skimage.color.label2rgb(community.mask[w[0]:w[1], w[2]:w[3]],
                                                  numpy.zeros_like(community.mask[w[0]:w[1], w[2]:w[3]]),
                                                  alpha=1.0,
                                                  bg_label=0))
        axes[0, 1].imshow(skimage.color.label2rgb(community.mask[w[0]:w[1], w[2]:w[3]],
                                                  self.tiling.raw_img[0, w[0]:w[1], w[2]:w[3]].cpu().numpy(),
                                                  alpha=0.25,
                                                  bg_label=0))
        axes[1, 0].imshow(skimage.color.label2rgb(self.tiling.integer_mask[0, w[0]:w[1], w[2]:w[3]].cpu().numpy(),
                                                  numpy.zeros_like(community.mask[w[0]:w[1], w[2]:w[3]]),
                                                  alpha=1.0,
                                                  bg_label=0))
        axes[1, 1].imshow(skimage.color.label2rgb(self.tiling.integer_mask[0, w[0]:w[1], w[2]:w[3]].cpu().numpy(),
                                                  self.tiling.raw_img[0, w[0]:w[1], w[2]:w[3]].cpu().numpy(),
                                                  alpha=0.25,
                                                  bg_label=0))
        axes[2, 0].imshow(self.tiling.raw_img[0, w[0]:w[1], w[2]:w[3]].cpu(), cmap='gray')
        axes[2, 1].hist(sizes_fg, **kargs)

        axes[0, 0].set_title("N = "+str(community.n))
        axes[0, 1].set_title("resolution = "+str(community.resolution))
        axes[1, 0].set_title("one segmentation")
        axes[1, 1].set_title("one segmentation")
        axes[2, 0].set_title("raw image")
        axes[2, 1].set_title("size distribution")

    def plot_seg_masks(self, masks, ncols: int = 4, figsize: tuple = (20, 20)):
        assert isinstance(masks, list)
        max_row = int(numpy.ceil(len(masks)/ncols))
    
        figure, axes = plt.subplots(ncols=ncols, nrows=max_row, figsize=figsize)
        for n in range(len(masks)):
            row = n//ncols
            col = n % ncols
            axes[row, col].imshow(skimage.color.label2rgb(masks[n], numpy.zeros_like(masks[n]), alpha=1.0, bg_label=0))
