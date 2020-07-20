import torch
import numpy
import skimage.segmentation
import skimage.color
import torch.nn.functional as F
import leidenalg as la
import igraph as ig
import functools 
from MODULES.namedtuple import Segmentation, Partition, Similarity
from typing import Optional, Union
from matplotlib import pyplot as plt
import time
import hdbscan

#@ TODO
#EVERYTHING IN PYTORCH

with torch.no_grad():

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

        def __init__(self, segmentation: Segmentation,
                     min_fg_prob: float = 0.1,
                     min_edge_weight: Optional[float] = 0.01,
                     max_edge_weight: Optional[float] = None,
                     normalize_graph_edges: bool = False) -> None:
            super().__init__()

            # size = (2*r+1)*(2*r+1), w, h. Each channel contains the edges between pixel_i and pixel_j
            assert len(segmentation.similarity.data.shape) == 4  # b, ch, width, height
            b, ch, nx, ny = segmentation.similarity.data.shape
            assert b == 1

            self.device = segmentation.integer_mask.device
            self.min_edge_weight = min_edge_weight
            self.max_edge_weight = max_edge_weight
            self.raw_image = segmentation.raw_image[0]
            self.example_integer_mask = segmentation.integer_mask[0]

            # Build the graph only with foreground points which are connected
            vertex_mask = segmentation.fg_prob[0, 0] > min_fg_prob
            if min_edge_weight is not None:
                vertex_mask *= torch.max(segmentation.similarity.data[0], dim=-3)[0] > min_edge_weight
            if max_edge_weight is not None:
                vertex_mask *= torch.min(segmentation.similarity.data[0], dim=-3)[0] < max_edge_weight

            self.n_fg_pixel = torch.sum(vertex_mask).item()
            self.radius_nn = numpy.max(segmentation.similarity.ch_to_dxdy)

            ix_matrix, iy_matrix = torch.meshgrid([torch.arange(nx, dtype=torch.long, device=self.device),
                                                   torch.arange(ny, dtype=torch.long, device=self.device)])
            self.x_coordinate_fg_pixel = ix_matrix[vertex_mask]
            self.y_coordinate_fg_pixel = iy_matrix[vertex_mask]
            self.index_matrix = -1*torch.ones_like(ix_matrix)
            self.index_matrix[self.x_coordinate_fg_pixel, self.y_coordinate_fg_pixel] = torch.arange(self.n_fg_pixel,
                                                                                                     dtype=torch.long,
                                                                                                     device=self.device)

            self.graph, self.img_to_flood = self.similarity_2_graph_and_flood(similarity=segmentation.similarity,
                                                                              normalize_graph_edges=normalize_graph_edges)

            self.partition_connected_components = self.find_partition_connected_components()

        @staticmethod
        def filter_by_size(input: Union[Partition,torch.tensor], min_size: int) -> torch.tensor:
            """ If a cluster is too small, its label is set to zero (i.e. background value).
                The subsequent labels are shifted so that there are no gaps in the labels number.
            """
            if isinstance(input, Partition):
                membership = input.membership
            else:
                membership = input

            sizes = torch.bincount(membership)
            my_filter = sizes > min_size
            count = torch.cumsum(my_filter, dim=-1)
            old_2_new = (count - count[0]) * my_filter  # this makes sure that label 0 is always mapped to 0
            new_membership = old_2_new[membership]

            if isinstance(input, Partition):
                new_dict = input.params
                new_dict["filter_by_size"] = min_size
                return input._replace(membership=new_membership, params=new_dict, sizes=torch.bincount(new_membership))
            else:
                return new_membership

        def subgraphs_by_partition(self, partition: Partition, include_bg: bool = False):
            vertex_labels = torch.tensor(self.graph.vs["label"], device=self.device)
            n_start = 0 if include_bg else 1
            for n in range(n_start, partition.sizes.shape[0]):
                sub_vertex_list = vertex_labels[partition.membership == n].tolist()
                yield self.graph.subgraph(vertices=sub_vertex_list)

        def similarity_2_graph_and_flood(self, similarity: Similarity, normalize_graph_edges: bool = True) -> ig.Graph:

            assert similarity.data.shape[0] == 1  # making batch_shape = 1
            assert similarity.data.shape[-2:] == self.index_matrix.shape

            # Padding
            pad_list = [self.radius_nn+1]*4
            pad_index_matrix = F.pad(self.index_matrix, pad=pad_list, mode="constant", value=-1)
            pad_weight = F.pad(similarity.data, pad=pad_list, mode="constant", value=0.0)

            # Prepare the storage
            i_list, j_list, e_list = [], [], []  # i,j are verteces, e=edges
            sum_edges_at_vertex = torch.zeros(self.n_fg_pixel, device=self.device, dtype=torch.float)
            minimum_distance_to_border = 9999 * torch.ones_like(sum_edges_at_vertex)

            for ch, dxdy in enumerate(similarity.ch_to_dxdy):

                pad_index_matrix_shifted = torch.roll(torch.roll(pad_index_matrix, dxdy[0], dims=-2), dxdy[1], dims=-1)

                data = pad_weight[0, ch].flatten()
                row_ind = pad_index_matrix.flatten()
                col_ind = pad_index_matrix_shifted.flatten()

                # Do not add loops.
                my_filter = (row_ind >= 0) * (col_ind >= 0)

                e = data[my_filter]
                i = row_ind[my_filter]
                j = col_ind[my_filter]

                # compute the sum of edges touching any vertex
                sum_edges_at_vertex[i] += e
                sum_edges_at_vertex[j] += e

                # convert torch to numpy to list
                e_list += e.tolist()
                i_list += i.tolist()
                j_list += j.tolist()

    ####            # Minimum distance to a pixel of the wrong type
    ####            new_min = minimum_distance_to_border.clamp(max=distance)
    ####            #print(dxdy, new_min)
    ####
    ####            new_filter = e < 0.25  # Pixel ij are both fg and of opposite type
    ####            minimum_distance_to_border[i[new_filter]] = new_min[i[new_filter]]
    ####            minimum_distance_to_border[j[new_filter]] = new_min[j[new_filter]]
    ####
    ####            i_fg_and_j_bg = (row_ind >= 0) * (col_ind < 0)
    ####            i_bg_and_j_fg = (row_ind < 0) * (col_ind >= 0)
    ####            minimum_distance_to_border[row_ind[i_fg_and_j_bg]] = new_min[row_ind[i_fg_and_j_bg]]
    ####            minimum_distance_to_border[col_ind[i_bg_and_j_fg]] = new_min[col_ind[i_bg_and_j_fg]]

            # Build the image for the flooding
            img_to_flood = torch.zeros_like(self.index_matrix).float()
            img_to_flood[self.x_coordinate_fg_pixel,
                         self.y_coordinate_fg_pixel] = minimum_distance_to_border

            # Build the graph
            sqrt_d = numpy.sqrt(sum_edges_at_vertex)
            vertex_ids = [n for n in range(self.n_fg_pixel)]
            edgelist = list(zip(i_list, j_list))

            if normalize_graph_edges:
                edge_weight = [e/(sqrt_d[i] * sqrt_d[j]) for e, i, j in zip(e_list, i_list, j_list)]
            else:
                edge_weight = e_list

            return ig.Graph(vertex_attrs={"label": vertex_ids,
                                          "size": [1] * self.n_fg_pixel,
                                          "d": sum_edges_at_vertex},
                            edges=edgelist,
                            edge_attrs={"weight": edge_weight},
                            directed=False), img_to_flood

        def find_partition_connected_components(self, cluster_min_size=3):
            partition = self.graph.clusters(mode="STRONG")
            old_membership = torch.tensor(partition.membership, device=self.device, dtype=torch.long) + 1  # fg=1,2,3..
            new_membership = GraphSegmentation.filter_by_size(input=old_membership,
                                                              min_size=cluster_min_size)
            print("debug connected components", torch.bincount(new_membership))
            return Partition(type="connected_components",
                             sizes=torch.bincount(new_membership),
                             membership=new_membership,
                             params={"cluster_min_size": cluster_min_size})

        def find_partition_watershed(self, watershed_line: bool = False) -> Partition:

            img_labels = skimage.segmentation.watershed(-self.img_to_flood.cpu().numpy(),  # minus b/c watershed starts from minima
                                                        markers=None,
                                                        connectivity=1,
                                                        offset=None,
                                                        mask=self.index_matrix.cpu().numpy() > 0,
                                                        compactness=0,
                                                        watershed_line=watershed_line)

            sizes = list(numpy.bincount(img_labels.flatten())[1:])  # skip label=0 which is the background
            membership = list(img_labels[self.x_coordinate_fg_pixel, self.y_coordinate_fg_pixel])

            return Partition(type="watershed",
                             sizes=sizes,
                             membership=membership,
                             params={"watershed_line": watershed_line})

        @functools.lru_cache(maxsize=10)
        def find_partition_leiden(self, resolution: int = 0.03) -> Partition:

            max_label = 0
            membership = torch.zeros(self.n_fg_pixel, dtype=torch.long, device=self.device)
            for n, g in enumerate(self.subgraphs_by_partition(partition=self.partition_connected_components)):

                p = la.find_partition(graph=g,
                                      partition_type=la.RBConfigurationVertexPartition, # la.CPMVertexPartition,
                                      initial_membership=None,  # start from singleton
                                      weights=g.es['weight'],
                                      #node_sizes=g.vs['size'],
                                      n_iterations=2,
                                      resolution_parameter=resolution)

                labels = torch.tensor(p.membership, device=self.device, dtype=torch.long) + 1  # bg=0, fg=1,2,3,...
                shifted_labels = labels + max_label
                max_label += torch.max(labels)

                membership[g.vs['label']] = shifted_labels

            return Partition(type="leiden",
                             sizes=torch.bincount(membership),
                             membership=membership,
                             params={"resolution": resolution})

        @functools.lru_cache(maxsize=10)
        def find_partition_hdbscan(self) -> Partition:

            def csr_similarity_2_dense_distance(csr_similarity):
                distance = 1.0 - csr_similarity.todense()
                distance[distance == 1] = numpy.infty
                return distance.astype(numpy.double)

            clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=15,
                                        min_samples=15, allow_single_cluster=True)

            # find community in each connected cluster and then put the partition back together
            membership = torch.zeros(self.n_fg_pixel, dtype=torch.long, device=self.device)
            n_clusters = len(self.partition_connected_components.sizes)
            print(n_clusters)
            max_label = 0

            for n, g in enumerate(self.subgraphs_by_partition(partition=self.partition_connected_components)):

                #if (n % 100) == 0:
                #    print("cluster %s out of %s" % (n, n_clusters))

                csr_similarity = g.get_adjacency_sparse(attribute="weight")
                distance_matrix = csr_similarity_2_dense_distance(csr_similarity)
                clusterer.fit(distance_matrix)  # hdbscan. -1=bg, 0,1,2,3 = fg
                labels = torch.tensor(clusterer.labels_, device=self.device, dtype=torch.long) + 1  # bg=0, fg=1,2,3,...
                shifted_labels = labels + max_label
                max_label += torch.max(labels)

                membership[g.vs['label']] = shifted_labels

            return Partition(type="hdbscan",
                             sizes=torch.bincount(membership),
                             membership=membership,
                             params={"ciao": "ciao"})

        def plot_partition(self, partition: Partition,
                           size_threshold: int = 10,
                           figsize: Optional[tuple] = None,
                           windows: Optional[tuple] = None,
                           **kargs):
            """ kargs can include:
                density=True, bins=50, range=(10,100), ...
            """

            if size_threshold <= 0:
                new_membership = partition.membership
                sizes_fg = partition.sizes[1:]
            else:
                new_membership = self.filter_by_size(partition.membership, min_size=size_threshold)
                sizes_fg = torch.bincount(new_membership)[1:]

            n_instances = sizes_fg.shape[0]
            segmask = torch.zeros_like(self.index_matrix, device=self.device)
            segmask[self.x_coordinate_fg_pixel,
                    self.y_coordinate_fg_pixel] = new_membership

            w = [0, segmask.shape[-2], 0, segmask.shape[-1]] if windows is None else windows

            segmask = segmask.cpu().numpy()
            zeros_mask = numpy.zeros(segmask.shape)

            figure, axes = plt.subplots(ncols=2, nrows=3, figsize=figsize)
            axes[0, 0].imshow(skimage.color.label2rgb(segmask[w[0]:w[1], w[2]:w[3]],
                                                      zeros_mask[w[0]:w[1], w[2]:w[3]],
                                                      alpha=1.0,
                                                      bg_label=0))
            axes[0, 1].imshow(skimage.color.label2rgb(segmask[w[0]:w[1], w[2]:w[3]],
                                                      self.raw_image[0, w[0]:w[1], w[2]:w[3]].cpu().numpy(),
                                                      alpha=0.25,
                                                      bg_label=0))
            axes[1, 0].imshow(skimage.color.label2rgb(self.example_integer_mask[0, w[0]:w[1], w[2]:w[3]].cpu().numpy(),
                                                      zeros_mask[w[0]:w[1], w[2]:w[3]],
                                                      alpha=1.0,
                                                      bg_label=0))
            axes[1, 1].imshow(skimage.color.label2rgb(self.example_integer_mask[0, w[0]:w[1], w[2]:w[3]].cpu().numpy(),
                                                      self.raw_image[0, w[0]:w[1], w[2]:w[3]].cpu().numpy(),
                                                      alpha=0.25,
                                                      bg_label=0))
            axes[2, 0].imshow(self.raw_image[0, w[0]:w[1], w[2]:w[3]].cpu(), cmap='gray')
            axes[2, 1].hist(sizes_fg, **kargs)

            axes[0, 0].set_title("consensus, N_instances = "+str(n_instances))
            axes[0, 1].set_title("consensus, N_instances = "+str(n_instances))
            axes[1, 0].set_title("one segmentation")
            axes[1, 1].set_title("one segmentation")
            axes[2, 0].set_title("raw image")
            axes[2, 1].set_title("size distribution")

        def plot_cc(self):
            return self.plot_partition(partition=self.partition_connected_components)
