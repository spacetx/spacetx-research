import torch
import numpy
import skimage.segmentation
import skimage.color
import torch.nn.functional as F
import leidenalg as la
import igraph as ig
from MODULES.namedtuple import Segmentation, Partition, Similarity, Suggestion, Concordance
from typing import Optional, List
from matplotlib import pyplot as plt
import time

with torch.no_grad():
    class GraphSegmentation(object):
        """ Produce a consensus segmentation mask by finding communities on a graph.
            Each node is a foreground pixel and each edge is the similarity between spatially nearby pixels.
            The similarity measures if the pixels belong to the same object.

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

            assert len(segmentation.similarity.data.shape) == 4  # b, ch, width, height
            b, ch, ni, nj = segmentation.similarity.data.shape
            assert b == 1

            self.device = segmentation.integer_mask.device
            self.raw_image = segmentation.raw_image[0]
            self.example_integer_mask = segmentation.integer_mask[0, 0]  # set batch=0, ch=0

            # Build the graph only with foreground points which are connected
            vertex_mask = segmentation.fg_prob[0, 0] > min_fg_prob

            if min_edge_weight is not None:
                vertex_mask *= torch.max(segmentation.similarity.data[0], dim=-3)[0] > min_edge_weight
            if max_edge_weight is not None:
                vertex_mask *= torch.min(segmentation.similarity.data[0], dim=-3)[0] < max_edge_weight

            self.n_fg_pixel = torch.sum(vertex_mask).item()
            self.radius_nn = numpy.max(segmentation.similarity.ch_to_dxdy)

            i_matrix, j_matrix = torch.meshgrid([torch.arange(ni, dtype=torch.long, device=self.device),
                                                 torch.arange(nj, dtype=torch.long, device=self.device)])
            self.i_coordinate_fg_pixel = i_matrix[vertex_mask]
            self.j_coordinate_fg_pixel = j_matrix[vertex_mask]
            self.index_matrix = -1*torch.ones_like(i_matrix)
            self.index_matrix[self.i_coordinate_fg_pixel,
                              self.j_coordinate_fg_pixel] = torch.arange(self.n_fg_pixel,
                                                                         dtype=torch.long,
                                                                         device=self.device)

            labels = skimage.measure.label(vertex_mask, connectivity=2, background=0, return_num=False)
            membership_from_cc = torch.tensor(labels,
                                              dtype=torch.long,
                                              device=self.device)[self.i_coordinate_fg_pixel,
                                                                  self.j_coordinate_fg_pixel]
            self.partition_connected_components = Partition(type="connected",
                                                            membership=membership_from_cc,
                                                            sizes=torch.bincount(membership_from_cc),
                                                            params={})

            membership_from_example_segmask = self.example_integer_mask[self.i_coordinate_fg_pixel,
                                                                        self.j_coordinate_fg_pixel].long()
            self.partition_sample_segmask = Partition(type="one_sample",
                                                      membership=membership_from_example_segmask,
                                                      sizes=torch.bincount(membership_from_example_segmask),
                                                      params={})

            self.graph = self.similarity_2_graph(similarity=segmentation.similarity,
                                                 normalize_graph_edges=normalize_graph_edges)

        def similarity_2_graph(self, similarity: Similarity, normalize_graph_edges: bool = True) -> ig.Graph:

            assert similarity.data.shape[0] == 1  # making batch_shape = 1
            assert similarity.data.shape[-2:] == self.index_matrix.shape

            # Padding
            pad_list = [self.radius_nn + 1] * 4
            pad_index_matrix = F.pad(self.index_matrix, pad=pad_list, mode="constant", value=-1)
            pad_weight = F.pad(similarity.data, pad=pad_list, mode="constant", value=0.0)

            # Prepare the storage
            i_list, j_list, e_list = [], [], []  # i,j are verteces, e=edges
            sum_edges_at_vertex = torch.zeros(self.n_fg_pixel, device=self.device, dtype=torch.float)

            for ch, dxdy in enumerate(similarity.ch_to_dxdy):
                pad_index_matrix_shifted = torch.roll(torch.roll(pad_index_matrix, dxdy[0], dims=-2), dxdy[1],
                                                      dims=-1)

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

            # Build the graph
            vertex_ids = [n for n in range(self.n_fg_pixel)]
            edgelist = list(zip(i_list, j_list))

            if normalize_graph_edges:
                i = torch.tensor(i_list, dtype=torch.long, device=self.device)
                j = torch.tensor(j_list, dtype=torch.long, device=self.device)
                sqrt_d = torch.sqrt(sum_edges_at_vertex)

                edge_weight = torch.tensor(e_list)/(sqrt_d[i]*sqrt_d[j])
                total_edge_weight = torch.sum(edge_weight).item()
                e_list = edge_weight.tolist()
            else:
                total_edge_weight = sum(e_list)

            return ig.Graph(vertex_attrs={"label": vertex_ids},
                            edges=edgelist,
                            edge_attrs={"weight": e_list},
                            graph_attrs={"total_edge_weight":  total_edge_weight,
                                         "total_nodes": self.n_fg_pixel},
                            directed=False)

        def partition_2_mask(self, partition: Partition):
            segmask = torch.zeros_like(self.index_matrix)
            segmask[self.i_coordinate_fg_pixel, self.j_coordinate_fg_pixel] = partition.membership
            return segmask

        def is_vertex_in_window(self, window: tuple):
            """ Same convention as scikit image:
                window = (min_row, min_col, max_row, max_col)
                Return boolean array describing whether the vertex is in the spatial windows
            """
            row_filter = (self.i_coordinate_fg_pixel >= window[0]) * (self.i_coordinate_fg_pixel < window[2])
            col_filter = (self.j_coordinate_fg_pixel >= window[1]) * (self.j_coordinate_fg_pixel < window[3])
            vertex_in_window = row_filter * col_filter
            return vertex_in_window

        def subgraphs_by_partition_and_window(self,
                                              partition: Optional[Partition],
                                              window: Optional[tuple],
                                              include_bg: bool = False) -> List[ig.Graph]:
            """ same covention as scikit image: window = (min_row, min_col, max_row, max_col) """

            vertex_labels = torch.tensor(self.graph.vs["label"],
                                         device=self.device,
                                         dtype=torch.long)

            if (partition is None) and (window is None):
                # nothing to do
                for i in range(1):
                    yield self.graph
            elif (partition is None) and (window is not None):
                # return a single graph
                vertex_in_window = self.is_vertex_in_window(window)
                for i in range(1):
                    yield self.graph.subgraph(vertices=vertex_labels[vertex_in_window].tolist())
            elif partition is not None:
                # return many graphs

                if window is None:
                    vertex_in_window = torch.ones_like(partition.membership).bool()
                else:
                    vertex_in_window = self.is_vertex_in_window(window)

                n_start = 0 if include_bg else 1
                for n in range(n_start, len(partition.sizes)):
                    vertex_mask = (partition.membership == n) * vertex_in_window
                    sub_vertex_list = vertex_labels[vertex_mask].tolist()
                    yield self.graph.subgraph(vertices=sub_vertex_list)

        def suggest_resolution_parameter(self,
                                         window: Optional[tuple] = None,
                                         min_size: Optional[float] = 10,
                                         max_size: Optional[float] = None,
                                         cpm_or_modularity: str = "modularity",
                                         each_cc_separately: bool = False,
                                         show_graph: bool = True,
                                         figsize: tuple = (12, 12),
                                         fontsize: int = 20,
                                         sweep_range: Optional[numpy.ndarray] = None) -> Suggestion:
            """ This function select the resolution parameter which gives the hightest
                Intersection Over Union with the target partition.
                By default the target partition is self.partition_sample_segmask.

                To speed up the calculation the optimal resolution parameter is computed based
                on a windows of the original image. If window is None the entire image is used. This might be very slow

                The suggested resolution parameter is NOT necessarily optimal.
                Try smaller values to undersegment and larger value to oversegment. """

            # filter by window
            if window is None:
                window = [0, 0, self.raw_image.shape[-2], self.raw_image.shape[-1]]
                other_partition = self.partition_sample_segmask
            else:
                window = (max(0, window[0]),
                          max(0, window[1]),
                          min(self.raw_image.shape[-2], window[2]),
                          min(self.raw_image.shape[-1], window[3]))
                vertex_in_window = self.is_vertex_in_window(window)
                other_partition = self.partition_sample_segmask.filter_by_vertex(keep_vertex=vertex_in_window)

            resolutions = numpy.arange(0.5, 10, 0.5) if sweep_range is None else sweep_range
            iou = numpy.zeros_like(resolutions)
            seg = numpy.zeros((resolutions.shape[0], window[2]-window[0], window[3]-window[1]), dtype=numpy.int)
            delta_n_cells = numpy.zeros(resolutions.shape[0], dtype=numpy.int)

            for n, res in enumerate(resolutions):
                print("resolution sweep, {0:3d} out of {1:3d}".format(n+1, resolutions.shape[0]))
                p_tmp = self.find_partition_leiden(resolution=res,
                                                   window=window,
                                                   min_size=min_size,
                                                   max_size=max_size,
                                                   cpm_or_modularity=cpm_or_modularity,
                                                   each_cc_separately=each_cc_separately)

                c_tmp: Concordance = p_tmp.concordance_with_partition(other_partition=other_partition)
                delta_n_cells[n] = c_tmp.delta_n
                iou[n] = c_tmp.iou
                seg[n] = self.partition_2_mask(p_tmp)[window[0]:window[2], window[1]:window[3]].cpu().numpy()

            if show_graph:

                figure, ax = plt.subplots(figsize=figsize)
                ax.set_title("resolution parameter sweep", fontsize=fontsize)
                color = 'tab:red'
                _ = ax.plot(resolutions, delta_n_cells, 'x-', label="delta_n_cell", color=color)
                ax.set_xlabel("resolution", fontsize=fontsize)
                ax.set_ylabel('delta_n_cell', color=color, fontsize=fontsize)
                ax.tick_params(axis='y', labelcolor=color, labelsize=fontsize)
                ax.legend(loc='upper left')
                ax.grid()

                ax_2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
                color = 'tab:green'
                _ = ax_2.plot(resolutions, iou, 'o-', label="IoU", color=color)
                ax_2.set_ylabel('Intersection Over Union', color=color, fontsize=fontsize)
                ax_2.tick_params(axis='y', labelcolor=color, labelsize=fontsize)
                ax_2.legend(loc='upper right')

            i_max = numpy.argmax(iou)
            return Suggestion(best_resolution=resolutions[i_max],
                              best_index=i_max.item(),
                              sweep_resolution=resolutions,
                              sweep_iou=iou,
                              sweep_delta_n=delta_n_cells,
                              sweep_seg_mask=seg)

        # TODO USE BUILT-IN LEIDEN
        def find_partition_leiden(self,
                                  resolution: float,
                                  window: Optional[tuple] = None,
                                  min_size: Optional[float] = 10,
                                  max_size: Optional[float] = None,
                                  cpm_or_modularity: str = "modularity",
                                  each_cc_separately: bool = False) -> Partition:
            """ Find a partition of the graph by greedy maximization of CPM or Modularity metric.
                The graph can have both normalized and un-normalized weight.
                The metric can be both cpm or modularity
                The results are all similar (provided the resolution parameter is tuned correctly).

                If you want to use the suggest_resolution_parameter function with full automatic you should use either:
                1. CPM with normalized edge weight
                2. MODULARITY with UN Normalized edge_weight

                You can also pass a sweep_range.

                The resolution parameter can be increased (to obtain smaller communities) or
                decreased (to obtain larger communities).

                To speed up the calculation the graph partitioning can be done separately for each connected components.
                This is absolutely ok for CPM metric while a bit questionable for MOdularity metric.
                It is not likely to make much difference either way.
            """

            if cpm_or_modularity == "cpm":
                partition_type = la.CPMVertexPartition
                n = self.graph["total_nodes"]
                overall_graph_density = self.graph["total_edge_weight"] * 2.0 / (n * (n - 1))
                resolution = overall_graph_density * resolution
            elif cpm_or_modularity == "modularity":
                partition_type = la.RBConfigurationVertexPartition
            else:
                raise Exception("Warning!! Argument not recognized. \
                                           CPM_or_modularity can only be 'CPM' or 'modularity'")

            # Subset graph by connected components and windows if necessary
            max_label = 0
            membership = torch.zeros(self.n_fg_pixel, dtype=torch.long, device=self.device)
            partition_for_subgraphs = self.partition_connected_components if each_cc_separately else None

            for n, g in enumerate(self.subgraphs_by_partition_and_window(window=window,
                                                                         partition=partition_for_subgraphs)):

                # print("g summary", g.summary())
                if g.vcount() > 0:

                    p = la.find_partition(graph=g,
                                          partition_type=partition_type,
                                          initial_membership=None,
                                          weights=g.es['weight'],
                                          n_iterations=2,
                                          resolution_parameter=resolution)

                    labels = torch.tensor(p.membership, device=self.device, dtype=torch.long) + 1
                    shifted_labels = labels + max_label
                    max_label += torch.max(labels)
                    membership[g.vs['label']] = shifted_labels

            return Partition(type="leiden_cpm",
                             sizes=torch.bincount(membership),
                             membership=membership,
                             params={"resolution": resolution,
                                     "each_cc_separately": each_cc_separately}).filter_by_size(min_size=min_size,
                                                                                               max_size=max_size)

        def plot_partition(self, partition: Optional[Partition] = None,
                           figsize: Optional[tuple] = (12, 12),
                           window: Optional[tuple] = None,
                           **kargs) -> torch.tensor:
            """
                If partition is None it prints the connected components
                window has the same convention as scikit image, i.e. window = (min_row, min_col, max_row, max_col)
                kargs can include:
                density=True, bins=50, range=(10,100), ...
            """

            if partition is None:
                partition = self.partition_connected_components

            if window is None:
                w = [0, 0, self.raw_image.shape[-2], self.raw_image.shape[-1]]
                sizes_fg = partition.sizes[1:]
                sizes_fg = sizes_fg[sizes_fg > 0]

            else:
                sizes = torch.bincount(self.is_vertex_in_window(window=window) * partition.membership)
                sizes_fg = sizes[1:]  # no background
                sizes_fg = sizes_fg[sizes_fg > 0]
                w = window

            example_integer_mask = self.example_integer_mask[w[0]:w[2], w[1]:w[3]]
            sizes_fg_example = torch.bincount(example_integer_mask.flatten())
            sizes_fg_example = sizes_fg_example[1:]  # no background
            sizes_fg_example = sizes_fg_example[sizes_fg_example > 0]

            segmask = self.partition_2_mask(partition)[w[0]:w[2], w[1]:w[3]].cpu().numpy()
            example_integer_mask = example_integer_mask.cpu().numpy()
            raw_img = self.raw_image[0, w[0]:w[2], w[1]:w[3]].cpu().numpy()

            figure, axes = plt.subplots(ncols=3, nrows=2, figsize=figsize)
            axes[0, 0].imshow(skimage.color.label2rgb(label=segmask,
                                                      bg_label=0))
            axes[0, 1].imshow(skimage.color.label2rgb(label=segmask,
                                                      image=raw_img,
                                                      alpha=0.25,
                                                      bg_label=0))
            axes[0, 2].hist(sizes_fg, **kargs)

            axes[1, 0].imshow(skimage.color.label2rgb(label=example_integer_mask,
                                                      bg_label=0))
            axes[1, 1].imshow(skimage.color.label2rgb(label=example_integer_mask,
                                                      image=raw_img,
                                                      alpha=0.25,
                                                      bg_label=0))
            axes[1, 2].imshow(raw_img, cmap='gray')

            # titles
            title_sample = 'one sample, #cells -> {0:3d}'.format(sizes_fg_example.shape[0])
            if hasattr(partition.params, "resolution"):
                title_partition = '{0:s}, #cells -> {1:3d}, resolution -> {2:.23f}'.format(partition.type,
                                                                                           sizes_fg.shape[0],
                                                                                           partition.params["resolution"])
            else:
                title_partition = '{0:s}, #cells -> {1:3d}'.format(partition.type, sizes_fg.shape[0])
            axes[0, 0].set_title(title_partition)
            axes[0, 1].set_title(title_partition)
            axes[0, 2].set_title("size distribution")
            axes[1, 0].set_title(title_sample)
            axes[1, 1].set_title(title_sample)
            axes[1, 2].set_title("raw image")

