import neptune
import torch
import numpy
import skimage.segmentation
import skimage.color
import leidenalg as la
import igraph as ig
from typing import Optional, List, Iterable
from matplotlib import pyplot as plt
from MODULES.namedtuple import Segmentation, Partition, SparseSimilarity, Suggestion, ConcordanceIntMask
from MODULES.utilities_neptune import log_img_and_chart
from MODULES.utilities import concordance_integer_masks

# I HAVE LEARNED:
# 1. If I use a lot of negihbours then all methods are roughly equivalent b/c graph becomes ALL-TO-ALL
# 2. Radius=10 means each pixel has 121 neighbours
# 3. CPM does not suffer from the resolution limit which means that it tends to shave off small part from a cell.
# 4. For now I prefer to use a graph with normalized edges,
# modularity and single gigantic cluster (i.e. each_cc_component=False)

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
                     normalize_graph_edges: bool = False) -> None:
            super().__init__()

            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
                
            self.raw_image = segmentation.raw_image[0].to(self.device)
            self.example_integer_mask = segmentation.integer_mask[0, 0].to(self.device)  # set batch=0, ch=0

            # it should be able to handle both DenseSimilarity and SparseSimilarity
            b, c, ni, nj = segmentation.integer_mask.shape
            assert b == c == 1
            assert (1, 1, ni, nj) == segmentation.fg_prob.shape

            self.index_matrix = None
            self.n_fg_pixel = None
            self.i_coordinate_fg_pixel = None
            self.j_coordinate_fg_pixel = None
            self.graph = self.similarity_2_graph(similarity=segmentation.similarity,
                                                 fg_prob=segmentation.fg_prob,
                                                 min_fg_prob=min_fg_prob,
                                                 min_edge_weight=min_edge_weight,
                                                 normalize_graph_edges=normalize_graph_edges)

        def similarity_2_graph(self, similarity: SparseSimilarity,
                               fg_prob: torch.tensor,
                               min_fg_prob: float,
                               min_edge_weight: float,
                               normalize_graph_edges: bool) -> ig.Graph:
            """ Create the graph from the sparse similarity matrix """
            if not normalize_graph_edges:
                print("WARNING! You are going to create a graph without normalizing the edges by the sqrt of the node degree. \
                       Are you sure you know what you are doing?!")

            # Move operation on GPU is available. Only at the end move back to cpu
            if torch.cuda.is_available():
                fg_prob = fg_prob.cuda()
                sparse_matrix = similarity.sparse_matrix.cuda()
                similarity_index_matrix = similarity.index_matrix.cuda()
            else:
                sparse_matrix = similarity.sparse_matrix.cpu()
                similarity_index_matrix = similarity.index_matrix.cpu()

            assert sparse_matrix._nnz() > 0, "WARNING: Graph is empty. Nothing to do"

            # Map the location with small fg_prob to index = -1
            vertex_mask = fg_prob[0, 0] > min_fg_prob
            n_max = torch.max(similarity.index_matrix).item()
            transform_index = -1 * torch.ones(n_max + 1, dtype=torch.long, device=similarity_index_matrix.device)
            transform_index[similarity_index_matrix[vertex_mask]] = similarity_index_matrix[vertex_mask]

            # Do the remapping (old to medium)
            v_tmp = sparse_matrix._values()
            ij_tmp = transform_index[sparse_matrix._indices()]

            # Do the filtering
            my_filter = (v_tmp > min_edge_weight) * (ij_tmp[0, :] >= 0) * (ij_tmp[1, :] >= 0)
            v = v_tmp[my_filter]
            ij = ij_tmp[:, my_filter]
            
            # Shift the labels so that there are no gaps (medium to new)
            ij_present = (torch.bincount(ij.view(-1)) > 0)
            self.n_fg_pixel = ij_present.sum().item()
            medium_2_new = (torch.cumsum(ij_present, dim=-1) * ij_present) - 1
            ij_new = medium_2_new[ij]
            
            # Make a transformation of the index_matrix (old to new)
            transform_index.fill_(-1)
            transform_index[sparse_matrix._indices()[:, my_filter]] = ij_new
            self.index_matrix = transform_index[similarity_index_matrix]
            ni, nj = self.index_matrix.shape[-2:]

            i_matrix, j_matrix = torch.meshgrid([torch.arange(ni, dtype=torch.long, device=self.index_matrix.device),
                                                 torch.arange(nj, dtype=torch.long, device=self.index_matrix.device)])
            self.i_coordinate_fg_pixel = i_matrix[self.index_matrix >= 0]
            self.j_coordinate_fg_pixel = j_matrix[self.index_matrix >= 0]

#            # Check
#            tmp = -1 * torch.ones_like(self.index_matrix)
#            tmp[self.i_coordinate_fg_pixel,
#                self.j_coordinate_fg_pixel] = torch.arange(self.n_fg_pixel,
#                                                           dtype=torch.long,
#                                                           device=self.device)
#            assert (tmp == self.index_matrix).all()

            # Normalize the edges if necessary
            if normalize_graph_edges:
                # Before normalization v ~ 1.
                # After normalization v ~ 1/#neighbors so that sum_i v_ij ~ 1
                m = torch.sparse.FloatTensor(ij_new, v, torch.Size([self.n_fg_pixel, self.n_fg_pixel]))
                if m._nnz() > 0:
                    m_tmp = (torch.sparse.sum(m, dim=-1) + torch.sparse.sum(m, dim=-2)).coalesce()
                    sqrt_sum_edges_at_vertex = torch.sqrt(m_tmp._values())
                    v.div_(sqrt_sum_edges_at_vertex[ij_new[0]]*sqrt_sum_edges_at_vertex[ij_new[1]])
                else:
                    raise Exception("WARNING: Graph is empty. Nothing to do")

            print("Building the graph with python-igraph")
            return ig.Graph(vertex_attrs={"label": numpy.arange(self.n_fg_pixel, dtype=numpy.int64)},
                            edges=ij_new.permute(1, 0).cpu().numpy(),
                            edge_attrs={"weight": v.cpu().numpy()},
                            graph_attrs={"total_edge_weight": v.sum().item(),
                                         "total_nodes": self.n_fg_pixel},
                            directed=False)

        def get_cc_partition(self) -> Partition:
            labels = skimage.measure.label(self.index_matrix > 0, connectivity=2, background=0, return_num=False)
            membership = torch.tensor(labels,
                                      dtype=torch.long,
                                      device=self.device)[self.i_coordinate_fg_pixel,
                                                          self.j_coordinate_fg_pixel]
            return Partition(membership=membership, sizes=torch.bincount(membership))

        def get_simple_partition(self) -> Partition:
            membership = torch.tensor(self.example_integer_mask,
                                      dtype=torch.long,
                                      device=self.device)[self.i_coordinate_fg_pixel,
                                                          self.j_coordinate_fg_pixel]
            return Partition(membership=membership, sizes=torch.bincount(membership))

        def partition_2_integer_mask(self, partition: Partition):
            label = torch.zeros_like(self.index_matrix, 
                                     dtype=partition.membership.dtype,
                                     device=partition.membership.device)
            label[self.i_coordinate_fg_pixel, self.j_coordinate_fg_pixel] = partition.membership
            return label

        def is_vertex_in_window(self, window: tuple):
            """ Same convention as scikit image:
                window = (min_row, min_col, max_row, max_col)
                Return boolean array describing whether the vertex is in the spatial windows
            """
            row_filter = (self.i_coordinate_fg_pixel >= window[0]) * (self.i_coordinate_fg_pixel < window[2])
            col_filter = (self.j_coordinate_fg_pixel >= window[1]) * (self.j_coordinate_fg_pixel < window[3])
            vertex_in_window = row_filter * col_filter
            if (~vertex_in_window).all():
                raise Exception("All vertices are outside the chosen window. \
                                 This is wrong. Doulbe check your window specifications")
            return vertex_in_window

        def subgraphs_by_partition_and_window(self,
                                              partition: Optional[Partition],
                                              window: Optional[tuple],
                                              include_bg: bool = False) -> List[ig.Graph]:
            """ same covention as scikit image: window = (min_row, min_col, max_row, max_col) """

            if (partition is None) and (window is None):
                # nothing to do
                for i in range(1):
                    yield self.graph

            elif (partition is None) and (window is not None):
                # return a single graph by window
                mask_vertex_in_window = self.is_vertex_in_window(window)  # torch.bool
                vertex_label = torch.tensor(self.graph.vs["label"], dtype=torch.long, device=self.device)
                for i in range(1):
                    yield self.graph.subgraph(vertices=vertex_label[mask_vertex_in_window])

            elif partition is not None:
                # return many graphs

                if window is None:
                    vertex_in_window = torch.ones_like(partition.membership).bool()
                else:
                    vertex_in_window = self.is_vertex_in_window(window)

                vertex_label = torch.tensor(self.graph.vs["label"], dtype=torch.long, device=self.device)
                n_start = 0 if include_bg else 1
                for n in range(n_start, len(partition.sizes)):
                    vertex_mask = (partition.membership == n) * vertex_in_window
                    yield self.graph.subgraph(vertices=vertex_label[vertex_mask])

        def suggest_resolution_parameter(self,
                                         window: Optional[tuple] = None,
                                         min_size: Optional[float] = 20,
                                         max_size: Optional[float] = None,
                                         cpm_or_modularity: str = "cpm",
                                         each_cc_separately: bool = False,
                                         sweep_range: Optional[Iterable] = None) -> Suggestion:
            """ This function select the resolution parameter which gives the hightest
                Intersection Over Union with the target partition.
                By default the target partition is self.partition_sample_segmask.

                To speed up the calculation the optimal resolution parameter is computed based
                on a windows of the original image. If window is None the entire image is used. This might be very slow.
                
                Only CPM is scale invariant. 
                If using modularity the same resolution paerameter will give different results depending on the size of the analyzed window.

                The suggested resolution parameter is NOT necessarily optimal.
                Try smaller values to undersegment and larger value to oversegment.

                window = (min_row, min_col, max_row, max_col)
            """
            # filter by window
            if window is None:
                window = [0, 0, self.raw_image.shape[-2], self.raw_image.shape[-1]]
            else:
                window = (max(0, window[0]),
                          max(0, window[1]),
                          min(self.raw_image.shape[-2], window[2]),
                          min(self.raw_image.shape[-1], window[3]))

            other_integer_mask = self.example_integer_mask[window[0]:window[2], window[1]:window[3]].long()

            resolutions = torch.arange(0.5, 10, 0.5) if sweep_range is None else sweep_range
            iou = torch.zeros(len(resolutions), dtype=torch.float)
            mi = torch.zeros_like(iou)
            n_reversible_instances = torch.zeros_like(iou)
            total_intersection = torch.zeros_like(iou)
            integer_mask = torch.zeros((resolutions.shape[0], window[2]-window[0],
                                        window[3]-window[1]), dtype=torch.int)
            delta_n_cells = torch.zeros(resolutions.shape[0], dtype=torch.int)
            n_cells = torch.zeros_like(delta_n_cells)
            sizes_list = list()
            
            for n, res in enumerate(resolutions):
                if (n % 10 == 0) or (n == resolutions.shape[0]-1):
                    print("resolution sweep, {0:3d} out of {1:3d}".format(n, resolutions.shape[0]-1))
                
                p_tmp = self.find_partition_leiden(resolution=res,
                                                   window=window,
                                                   min_size=min_size,
                                                   max_size=max_size,
                                                   cpm_or_modularity=cpm_or_modularity,
                                                   each_cc_separately=each_cc_separately)
                int_mask = self.partition_2_integer_mask(p_tmp)[window[0]:window[2], window[1]:window[3]]
                sizes_list.append(p_tmp.sizes.cpu())

                n_cells[n] = len(p_tmp.sizes)-1
                integer_mask[n] = int_mask.cpu()
                c_tmp: ConcordanceIntMask = concordance_integer_masks(integer_mask[n].long(), other_integer_mask)
                delta_n_cells[n] = c_tmp.delta_n
                iou[n] = c_tmp.iou
                mi[n] = c_tmp.mutual_information
                n_reversible_instances[n] = c_tmp.n_reversible_instances
                total_intersection[n] = c_tmp.intersection_mask.sum().float()

            i_max = torch.argmax(iou).item()
            try:
                best_resolution = resolutions[i_max].item()
            except:
                best_resolution = resolutions[i_max]

            return Suggestion(best_resolution=best_resolution,
                              best_index=i_max,
                              sweep_resolution=resolutions,
                              sweep_mi=mi,
                              sweep_iou=iou,
                              sweep_delta_n=delta_n_cells,
                              sweep_seg_mask=integer_mask,
                              sweep_sizes=sizes_list,
                              sweep_n_cells=n_cells)

        # TODO use built-in leiden algorithm instead of leidenalg?
        def find_partition_leiden(self,
                                  resolution: float,
                                  window: Optional[tuple] = None,
                                  min_size: Optional[float] = 20,
                                  max_size: Optional[float] = None,
                                  cpm_or_modularity: str = "cpm",
                                  each_cc_separately: bool = False,
                                  n_iterations: int = 2,
                                  initial_membership: Optional[numpy.ndarray] = None) -> Partition:
            """ Find a partition of the graph by greedy maximization of CPM or Modularity metric.
                The graph can have both normalized and un-normalized weight.
                The strong recommendation is to use CPM with normalized edge weight.
                
                The metric can be both cpm or modularity
                The results are all similar (provided the resolution parameter is tuned correctly).

                If you want to use the suggest_resolution_parameter function with full automatic you should use either:
                1. CPM with normalized edge weight
                2. MODULARITY with UN Normalized edge_weight

                You can also pass a sweep_range.

                The resolution parameter can be increased (to obtain smaller communities) or
                decreased (to obtain larger communities).

                To speed up the calculation the graph partitioning can be done separately for each connected components.
                This is absolutely ok for CPM metric while a bit questionable for Modularity metric.
                It is not likely to make much difference either way.

                window has the same convention as scikit image, i.e. window = (min_row, min_col, max_row, max_col)
            """
            
            if cpm_or_modularity == "cpm":
                partition_type = la.CPMVertexPartition
                
                # TODO: Rescale the resolution by some (robust) properties of the full graph so that the right resolution parameter is about 1
                # raise NotImplementedError
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
            partition_for_subgraphs = self.get_cc_partition() if each_cc_separately else None

            for n, g in enumerate(self.subgraphs_by_partition_and_window(window=window,
                                                                         partition=partition_for_subgraphs)):
                
                # With this rescaling the value of the resolution parameter optimized 
                # for a small window can be used to segment a large window
                if cpm_or_modularity == "modularity":
                    tmp = numpy.sum(g.es["weight"]) / g["total_edge_weight"]
                    resolution = resolution * tmp

                if g.vcount() > 0:
                    # Only if the graph has node I tried to find the partition

                    print("find partition internal")
                    p = la.find_partition(graph=g,
                                          partition_type=partition_type,
                                          initial_membership=initial_membership,
                                          weights=g.es['weight'],
                                          n_iterations=n_iterations,
                                          resolution_parameter=resolution)

                    labels = torch.tensor(p.membership, device=self.device, dtype=torch.long) + 1
                    shifted_labels = labels + max_label
                    max_label += torch.max(labels)
                    membership[g.vs['label']] = shifted_labels
                    
            # TODO: filter_by_size is slow
            return Partition(sizes=torch.bincount(membership),
                             membership=membership).filter_by_size(min_size=min_size, max_size=max_size)

        def plot_partition(self, partition: Partition,
                           figsize: Optional[tuple] = (12, 12),
                           window: Optional[tuple] = None,
                           experiment: Optional[neptune.experiments.Experiment] = None,
                           neptune_name: Optional[str] = None,
                           **kargs) -> torch.tensor:
            """
                If partition is None it prints the connected components
                window has the same convention as scikit image, i.e. window = (min_row, min_col, max_row, max_col)
                kargs can include:
                density=True, bins=50, range=(10,100), ...
            """

            if partition is None:
                FIX
                partition = self.partition_connected_components

            if window is None:
                w = [0, 0, self.raw_image.shape[-2], self.raw_image.shape[-1]]
                sizes_fg = partition.sizes[1:]  # no background
            else:
                sizes = torch.bincount(self.is_vertex_in_window(window=window) * partition.membership)
                sizes_fg = sizes[1:]  # no background
                sizes_fg = sizes_fg[sizes_fg > 0]  # since I am filtering the vertex some sizes might become zero
                w = window

            integer_mask = self.partition_2_integer_mask(partition)[w[0]:w[2], w[1]:w[3]].cpu().long().numpy()  # shape: w, h
            image = self.raw_image[:, w[0]:w[2], w[1]:w[3]].permute(1, 2, 0).cpu().float().numpy()  # shape: w, h, ch
            if len(image.shape) == 3 and (image.shape[-1] != 3):
                image = image[..., 0]

            fig, axes = plt.subplots(ncols=2, nrows=2, figsize=figsize)
            axes[0, 0].imshow(skimage.color.label2rgb(label=integer_mask,
                                                      bg_label=0))
            axes[0, 1].imshow(skimage.color.label2rgb(label=integer_mask,
                                                      image=image,
                                                      alpha=0.25,
                                                      bg_label=0))
            axes[1, 0].imshow(image)
            axes[1, 1].hist(sizes_fg.cpu(), **kargs)
            
            title_partition = 'Partition, #cells -> '+str(sizes_fg.shape[0])
            axes[0, 0].set_title(title_partition)
            axes[0, 1].set_title(title_partition)
            axes[1, 0].set_title("raw image")
            axes[1, 1].set_title("size distribution")
            
            fig.tight_layout()
            if neptune_name is not None:
                log_img_and_chart(name=neptune_name, fig=fig, experiment=experiment)
            plt.close(fig)
            return fig


