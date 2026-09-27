# torch_geometric.nn on Hwacha

The modules and functions of pytorch-geometric.readthedocs.io/en/latest/modules/nn.html
(torch_geometric 2.8.0): the convolutional layers, aggregation operators, attention, normalization,
pooling and unpooling, models, KGE models, encodings, functional and dense sections, one case per
entry, run through PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc and checked on Spike against
PyTorch. Layout, generic host, `gen.sh` and Makefile are those of `../tafunc`; the cases live in
`export_tg.py`, the tensor re-implementations in `tg_lib.py`; `probe.sh` exports every case in its
own process (torch-mlir crashes on some graphs).

Every case runs on one small fixed graph: 8 nodes with 4 features, 16 directed edges (8 undirected
pairs, no self loops), 2 graphs of 4 nodes in the batch vector, 3-dim edge features / positions.
Edge indices, edge attributes, positions, batch vectors, boxes and labels are constant buffers, the
node features (or the messages, the positions, the atom types) are the input; integer outputs are
cast to float, several outputs are flattened and concatenated. Modules run in eval mode with random
weights (fixed seed). Aggregations take the 16 edge messages into their 8 target nodes; the pooling
layers keep ratio 0.5 (2 of 4 nodes per graph); the molecular models (SchNet, DimeNet, GNNFF, ViSNet)
take 8 atoms with random positions and a 5.0 cutoff.

## Cases: 182, all PASS

| group | case | Hwacha |
|---|---|---|
| conv | agnn_conv | PASS, max\|diff\| 0, 3,846 周期 |
| conv | anti_symmetric_conv | PASS, max\|diff\| 0, 3,361 周期 |
| conv | appnp | PASS, max\|diff\| 0, 2,760 周期 |
| conv | arma_conv | PASS, max\|diff\| 0, 3,119 周期 |
| conv | cg_conv | PASS, max\|diff\| 0, 1,955 周期 |
| conv | cheb_conv | PASS, max\|diff\| 0, 1,071 周期 |
| conv | cluster_gcn_conv | PASS, max\|diff\| 0, 646 周期 |
| conv | dir_gnn_conv | PASS, max\|diff\| 0, 3,488 周期 |
| conv | dna_conv | PASS, max\|diff\| 0, 3,862 周期 |
| conv | dynamic_edge_conv | PASS, max\|diff\| 0, 3,033 周期 |
| conv | edge_conv | PASS, max\|diff\| 0, 2,451 周期 |
| conv | eg_conv | PASS, max\|diff\| 0, 2,227 周期 |
| conv | fa_conv | PASS, max\|diff\| 0, 2,354 周期 |
| conv | fast_rgcn_conv | PASS, max\|diff\| 0, 826 周期 |
| conv | feast_conv | PASS, max\|diff\| 0, 3,364 周期 |
| conv | film_conv | PASS, max\|diff\| 0, 3,995 周期 |
| conv | gat_conv | PASS, max\|diff\| 0, 4,377 周期 |
| conv | gated_graph_conv | PASS, max\|diff\| 0, 3,554 周期 |
| conv | gatv2_conv | PASS, max\|diff\| 0, 4,478 周期 |
| conv | gcn2_conv | PASS, max\|diff\| 0, 2,044 周期 |
| conv | gcn_conv | PASS, max\|diff\| 0, 1,871 周期 |
| conv | gen_conv | PASS, max\|diff\| 0, 4,209 周期 |
| conv | general_conv | PASS, max\|diff\| 0, 3,760 周期 |
| conv | gin_conv | PASS, max\|diff\| 0, 1,252 周期 |
| conv | gine_conv | PASS, max\|diff\| 0, 1,553 周期 |
| conv | gmm_conv | PASS, max\|diff\| 0, 2,665 周期 |
| conv | gps_conv | PASS, max\|diff\| 0, 5,531 周期 |
| conv | graph_conv | PASS, max\|diff\| 0, 1,088 周期 |
| conv | hypergraph_conv | PASS, max\|diff\| 0, 1,608 周期 |
| conv | le_conv | PASS, max\|diff\| 0, 1,435 周期 |
| conv | lg_conv | PASS, max\|diff\| 0, 1,325 周期 |
| conv | mf_conv | PASS, max\|diff\| 0, 6,289 周期 |
| conv | mix_hop_conv | PASS, max\|diff\| 0, 2,814 周期 |
| conv | nn_conv | PASS, max\|diff\| 0, 1,266 周期 |
| conv | pan_conv | PASS, max\|diff\| 0, 1,479 周期 |
| conv | pdn_conv | PASS, max\|diff\| 0, 2,594 周期 |
| conv | pna_conv | PASS, max\|diff\| 0, 6,833 周期 |
| conv | point_gnn_conv | PASS, max\|diff\| 0, 3,009 周期 |
| conv | point_net_conv | PASS, max\|diff\| 0, 3,483 周期 |
| conv | point_transformer_conv | PASS, max\|diff\| 0, 5,013 周期 |
| conv | ppf_conv | PASS, max\|diff\| 0, 7,635 周期 |
| conv | res_gated_graph_conv | PASS, max\|diff\| 0, 1,932 周期 |
| conv | rgat_conv | PASS, max\|diff\| 0, 3,436 周期 |
| conv | rgcn_conv | PASS, max\|diff\| 0, 826 周期 |
| conv | sage_conv | PASS, max\|diff\| 0, 1,933 周期 |
| conv | sg_conv | PASS, max\|diff\| 0, 2,268 周期 |
| conv | signed_conv | PASS, max\|diff\| 0, 3,974 周期 |
| conv | simple_conv | PASS, max\|diff\| 0, 1,457 周期 |
| conv | spline_conv | PASS, max\|diff\| 0, 4,742 周期 |
| conv | ssg_conv | PASS, max\|diff\| 0, 2,537 周期 |
| conv | super_gat_conv | PASS, max\|diff\| 0, 4,655 周期 |
| conv | tag_conv | PASS, max\|diff\| 0, 2,357 周期 |
| conv | transformer_conv | PASS, max\|diff\| 0, 3,973 周期 |
| conv | wl_conv_continuous | PASS, max\|diff\| 0, 1,099 周期 |
| conv | x_conv | PASS, max\|diff\| 0, 4,375 周期 |
| aggr | attentional_aggregation | PASS, max\|diff\| 0, 2,873 周期 |
| aggr | deep_sets_aggregation | PASS, max\|diff\| 0, 1,021 周期 |
| aggr | degree_scaler_aggregation | PASS, max\|diff\| 0, 3,772 周期 |
| aggr | graph_multiset_transformer | PASS, max\|diff\| 0, 12,025 周期 |
| aggr | gru_aggregation | PASS, max\|diff\| 0, 5,147 周期 |
| aggr | lcm_aggregation | PASS, max\|diff\| 0, 5,131 周期 |
| aggr | lstm_aggregation | PASS, max\|diff\| 0, 5,550 周期 |
| aggr | max_aggregation | PASS, max\|diff\| 0, 1,768 周期 |
| aggr | mean_aggregation | PASS, max\|diff\| 0, 1,384 周期 |
| aggr | median_aggregation | PASS, max\|diff\| 0, 2,610 周期 |
| aggr | min_aggregation | PASS, max\|diff\| 0, 1,864 周期 |
| aggr | mlp_aggregation | PASS, max\|diff\| 0, 2,123 周期 |
| aggr | mul_aggregation | PASS, max\|diff\| 0, 1,348 周期 |
| aggr | multi_aggregation | PASS, max\|diff\| 0, 2,981 周期 |
| aggr | multi_aggregation_proj | PASS, max\|diff\| 0, 2,492 周期 |
| aggr | patch_transformer_aggregation | PASS, max\|diff\| 0, 7,981 周期 |
| aggr | power_mean_aggregation | PASS, max\|diff\| 0, 1,567 周期 |
| aggr | quantile_aggregation | PASS, max\|diff\| 0, 3,569 周期 |
| aggr | set2set | PASS, max\|diff\| 0, 6,809 周期 |
| aggr | set_transformer_aggregation | PASS, max\|diff\| 0, 12,095 周期 |
| aggr | softmax_aggregation | PASS, max\|diff\| 0, 2,706 周期 |
| aggr | sort_aggregation | PASS, max\|diff\| 0, 4,564 周期 |
| aggr | std_aggregation | PASS, max\|diff\| 0, 2,036 周期 |
| aggr | sum_aggregation | PASS, max\|diff\| 0, 537 周期 |
| aggr | var_aggregation | PASS, max\|diff\| 0, 1,858 周期 |
| aggr | variance_preserving_aggregation | PASS, max\|diff\| 0, 1,550 周期 |
| attention | performer_attention | PASS, max\|diff\| 0, 2,571 周期 |
| attention | polynormer_attention | PASS, max\|diff\| 0, 3,327 周期 |
| attention | qformer | PASS, max\|diff\| 0, 5,764 周期 |
| attention | sgformer_attention | PASS, max\|diff\| 0, 3,009 周期 |
| norm | batch_norm | PASS, max\|diff\| 0, 333 周期 |
| norm | diff_group_norm | PASS, max\|diff\| 0, 1,241 周期 |
| norm | graph_norm | PASS, max\|diff\| 0, 1,582 周期 |
| norm | graph_size_norm | PASS, max\|diff\| 0, 879 周期 |
| norm | hetero_batch_norm | PASS, max\|diff\| 0, 532 周期 |
| norm | hetero_layer_norm | PASS, max\|diff\| 0, 1,114 周期 |
| norm | instance_norm | PASS, max\|diff\| 0, 1,599 周期 |
| norm | layer_norm | PASS, max\|diff\| 0, 1,859 周期 |
| norm | layer_norm_node | PASS, max\|diff\| 0, 984 周期 |
| norm | mean_subtraction_norm | PASS, max\|diff\| 0, 1,024 周期 |
| norm | message_norm | PASS, max\|diff\| 0, 583 周期 |
| norm | pair_norm | PASS, max\|diff\| 0, 1,573 周期 |
| pool | asa_pooling | PASS, max\|diff\| 0, 8,309 周期 |
| pool | avg_pool | PASS, max\|diff\| 0, 941 周期 |
| pool | avg_pool_neighbor_x | PASS, max\|diff\| 0, 2,058 周期 |
| pool | avg_pool_x | PASS, max\|diff\| 0, 941 周期 |
| pool | cluster_pooling | PASS, max\|diff\| 0, 1,608 周期 |
| pool | edge_pooling | PASS, max\|diff\| 0, 3,432 周期 |
| pool | fps | PASS, max\|diff\| 0, 2,039 周期 |
| pool | global_add_pool | PASS, max\|diff\| 0, 420 周期 |
| pool | global_max_pool | PASS, max\|diff\| 0, 1,228 周期 |
| pool | global_mean_pool | PASS, max\|diff\| 0, 914 周期 |
| pool | graclus | PASS, max\|diff\| 0, 8,941 周期 |
| pool | knn | PASS, max\|diff\| 0, 1,028 周期 |
| pool | knn_graph | PASS, max\|diff\| 0, 1,252 周期 |
| pool | max_pool | PASS, max\|diff\| 0, 1,255 周期 |
| pool | max_pool_neighbor_x | PASS, max\|diff\| 0, 2,525 周期 |
| pool | max_pool_x | PASS, max\|diff\| 0, 1,255 周期 |
| pool | mem_pooling | PASS, max\|diff\| 0, 1,522 周期 |
| pool | nearest | PASS, max\|diff\| 0, 553 周期 |
| pool | pan_pooling | PASS, max\|diff\| 0, 2,025 周期 |
| pool | radius | PASS, max\|diff\| 0, 442 周期 |
| pool | radius_graph | PASS, max\|diff\| 0, 686 周期 |
| pool | sag_pooling | PASS, max\|diff\| 0, 3,459 周期 |
| pool | topk_pooling | PASS, max\|diff\| 0, 3,094 周期 |
| pool | voxel_grid | PASS, max\|diff\| 0, 1,672 周期 |
| unpool | knn_interpolate | PASS, max\|diff\| 0, 1,367 周期 |
| models | ar_link_predictor | PASS, max\|diff\| 0, 1,520 周期 |
| models | arga | PASS, max\|diff\| 0, 2,606 周期 |
| models | argva | PASS, max\|diff\| 0, 2,606 周期 |
| models | attentive_fp | PASS, max\|diff\| 0, 17,815 周期 |
| models | correct_and_smooth | PASS, max\|diff\| 0, 4,814 周期 |
| models | deep_gcn_layer | PASS, max\|diff\| 0, 4,651 周期 |
| models | deep_graph_infomax | PASS, max\|diff\| 0, 3,184 周期 |
| models | dimenet | PASS, max\|diff\| 0, 31,601 周期 |
| models | dimenet_plus_plus | PASS, max\|diff\| 0, 27,962 周期 |
| models | edge_cnn | PASS, max\|diff\| 0, 4,497 周期 |
| models | gae | PASS, max\|diff\| 0, 2,269 周期 |
| models | gat | PASS, max\|diff\| 0, 7,238 周期 |
| models | gcn | PASS, max\|diff\| 0, 2,557 周期 |
| models | gin | PASS, max\|diff\| 0, 2,203 周期 |
| models | gnnff | PASS, max\|diff\| 0, 26,340 周期 |
| models | gpse | PASS, max\|diff\| 0, 6,625 周期 |
| models | gpse_node_encoder | PASS, max\|diff\| 0, 1,244 周期 |
| models | graph_sage | PASS, max\|diff\| 0, 2,838 周期 |
| models | graph_unet | PASS, max\|diff\| 0, 6,445 周期 |
| models | group_add_rev | PASS, max\|diff\| 0, 2,767 周期 |
| models | hetero_jumping_knowledge | PASS, max\|diff\| 0, 12,399 周期 |
| models | inner_product_decoder | PASS, max\|diff\| 0, 404 周期 |
| models | jumping_knowledge | PASS, max\|diff\| 0, 8,355 周期 |
| models | jumping_knowledge_max | PASS, max\|diff\| 0, 475 周期 |
| models | label_propagation | PASS, max\|diff\| 0, 2,080 周期 |
| models | light_gcn | PASS, max\|diff\| 0, 2,257 周期 |
| models | linkx | PASS, max\|diff\| 0, 1,856 周期 |
| models | mask_label | PASS, max\|diff\| 0, 174 周期 |
| models | meta_layer | PASS, max\|diff\| 0, 2,932 周期 |
| models | metapath2vec | PASS, max\|diff\| 0, 85 周期 |
| models | mlp | PASS, max\|diff\| 0, 833 周期 |
| models | neural_fingerprint | PASS, max\|diff\| 0, 16,930 周期 |
| models | node2vec | PASS, max\|diff\| 0, 84 周期 |
| models | pmlp | PASS, max\|diff\| 0, 4,757 周期 |
| models | pna | PASS, max\|diff\| 0, 7,772 周期 |
| models | polynormer | PASS, max\|diff\| 0, 5,189 周期 |
| models | rect_l | PASS, max\|diff\| 0, 2,083 周期 |
| models | renet | PASS, max\|diff\| 0, 9,969 周期 |
| models | schnet | PASS, max\|diff\| 0, 5,123 周期 |
| models | sgformer | PASS, max\|diff\| 0, 8,865 周期 |
| models | signed_gcn | PASS, max\|diff\| 0, 5,339 周期 |
| models | tgn_memory | PASS, max\|diff\| 0, 325 周期 |
| models | vgae | PASS, max\|diff\| 0, 2,269 周期 |
| models | visnet | PASS, max\|diff\| 0, 15,156 周期 |
| kge | complex | PASS, max\|diff\| 0, 1,463 周期 |
| kge | distmult | PASS, max\|diff\| 0, 452 周期 |
| kge | rotate | PASS, max\|diff\| 0, 1,381 周期 |
| kge | transe | PASS, max\|diff\| 0, 1,191 周期 |
| encoding | positional_encoding | PASS, max\|diff\| 0, 420 周期 |
| encoding | temporal_encoding | PASS, max\|diff\| 0, 221 周期 |
| functional | bro | PASS, max\|diff\| 0, 745 周期 |
| functional | gini | PASS, max\|diff\| 0, 3,626 周期 |
| dense | dense_diff_pool | PASS, max\|diff\| 0, 6,899 周期 |
| dense | dense_gat_conv | PASS, max\|diff\| 0, 1,944 周期 |
| dense | dense_gcn_conv | PASS, max\|diff\| 0, 805 周期 |
| dense | dense_gin_conv | PASS, max\|diff\| 0, 832 周期 |
| dense | dense_graph_conv | PASS, max\|diff\| 0, 666 周期 |
| dense | dense_mincut_pool | PASS, max\|diff\| 0, 5,761 周期 |
| dense | dense_sage_conv | PASS, max\|diff\| 0, 852 周期 |
| dense | dmon_pooling | PASS, max\|diff\| 0, 7,037 周期 |

Not cases: `WLConv` (colours a Python hash of the neighbourhood multisets), `EquilibriumAggregation`
(an inner gradient descent through `torch.autograd.grad`, which torch.export does not trace), `LPFormer`
(torch.sparse tensors throughout: coalesce, sparse indices), `KNNIndex` / `L2KNNIndex` / `MIPSKNNIndex` /
`ApproxL2KNNIndex` / `ApproxMIPSKNNIndex` (faiss index wrappers, not a tensor computation), and the
page's last sections (`to_hetero` / `to_hetero_with_bases` model transformations, `DataParallel`, the
model hub mixin, `summary`), which are not computations.

## How PyG's data-dependent utilities export

torch.export needs static shapes and torch-mlir has no lowering for several of the operators PyG's
utilities reach for. `export_tg.py` binds static versions of the utilities in every loaded
`torch_geometric` module (`swap_loops`) and binds the genuine ones back to compute the reference
(`with_orig`), so every case is checked against the real PyG call on the same weights:

- `remove_self_loops` / `add_self_loops` / `add_remaining_self_loops` filter the edge list with a boolean
  mask (`edge_index[:, src != dst]`): a data-dependent shape torch.export keeps symbolic and the
  torch-mlir importer then crashes on. The case graph has no loops, so the static versions are the
  identity and the concatenation of the constant loop index.
- `scatter` / `degree` (every aggregation of every layer funnels through them) export as
  `tm_tensor.scatter`, which hwacha-mlir does not take: the static version is the one-hot matmul
  `onehot(index).T @ src` (sum, mean), a masked max / min over the broadcast (n, E, F) block, and
  exp(onehot.T @ log|x|) with the sign count for mul.
- `int(index.max()) + 1` (the default `dim_size` / `batch_size` / `num_classes` / `num_nodes` of a dozen
  helpers) is a data-dependent integer torch.export cannot resolve. Every index is a constant, so while
  exporting `Tensor.max` of an integer tensor returns the value recorded when the case was built
  (`record_maxes`, keyed by shape and dtype; the per-graph counts too).
- `to_dense_batch` (the set aggregations, MemPooling, SGFormer) fills a padded block with `out[idx] = x`
  (`tm_tensor.scatter`) after a `cumsum` (`tm_tensor.scan`): the static version is a one-hot placement
  matrix built with a triangular matmul.
- the pyg-lib / torch-cluster kernels (`knn`, `knn_graph`, `radius`, `radius_graph`, `fps`, `nearest`,
  `voxel_grid`, `graclus`, DimeNet's `triplets`) have no fake-tensor kernel; `tg_lib` computes them as
  tensor algorithms on the eager run and the results are recorded as buffers replayed in the export
  (`record_searches`: a tensor created inside the trace, however constant, gets an unbacked size). The
  eight `pool` cases whose input *is* the positions run the algorithm itself on Hwacha (pairwise
  distances, ranks, masks).
- torch-mlir's fx importer rejects a literal tensor inside an `aten.index` list (`x[idx]` with `idx` a
  buffer is constant-folded): `_import_list_argument` is patched to import it as a literal.
- `torch.atan2` (DimeNet's bond angles, PPFConv's point-pair features) calls libm's `atan2f`, which
  hwacha-cc has no vector version of: exported as `atan` of the ratio with the quadrant fixes; `ceil` as
  `-floor(-x)`; `torch.cdist` (`aten._cdist_forward`) as the explicit pairwise difference.
- `sort` / `topk` (`tm_tensor.sort`) and boolean-mask assignment (`x[mask] = ..`) have no lowering; the
  top-k selections (TopKPooling, SAGPooling, ASAPooling, PANPooling, GraphUNet, SortAggregation, the
  neighbour searches) are rank one-hots (`rank_j` = number of strictly greater scores, ties by index) and
  the masked assignments multiplies / `torch.where`.

**Per-layer rewrites** (the exported body differs from the PyG forward; each is asserted equal to the
genuine call before export):

- ChebConv: the scaled Laplacian as a dense matrix and the Chebyshev recursion (`get_laplacian` masks).
- ClusterGCNConv: `edge_weight[row == col] +=` is the diagonal of the dense normalised adjacency.
- RGCNConv / FastRGCNConv / FiLMConv: the per-relation edge lists (`edge_index[:, edge_type == r]`) are
  constants; the relation sums are dense mean-adjacency matmuls (RGCN) or per-relation FiLM messages.
- HypergraphConv: `D^-1 H B^-1 H^T X Theta` with the incidence matrix (`D[D == inf] = 0`).
- GPSConv, SGFormer, MemPooling, bro: `to_dense_batch` on 2 graphs of 4 sorted nodes is a reshape.
- PANConv / PANPooling: the path-integral matrix `sum_i (prod_{j<=i} w_j) A^i` and its MET normalisation
  densely (torch_sparse); PANPooling's SelectTopK as a rank one-hot.
- SplineConv: torch_spline_conv's degree-1 B-spline basis of the constant pseudo-coordinates (`tg_lib
  .spline_basis`, checked against the C++ kernel) and the weighted per-basis matmuls.
- DynamicEdgeConv / XConv: the k-nn of the input features / positions as constant neighbour lists
  (pyg's `knn` includes the point itself), then the module's own message / transform code.
- MFConv (also inside NeuralFingerprint): the per-degree linears picked with `nonzero` / `index_copy_`
  become every degree's linear evaluated and selected by the one-hot of the degree.
- DenseGCNConv / DenseGATConv: `adj[:, idx, idx] = 1` becomes the adjacency with the loops set,
  `add_loop=False`; dense_mincut_pool / DMoNPooling: `out_adj[:, ind, ind] = 0` is a `(1 - I)` multiply.
- MultiAggregation / DegreeScalerAggregation: PyG fuses sum / mean / max / .. into `FusedAggregation`
  (a `scatter_add_` count); unfused, each goes through the static scatter.
- Set2Set: the LSTM state re-shaped between steps (torch.export otherwise hands it back 4-D).
- SortAggregation: the per-graph descending sort by the last feature as a rank one-hot.
- SetTransformerAggregation / GraphMultisetTransformer: the key-padding mask of `nn.MultiheadAttention`
  as an additive float mask, `out[~mask] = 0` as a multiply; a query with no keys (node 7 has no
  in-edges) is zeroed, as PyG's `nan_to_num` does.
- TopKPooling / SAGPooling / ASAPooling: SelectTopK's score `tanh(x . w / |w|)`, the top 2 of each
  graph as a rank one-hot, `x[perm] * score[perm]`; ASAP's attention / LEConv fitness before it.
- EdgePooling / ClusterPooling: the edge contraction is a greedy matching (resp. connected components
  via scipy) in Python over the sorted scores, data dependent by nature; the cluster matrix of the
  case's scores is precomputed from the module's own eager run, the export computes the scores and the
  pooled sum with it (EdgePooling's clusters scaled by their contracted edge's score).
- GraphUNet: depth 1 on the dense adjacency, `A2 = (A + I)^2` without its diagonal, the TopKPooling and
  unpooling as the same one-hot `P` and `P^T`.
- graclus: torch_cluster's kernel visits the nodes in a random permutation (two calls give different
  matchings), so the case is the greedy heavy-edge matching in index order as a tensor loop.
- LabelPropagation / MaskLabel / CorrectAndSmooth: `out[mask] = y[mask]`, `x[mask] += emb(y[mask])`,
  `error[mask] = ..`, `scale[isinf | > 1000] = 1` as one-hot / mask products and `torch.where`.
- ViSNet: the loop edges' `edge_weight[mask] = |vec|[mask]` / `edge_vec[mask] /= |vec|` (zero vectors)
  as unmasked norms with a clamp, NeighborEmbedding's loop messages zeroed through `W`.
- SGFormerAttention: `qs[qs == 0] = eps` as `torch.where`.
- KGE models (TransE, ComplEx, DistMult, RotatE) score 4 constant (head, relation, tail) triples;
  Node2Vec / MetaPath2Vec / LightGCN / TGNMemory / RENet are their embedding lookups (the random walks
  and message stores are training-time machinery).

The torch_scatter / torch_sparse / torch_cluster / torch_spline_conv / pyg_lib wheels (pt29cu128,
data.pyg.org) and scipy are installed in `../.tmenv` for the references only.
