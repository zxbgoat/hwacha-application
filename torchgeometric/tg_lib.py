"""Tensor re-implementations for the torch_geometric.nn cases whose PyG forward does not export: data-
dependent shapes (boolean edge masks per relation, hashes, int(tensor.max())), sparse tensors, or the
pyg-lib / torch-cluster C++ kernels (knn, spline basis). Every function here is checked against the
genuine PyG call in export_tg.py (the reference)."""
import math, torch, torch.nn.functional as NF

def cdist(a, b):
    """pairwise Euclidean distances (a (M, D), b (N, D)) -> (M, N); aten._cdist_forward has no lowering"""
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1).clamp(min=0).sqrt()

def adj(ei, N, w=None):
    """dense adjacency A[dst, src] = w (or 1) of an edge list (2, E); duplicate edges accumulate"""
    A = torch.zeros(N, N, dtype=torch.float32 if w is None else w.dtype)
    return A.index_put((ei[1], ei[0]), torch.ones(ei.shape[1]) if w is None else w, accumulate=True)

def onehot(idx, n):
    """(E,) int -> (E, n) float one-hot: the static gather / scatter matrix"""
    return (idx[:, None] == torch.arange(n)[None, :]).to(torch.float32)

def scatter_sum(src, index, n):
    """sum src (E, F) into n rows by index: onehot(index).T @ src"""
    return onehot(index, n).t() @ src

def scatter_mean(src, index, n):
    o = onehot(index, n); cnt = o.sum(0).clamp(min=1)
    return (o.t() @ src) / cnt[:, None]

def scatter_max(src, index, n):
    """max over each group, 0 for empty groups (PyG's scatter 'max')"""
    o = onehot(index, n)                                        # (E, n)
    m = torch.where(o.t()[:, :, None] > 0, src[None], torch.full_like(src[None].expand(n, -1, -1), -1e30)).amax(1)
    return torch.where(o.sum(0)[:, None] > 0, m, torch.zeros_like(m))

def scatter_min(src, index, n):
    return -scatter_max(-src, index, n)

def scatter_softmax(src, index, n):
    """softmax of src (E, H) over the edges that share index (PyG's utils.softmax)"""
    o = onehot(index, n)                                        # (E, n)
    mx = scatter_max(src, index, n)                             # (n, H)
    e = torch.exp(src - o @ mx)
    den = o @ (o.t() @ e)
    return e / den.clamp(min=1e-16)

def knn_idx(pos, k, loop=False):
    """(N, k) indices of the k nearest neighbours by Euclidean distance (torch_cluster.knn order: by
    distance, ties by index); a constant when pos is"""
    d = cdist(pos, pos)
    if not loop: d = d + torch.eye(pos.shape[0]) * 1e9
    return d.topk(k, largest=False).indices

def cheb_norm(ei, N, ew=None, lambda_max=2.0):
    """ChebConv's scaled Laplacian L~ = 2 L / lambda_max - I with sym normalisation, as a dense matrix"""
    A = adj(ei, N, ew); deg = A.sum(0); dis = deg.pow(-0.5); dis[dis == float('inf')] = 0   # deg over the source index, as get_laplacian
    Ln = torch.eye(N) - dis[:, None] * A * dis[None, :]
    return 2.0 * Ln / lambda_max - torch.eye(N)

def spline_basis(pseudo, kernel_size, degree=1):
    """torch_spline_conv's B-spline basis (open splines, degree 1) for pseudo (E, D) in [0, 1]:
    returns (basis (E, K), weight_index (E, K)) with K = 2**D over the kernel_size**D weights"""
    E, D = pseudo.shape; S = kernel_size
    v = pseudo * (S - 1); frac = v - v.floor(); lo = v.floor().clamp(max=S - 2)
    K = 2 ** D; basis = torch.ones(E, K); widx = torch.zeros(E, K, dtype=torch.long)
    for k in range(K):
        stride = 1
        for d in range(D):
            bit = (k >> d) & 1
            f = frac[:, d]; i = lo[:, d]
            basis[:, k] = basis[:, k] * (f if bit else 1 - f)
            widx[:, k] = widx[:, k] + (i.long() + bit) * stride
            stride *= S
    return basis, widx

# ---- the pyg-lib / torch-cluster kernels (knn, radius, fps, nearest, grid_cluster, graclus) as tensor
# algorithms on the fixed-size inputs. The neighbour searches return the same edge sets as pyg's
# (padded to a static count where the count is data dependent: radius / radius_graph give up to
# max_num_neighbors per query; the export pins the actual count of the case data as a constant).

def knn_pairs(x, y, k, batch_x=None, batch_y=None):
    """torch_geometric.nn.knn(x, y, k): for each row of y its k nearest rows of x (of the same graph when batch
    vectors are given) -> (2, M*k) [y index, x index] (pyg's order: by query, then by distance; ties by index)"""
    d = cdist(y, x)
    if batch_x is not None: d = d + (batch_y[:, None] != batch_x[None, :]).float() * 1e9
    idx = d.topk(k, largest=False).indices                                  # (M, k)
    return torch.stack([torch.arange(y.shape[0]).repeat_interleave(k), idx.reshape(-1)])

def knn_graph(x, k, loop=False, batch=None):
    """torch_geometric.nn.knn_graph(x, k) (flow source_to_target): (2, N*k) [neighbour (source), query (target)]"""
    d = cdist(x, x)
    if not loop: d = d + torch.eye(x.shape[0]) * 1e9
    if batch is not None: d = d + (batch[:, None] != batch[None, :]).float() * 1e9
    idx = d.topk(k, largest=False).indices
    return torch.stack([idx.reshape(-1), torch.arange(x.shape[0]).repeat_interleave(k)])

def radius_pairs(x, y, r, loop=True, batch_x=None, batch_y=None):
    """torch_geometric.nn.radius(x, y, r): all (y index, x index) pairs within r (within the same graph when
    batch vectors are given), ordered by y then x index"""
    d = cdist(y, x); m = d <= r
    if not loop: m = m & ~torch.eye(y.shape[0], x.shape[0], dtype=torch.bool)
    if batch_x is not None: m = m & (batch_y[:, None] == batch_x[None, :])
    return m.nonzero().t()

def radius_graph(x, r):
    """torch_geometric.nn.radius_graph(x, r) (no loops, source_to_target): (2, E) [source, target] with the target-major order of pyg"""
    p = radius_pairs(x, x, r, loop=False)
    return torch.stack([p[1], p[0]])

def fps(x, ratio=0.5):
    """farthest point sampling from index 0 (random_start=False): ceil(ratio*N) indices"""
    N = x.shape[0]; n = int(math.ceil(ratio * N)); sel = [0]
    dist = cdist(x, x)[0].clone()
    for _ in range(n - 1):
        j = int(dist.argmax()); sel.append(j); dist = torch.minimum(dist, cdist(x, x)[j])
    return torch.tensor(sel)

def nearest(x, y):
    """index of the nearest row of y for each row of x"""
    return cdist(x, y).argmin(1)

def voxel_grid(pos, size):
    """torch_geometric.nn.voxel_grid(pos, size) with start = min, end = max (the defaults): cluster id of each point"""
    start = pos.min(0).values; end = pos.max(0).values
    n = ((end - start) / size).floor().long() + 1
    c = ((pos - start) / size).floor().long()
    stride = torch.stack([torch.ones_like(n[0]), n[0], n[0] * n[1]])           # 3-D strides (cumprod has no lowering)
    return (c * stride).sum(1)

def graclus(ei, w, N):
    """graclus_cluster: greedy heavy-edge matching, nodes in index order, each unmatched node picks its
    unmatched neighbour of largest normalised weight w / (deg_i deg_j) (torch_cluster's order); the
    cluster id is the smaller index of the pair"""
    A = adj(ei, N, w); deg = A.sum(1)
    cl = torch.full((N,), -1, dtype=torch.long)
    for i in range(N):
        if cl[i] >= 0: continue
        best, bj = -1.0, -1
        for j in range(N):
            if j == i or cl[j] >= 0 or A[i, j] == 0: continue
            v = float(A[i, j] / (deg[i] * deg[j]))
            if v > best: best, bj = v, j
        cl[i] = i
        if bj >= 0: cl[bj] = i
    return cl

def knn_mask(x, y, k, loop=True):
    """(M, N) 0/1 mask of the k nearest rows of x for each row of y (rank by distance, ties by index)"""
    d = cdist(y, x)
    if not loop: d = d + torch.eye(y.shape[0], x.shape[0]) * 1e9
    ar_x = torch.arange(x.shape[0])
    lt = (d[:, None, :] < d[:, :, None]) | ((d[:, None, :] == d[:, :, None]) & (ar_x[None, None, :] < ar_x[None, :, None]))   # [m, j, i]: x_i ranks before x_j for query m
    rank = lt.float().sum(-1)                                                   # (M, N)
    return (rank < k).float()

def fps_t(x, n):
    """farthest point sampling as a tensor loop (start 0, n points): the argmax of the running min
    distance, the chosen row of D gathered with a one-hot (no data-dependent indexing)"""
    D = cdist(x, x); N = x.shape[0]; sel = torch.zeros(1, dtype=torch.long); dist = D[0]
    for _ in range(n - 1):
        oh = (dist == dist.max()).float(); oh = oh * (torch.cumsum(oh, 0) == 1).float() if False else oh   # (ties impossible in random data)
        j = (oh * torch.arange(N).float()).sum().long(); sel = torch.cat([sel, j.view(1)])
        dist = torch.minimum(dist, oh @ D)
    return sel

def radius_mask(x, y, r, loop=True):
    """(M, N) 0/1 mask of the (y, x) pairs within r: the radius search with a static shape"""
    m = cdist(y, x) <= r
    if not loop: m = m & ~torch.eye(y.shape[0], x.shape[0], dtype=torch.bool)
    return m

def pair_mask(pairs, M, N):
    """(2, P) [y index, x index] pairs -> the (M, N) 0/1 mask"""
    return torch.zeros(M, N).index_put((pairs[0], pairs[1]), torch.ones(pairs.shape[1]))

def graclus_t(A, N):
    """graclus as a tensor computation: nodes in index order, each unmatched node i is matched with its
    unmatched neighbour of largest A_ij / (deg_i deg_j); cluster id = the smaller index"""
    deg = A.sum(1); W = A / (deg[:, None] * deg[None, :]).clamp(min=1e-30)
    free = torch.ones(N); cl = torch.arange(N).float()
    for i in range(N):
        cand = W[i] * free * (torch.arange(N) != i).float()
        j = cand.argmax(); has = (cand.max() > 0).float() * free[i]
        onehot_j = (torch.arange(N) == j).float() * has
        cl = cl * (1 - onehot_j) + onehot_j * i
        free = free * (1 - onehot_j) * (1 - (torch.arange(N) == i).float())
    return cl

def topk_perm_static(score, k):
    """the k largest of score (N,), in descending order, as a (k, N) one-hot selection matrix (SelectTopK with
    ratio k/N on one graph): rank_j = number of strictly greater scores (random data, no ties)"""
    rank = (score[None, :] < score[:, None]).to(score.dtype).sum(0)          # rank[j] = #{m: s_m > s_j}
    return (rank[None, :] == torch.arange(k, dtype=score.dtype)[:, None]).to(score.dtype)   # (k, N): row i picks rank i

def topk_perm_batched(score, batch, k_per_graph):
    """the k largest per graph (batch (N,) sorted, k_per_graph a list): rows ordered graph by graph, as a
    (sum k, N) one-hot; ranks are taken within the graph"""
    same = (batch[:, None] == batch[None, :]).to(score.dtype); N = score.shape[0]
    ar = torch.arange(N)
    gt = (score[None, :] < score[:, None]) | ((score[None, :] == score[:, None]) & (ar[:, None] < ar[None, :]))   # ties: the lower index first (torch.sort's stable order)
    rank = (gt.to(score.dtype) * same).sum(0)
    rows = []
    for g, kg in enumerate(k_per_graph):
        ing = (batch == g).to(score.dtype)
        for i in range(kg): rows.append((rank == i).to(score.dtype) * ing)
    return torch.stack(rows)

def quantile_static(x, index, n, q, interpolation='linear', fill_value=0.0):
    """QuantileAggregation on x (E, F) grouped by index into n groups, for one q: per group and feature the
    sorted values are picked by rank (rank = #smaller + #equal-with-lower-position, no ties in random data)"""
    E, Fd = x.shape; o = onehot(index, n)                                      # (E, n)
    same = o @ o.t()                                                            # (E, E) same group
    lt = (x[None, :, :] < x[:, None, :]).to(x.dtype)                            # lt[j, i, f] = x_i < x_j? no: [a, b, f] = x_b < x_a
    rank = (lt * same[:, :, None]).sum(1)                                       # (E, F): #smaller in the group
    count = o.sum(0)                                                            # (n,)
    qp = q * (count - 1)                                                        # position within the group
    lo = qp.floor(); hi = -(-qp).floor(); frac = qp - lo                        # ceil as -floor(-x): llvm.ceil has no Hwacha lowering
    # value at rank r in group g: sum_e x_e [index_e = g][rank_e = r]
    def at(pos):
        target = o @ pos                                                        # (E,) the wanted rank for each element's group
        sel = (rank == target[:, None]).to(x.dtype)                             # (E, F)
        return o.t() @ (sel * x)                                                # (n, F)
    if interpolation == 'lower': v = at(lo)
    elif interpolation == 'higher': v = at(hi)
    elif interpolation == 'nearest': v = at((qp + 0.5).floor())
    elif interpolation == 'linear': v_lo = at(lo); v = v_lo + (at(hi) - v_lo) * frac[:, None]
    else: v = 0.5 * (at(lo) + at(hi))
    return torch.where(count[:, None] > 0, v, torch.full_like(v, fill_value))


def gcn_dense(conv, x, A, improved=False):
    """GCNConv on a dense weighted adjacency A[dst, src]: A~ = A + fill I (fill 2 if improved; the diagonal of A
    is first zeroed, as add_remaining_self_loops replaces existing loops), D^-1/2 A~ D^-1/2 (D over the source
    axis), out = norm @ (x W) + b"""
    N = A.shape[0]; fill = 2.0 if improved else 1.0
    At = A * (1 - torch.eye(N)) + fill * torch.eye(N)
    deg = At.sum(0); dis = deg.pow(-0.5); dis = torch.where(torch.isinf(dis), torch.zeros_like(dis), dis)
    norm = dis[:, None] * At * dis[None, :]
    out = norm @ conv.lin(x)
    return out + conv.bias if conv.bias is not None else out
