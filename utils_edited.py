import os
from pathlib import Path
from data_protocol import read_rankings, graph_fingerprint
import pandas as pd
import torch
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from variable import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import ot
from time import time
import torch.nn.functional as F



def prepare(tensor: torch.Tensor, number, device, max_iter=100, random_state: int = 999):
    # Handle the zero-cluster case
    if number <= 0:
        return None, torch.zeros((0, tensor.size(1))).to(device)
    # 1) convert to numpy for sklearn
    data = tensor.cpu().numpy()
    # 2) cluster into k groups
    kmeans = KMeans(n_clusters=number, max_iter=max_iter, random_state=random_state).fit(data)
    # labels = kmeans.fit_predict(data)       # array of length s1
    centroids = kmeans.cluster_centers_     # shape (k, s2)
    centroids = (torch.tensor(centroids)).to(device)
    return kmeans, centroids

def assign_prototype(kmeans, data, device):
    if kmeans is None:
        # Return a zero tensor of the same batch size and embedding dim
        return torch.zeros_like(data).to(device), None
    data = data.cpu().numpy()
    labels = kmeans.predict(data)       # array of length s1
    predict = kmeans.cluster_centers_[labels]     # shape (k, s2)
    return (torch.tensor(predict)).to(device), (torch.tensor(labels)).to(device)


def draw_train(train_losses, val_losses):
    iters = [iter * ITERS_PER_EVAL for iter in range(len(train_losses))]
    valid_iters = [iter * ITERS_PER_EVAL for iter in range(len(val_losses))]
    plt.plot(iters, train_losses, label='train')
    plt.plot(valid_iters, torch.Tensor.cpu(torch.tensor(val_losses)), label='validation')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('training and validation loss curves')
    plt.legend()
    plt.show()


def overlap(overlap_path):
    """
    USAGE:
    for s_uid, t_uid in self.overlap_src2tgt.items():
        self.user_emb_tgt.weight.data[t_uid] = self.user_emb_src.weight.data[s_uid]
    """
    # Build mapping from source_uid -> target_uid for overlapping users
    overlap_src2tgt = {}
    with open(overlap_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad format in overlap file: {line}")
            src_uid = int(parts[0])
            tgt_uid = int(parts[1])
            overlap_src2tgt[src_uid] = tgt_uid
    return overlap_src2tgt


def WDLOSS(xs, xt, lambda_e=0.01, numItermax=100, device='cpu', cost_bias=None):
    """Correspondence-guided entropic OT cost and its balanced transport plan."""
    warnings.filterwarnings("ignore", category=UserWarning, module="ot")
    a = torch.as_tensor(ot.utils.unif(xs.size(0)), dtype=xs.dtype, device=device)
    b = torch.as_tensor(ot.utils.unif(xt.size(0)), dtype=xt.dtype, device=device)
    M = ot.dist(xs, xt, metric='cosine').to(device)
    if cost_bias is not None:
        M = M + cost_bias
    plan = ot.sinkhorn(a, b, M, reg=lambda_e, method='sinkhorn_log', numItermax=numItermax)
    WD = torch.sum(plan * M)
    return WD.to(device), plan.to(device)

def UniformSample_pretrain(allPos, n_users, m_items, trainDataSize):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    users = np.random.randint(0, n_users, trainDataSize)
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)

def UniformSample_cdr(cdr_dataset, n_item):
    S = []
    for user, pos_items in cdr_dataset.trainAllPos.items():
        exclude = set(pos_items)
        if len(exclude) >= n_item:
            raise ValueError(f"User {user} has no unobserved training negatives")
        for pos_item in pos_items:
            while True:
                neg_item = np.random.randint(0, n_item)
                if neg_item not in exclude:
                    break
            S.append([user, pos_item, neg_item])
    return np.array(S)

# ===================end samplers==========================
# =====================utils====================================
def pretrain_minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', BATCH_SIZE)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


# ====================Metrics==============================
# =========================================================
def HR_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    hits = (right_pred > 0).astype(float)
    return np.sum(hits)  # caller divides by total users, same convention as RecallPrecision_ATk


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================

# ====================Pretrain Train/Test Dataloading (Loader)==========================
# ===============================================================================
class Loader:
    """
    Ranking-format dataset loader for LightGCN-style pretraining/testing.

    Expects `path` to contain:
        - train.txt  (required): one line per user, "user item1 item2 item3 ..."
        - test.txt   (optional): same format, held-out interactions.
                                   If absent, testDataSize is 0 and evaluation
                                   is skipped by callers (e.g. pretrain.py).

    Builds the user-item bipartite graph used by LightGCN's message passing,
    and caches the normalized adjacency matrix to `path/s_pre_adj_mat.npz`
    so it isn't recomputed on every run.
    """

    def __init__(self, path, device):
        self.path = path
        self.device = device
        self.n_user = 0
        self.m_item = 0

        train_file = os.path.join(path, 'train.txt')
        test_file = os.path.join(path, 'test.txt')

        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        if os.path.exists(test_file):
            with open(test_file) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip('\n').split(' ')
                        items = [int(i) for i in l[1:] if i]
                        uid = int(l[0])
                        if items:
                            testUniqueUsers.append(uid)
                            testUser.extend([uid] * len(items))
                            testItem.extend(items)
                            self.m_item = max(self.m_item, max(items))
                            self.n_user = max(self.n_user, uid)
                            self.testDataSize += len(items)
        else:
            print(f"No test file found at {test_file}, proceeding with train-only data")

        self.validDict = read_rankings(Path(path) / 'valid.txt')
        for filename, attribute in [('user-map.txt', 'n_user'), ('item-map.txt', 'm_item')]:
            with (Path(path) / filename).open(encoding='utf-8') as stream:
                indices = [int(line.split()[1]) for line in stream if line.strip()]
            setattr(self, attribute, max(getattr(self, attribute), max(indices)))
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")

        # (users, items), bipartite graph
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item)
        )
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1.
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
            A_fold = []
            self.folds = 100
            fold_len = (self.n_user + self.m_item) // self.folds
            for i_fold in range(self.folds):
                start = i_fold*fold_len
                if i_fold == self.folds - 1:
                    end = self.n_user + self.m_item
                else:
                    end = (i_fold + 1) * fold_len
                A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.device))
            return A_fold

    def getSparseGraph(self, device=None):
        """
        Builds (or loads a cached) symmetrically-normalized user-item
        adjacency matrix and returns it as a coalesced sparse tensor on `device`.
        """
        device = device if device is not None else self.device
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(os.path.join(self.path, 's_pre_adj_mat_' + graph_fingerprint(self.path) + '.npz'))
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except (FileNotFoundError, OSError):
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix(
                    (self.n_user + self.m_item, self.n_user + self.m_item), dtype=np.float32
                )
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_user, self.n_user:] = R
                adj_mat[self.n_user:, :self.n_user] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(os.path.join(self.path, 's_pre_adj_mat_' + graph_fingerprint(self.path) + '.npz'), norm_adj)
            
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(device)
            print("don't split the matrix")
        return self.Graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users: shape [-1]
        items: shape [-1]
        return: feedback [-1]
        """
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems


class CDRLoader:
    def __init__(self, path):
        self.path = path
        self.trainAllPos, self.trainDataSize = self._read_split(os.path.join(path, 'train.txt'))
        self.validDict, self.validDataSize = self._read_split(os.path.join(path, 'valid.txt'))
        self.testDict, self.testDataSize = self._read_split(os.path.join(path, 'test.txt'))
        print(f"{self.trainDataSize} CDR-train / {self.validDataSize} CDR-valid / "
              f"{self.testDataSize} CDR-test interactions across {len(self.trainAllPos)} overlap users")

    @staticmethod
    def _read_split(file_path):
        pos_map, size = {}, 0
        with open(file_path) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:] if i]
                    uid = int(l[0])
                    if items:
                        pos_map[uid] = items
                        size += len(items)
        return pos_map, size

    def getUserPosItems(self, user):
        """Complete observed history for descriptive analysis, not training or masking."""
        return (set(self.trainAllPos.get(user, []))
                | set(self.validDict.get(user, []))
                | set(self.testDict.get(user, [])))

# ==================== POTC Analysis Experiments ====================
# These functions are standalone analysis utilities. They do not modify the
# existing training/evaluation path. Call them from a separate notebook after
# training a selected POTC configuration.
# ===================================================================

def _model_attr(model, *names, required=True):
    """Return the first existing model attribute among *names*."""
    for name in names:
        value = getattr(model, name, None)
        if value is not None:
            return value
    if required:
        raise AttributeError(f"Could not locate any of {names} on the trained POTC model.")
    return None


@torch.no_grad()
def _analysis_components(model):
    """Collect the frozen embeddings/prototypes/mapping parameters from either POTC interface."""
    source_encoder = _model_attr(model, 'sencoder', 'source_encoder')
    target_encoder = _model_attr(model, 'tencoder', 'target_encoder')
    source_prototypes = _model_attr(model, 'sp', 'source_prototypes')
    target_prototypes = _model_attr(model, 'tp', 'target_prototypes')
    projection = _model_attr(model, 'cluster_map', 'projection')
    overlap_src2tgt = _model_attr(model, 'overlap_src2tgt', 'overlap_src_to_tgt')

    rho = float(_model_attr(model, 'rho'))
    skip_ot = bool(_model_attr(model, 'skip_ot', required=False) or False)
    epsilon = float(_model_attr(model, 'lambda_e', 'epsilon', required=False) or LAMBDA_E)
    iterations = int(_model_attr(model, 'maxiter', 'sinkhorn_iterations', required=False) or MAXITER)

    # The actual POTC encoders in this project expose `computer()` rather than
    # a zero-argument `forward()`. Keep the helper compatible with that API.
    source_users, _ = source_encoder.computer()
    target_users, _ = target_encoder.computer()

    return {
        'source_encoder': source_encoder,
        'target_encoder': target_encoder,
        'source_prototypes': source_prototypes,
        'target_prototypes': target_prototypes,
        'projection': projection,
        'overlap_src2tgt': overlap_src2tgt,
        'source_users': source_users,
        'target_users': target_users,
        'rho': rho,
        'skip_ot': skip_ot,
        'epsilon': epsilon,
        'iterations': iterations,
    }


@torch.no_grad()
def potc_analysis_association_matrix(
    source_encoder,
    target_encoder,
    source_prototypes,
    target_prototypes,
    overlap_src2tgt,
):
    """Recompute the normalized soft overlap association matrix A exactly as in POTC."""
    # Actual project encoder API: LightGCN.computer().
    source_users, _ = source_encoder.computer()
    target_users, _ = target_encoder.computer()

    src_ids = []
    tgt_ids = []
    for src_id, tgt_id in overlap_src2tgt.items():
        src_ids.append(int(src_id))
        tgt_ids.append(int(tgt_id))

    if not src_ids:
        raise ValueError('The overlap mapping is empty; POTC needs overlapping users.')

    src_ids = torch.as_tensor(src_ids, dtype=torch.long, device=source_users.device)
    tgt_ids = torch.as_tensor(tgt_ids, dtype=torch.long, device=target_users.device)

    source_overlap = source_users[src_ids]
    target_overlap = target_users[tgt_ids]

    ps = F.softmax(source_overlap @ source_prototypes.T, dim=1)
    pt = F.softmax(target_overlap @ target_prototypes.T, dim=1)
    association = ps.T @ pt
    row_max = association.amax(dim=1, keepdim=True)
    return association / row_max.clamp_min(1e-12)


@torch.no_grad()
def potc_analysis_hard_overlap_matrix(source_users, target_users,
                                       source_prototypes, target_prototypes,
                                       overlap_src2tgt):
    """
    Build the empirical hard overlap matrix from the actual paired overlap users.

    H[k_s, k_t] = number of overlapping users whose nearest source prototype is
    k_s and nearest target prototype is k_t.

    This is an identity-based empirical correspondence matrix. It is better
    labeled "Hard Overlap Matrix" rather than "ground truth" because prototype
    assignments themselves are induced by K-Means, not externally annotated.
    """
    src_ids = torch.as_tensor(
        [int(x) for x in overlap_src2tgt.keys()],
        dtype=torch.long,
        device=source_users.device,
    )
    tgt_ids = torch.as_tensor(
        [int(x) for x in overlap_src2tgt.values()],
        dtype=torch.long,
        device=target_users.device,
    )
    if src_ids.numel() == 0:
        raise ValueError('The overlap mapping is empty.')

    src_overlap = source_users[src_ids]
    tgt_overlap = target_users[tgt_ids]

    src_dist = torch.cdist(src_overlap, source_prototypes, p=2.0)
    tgt_dist = torch.cdist(tgt_overlap, target_prototypes, p=2.0)
    src_labels = torch.argmin(src_dist, dim=1)
    tgt_labels = torch.argmin(tgt_dist, dim=1)

    k = source_prototypes.shape[0]
    hard_counts = torch.zeros((k, k), dtype=torch.float64, device=source_users.device)
    hard_counts.index_put_((src_labels, tgt_labels),
                           torch.ones_like(src_labels, dtype=torch.float64),
                           accumulate=True)

    hard_norm = hard_counts / hard_counts.max().clamp_min(1.0)
    return hard_counts, hard_norm


@torch.no_grad()
def potc_analysis_transport_matrix(model):
    """
    Compute exactly ONE transport matrix for the CURRENT trained run.

    If rho > 0, this is the association-guided transport used by that run:
        T = Sinkhorn(M + rho * (1 - A)).

    If rho == 0, this is the geometry-only transport used by that run:
        T = Sinkhorn(M).

    This function intentionally does NOT compute an alternative transport
    under a different rho, because that would mix two differently trained
    models and would not be a clean within-run visualization.
    """
    c = _analysis_components(model)
    if c['skip_ot']:
        raise ValueError('This run used skip_ot=True and has no OT transport plan to visualize.')
    projected_source = c['projection'](c['source_prototypes'])
    source_n = F.normalize(projected_source, p=2, dim=1)
    target_n = F.normalize(c['target_prototypes'], p=2, dim=1)
    cosine_cost = 1.0 - source_n @ target_n.T

    association = potc_analysis_association_matrix(
        source_encoder=c['source_encoder'],
        target_encoder=c['target_encoder'],
        source_prototypes=c['source_prototypes'],
        target_prototypes=c['target_prototypes'],
        overlap_src2tgt=c['overlap_src2tgt'],
    )

    enhanced_cost = cosine_cost + c['rho'] * (1.0 - association)
    transport = _analysis_sinkhorn_uniform(
        enhanced_cost,
        epsilon=c['epsilon'],
        iterations=c['iterations'],
    )

    hard_counts, hard_norm = potc_analysis_hard_overlap_matrix(
        c['source_users'], c['target_users'],
        c['source_prototypes'], c['target_prototypes'],
        c['overlap_src2tgt'],
    )

    return {
        'association': association,
        'hard_overlap_counts': hard_counts,
        'hard_overlap': hard_norm,
        'cosine_cost': cosine_cost,
        'enhanced_cost': enhanced_cost,
        'transport': transport,
        'projected_source_prototypes': projected_source,
        'rho': c['rho'],
        'epsilon': c['epsilon'],
        'iterations': c['iterations'],
    }


def _analysis_sinkhorn_uniform(cost, epsilon, iterations):
    """Log-domain Sinkhorn with uniform KxK marginals, matching POTC."""
    if epsilon <= 0:
        raise ValueError('epsilon must be > 0.')
    k = cost.shape[0]
    log_mass = -float(np.log(k))
    log_kernel = -cost / float(epsilon)

    log_u = torch.zeros(k, device=cost.device, dtype=cost.dtype)
    log_v = torch.zeros(k, device=cost.device, dtype=cost.dtype)
    for _ in range(int(iterations)):
        log_u = log_mass - torch.logsumexp(
            log_kernel + log_v.unsqueeze(0), dim=1
        )
        log_v = log_mass - torch.logsumexp(
            log_kernel + log_u.unsqueeze(1), dim=0
        )

    return torch.exp(
        log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    )


def _rho_tag(rho):
    """Filesystem-safe rho tag."""
    return str(rho).replace('-', 'm').replace('.', 'p')


def _model_run_tag(model, source_domain, target_domain):
    """Separate output directories for OT and association-only runs."""
    rho = _model_attr(model, 'rho')
    skip_ot = bool(_model_attr(model, 'skip_ot', required=False) or False)
    mode = 'softmax' if skip_ot else 'ot'
    return f"{source_domain}-{target_domain}-rho{_rho_tag(rho)}-{mode}"


def save_potc_transport_analysis_svg(
    model,
    filename=None,
    save_npz=False,
    max_ticks=6,
):
    """
    Experiment 1 visualization for ONE trained run.

    Three heatmaps are produced:
      1) soft association matrix A
      2) hard identity-based overlap matrix H
      3) the actual learned transport plan T of this run

    No second hypothetical transport matrix is computed.
    """
    result = potc_analysis_transport_matrix(model)
    A = result['association'].detach().cpu().numpy()
    H = result['hard_overlap'].detach().cpu().numpy()
    T = result['transport'].detach().cpu().numpy()
    H_counts = result['hard_overlap_counts'].detach().cpu().numpy()
    K = A.shape[0]

    # T has row/column marginals approximately 1/K. K*T makes the heatmap
    # easier to interpret without changing relative transport structure.
    T_vis = K * T

    if filename is None:
        filename = f"potc_transport_analysis_rho_{_rho_tag(result['rho'])}.svg"
    output_path = os.path.join(visualize_path, filename)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 5.2),
        constrained_layout=True,
    )
    matrices = [
        (A, r'Association matrix $\tilde{\mathcal{A}}$', 'Target prototype'),
        (H, 'Hard overlap matrix $H$', 'Target prototype'),
        (T_vis, r'Learned transport $KT$', 'Target prototype'),
    ]

    for ax, (matrix, title, xlabel) in zip(axes, matrices):
        im = ax.imshow(matrix, aspect='auto', interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Source prototype')
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks, integer=True))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks, integer=True))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f'POTC Prototype Correspondence and Learned Transport '
        f'(K={K}, $\\rho$={result["rho"]:.4g})',
        fontsize=13,
    )
    fig.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close(fig)

    npz_path = None
    if save_npz:
        npz_path = os.path.join(
            visualize_path,
            os.path.splitext(filename)[0] + '.npz',
        )
        np.savez_compressed(
            npz_path,
            association=A,
            hard_overlap=H,
            hard_overlap_counts=H_counts,
            transport=T,
            transport_visual= T_vis,
            cosine_cost=result['cosine_cost'].detach().cpu().numpy(),
            enhanced_cost=result['enhanced_cost'].detach().cpu().numpy(),
        )

    print(f'Saved POTC transport analysis SVG: {output_path}')
    if npz_path:
        print(f'Saved raw transport-analysis arrays: {npz_path}')
    return {**result, 'svg_path': output_path, 'npz_path': npz_path}


# ------------------ Experiment 2: recall / HR / NDCG data ------------------

def _per_user_recall(test_items, ranked_items, k):
    truth = set(int(x) for x in test_items)
    if not truth:
        return 0.0
    hits = sum(1 for x in ranked_items[:k] if int(x) in truth)
    return float(hits / len(truth))


def _per_user_hr(test_items, ranked_items, k):
    truth = set(int(x) for x in test_items)
    if not truth:
        return 0.0
    return float(any(int(x) in truth for x in ranked_items[:k]))


def _per_user_ndcg(test_items, ranked_items, k):
    truth = set(int(x) for x in test_items)
    if not truth:
        return 0.0

    ranked = ranked_items[:k]
    dcg = 0.0
    for rank, item in enumerate(ranked, start=1):
        if int(item) in truth:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(truth), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )
    return float(dcg / idcg)


@torch.no_grad()
def _analysis_model_score_batch(model, source_users, target_users):
    """
    Score a batch using the actual POTC API or a compatible score API.

    For the current POTC implementation, forward(source_users, target_users)
    returns:
        mapped_source, wd_loss, target_item_embeddings
    so ranking scores are mapped_source @ target_item_embeddings.T.
    """
    was_training = model.training
    model.eval()
    try:
        score_method = getattr(model, 'score', None)
        if callable(score_method):
            try:
                output = score_method(
                    source_users, target_item_ids=None, training=False
                )
            except TypeError:
                try:
                    output = score_method(source_users)
                except TypeError:
                    output = score_method(source_users, target_users)

            if torch.is_tensor(output):
                if output.ndim != 2:
                    raise ValueError('Model score tensor must be 2-D.')
                return output

            if isinstance(output, (tuple, list)):
                if (
                    len(output) >= 2
                    and torch.is_tensor(output[0])
                    and output[0].ndim == 2
                    and output[0].shape[1] == target_users.numel()
                ):
                    return output[0]

        output = model(source_users, target_users)
        if isinstance(output, (tuple, list)) and len(output) >= 3:
            mapped_source, _, target_items = output[:3]
            if (
                torch.is_tensor(mapped_source)
                and torch.is_tensor(target_items)
                and mapped_source.ndim == 2
                and target_items.ndim == 2
            ):
                return mapped_source @ target_items.T

        if torch.is_tensor(output):
            if output.ndim != 2:
                raise ValueError('Model output tensor must be 2-D.')
            return output

        if isinstance(output, (tuple, list)) and output:
            first = output[0]
            if torch.is_tensor(first) and first.ndim == 2:
                return first

        raise TypeError(
            'Could not infer target-item scores from the supplied model output.'
        )
    finally:
        model.train(was_training)


@torch.no_grad()
def _analysis_user_confidence(model, source_user_ids):
    """
    User-specific confidence:

        C_i = max_j sum_k p_s(k | u_i) * A[k,j]
    """
    c = _analysis_components(model)
    source_emb = c['source_users'][source_user_ids]
    ps = F.softmax(source_emb @ c['source_prototypes'].T, dim=1)
    A = potc_analysis_association_matrix(
        c['source_encoder'], c['target_encoder'],
        c['source_prototypes'], c['target_prototypes'],
        c['overlap_src2tgt'],
    )
    return torch.max(ps @ A, dim=1).values


def save_correspondence_metric_analysis_csv(
    source_domain,
    target_domain,   
    model,
    cdr_dataset,
    tgt2src_tensor,
    k=10,
    test_batch_size=100,
    filename=None,
):
    """
    Experiment 2 data collection.

    For every target-side CDR test user u_i:
      C_i             = user-specific correspondence confidence;
      R_i^POTC        = per-user Recall@k;
      HR_i^POTC       = per-user Hit Rate@k;
      NDCG_i^POTC     = per-user NDCG@k.

    The baseline values are scalar average metrics and are recorded for
    reference only. No per-user gain is calculated here.
    """
    test_users = sorted(int(u) for u in cdr_dataset.testDict.keys())
    if not test_users:
        raise ValueError('cdr_dataset.testDict is empty.')

    overlap_src2tgt = _model_attr(
        model,
        'overlap_src2tgt',
        'overlap_src_to_tgt',
    )
    overlap_target_ids = {int(x) for x in overlap_src2tgt.values()}
    missing_overlap = [u for u in test_users if u not in overlap_target_ids]
    if missing_overlap:
        raise ValueError(
            f'{len(missing_overlap)} test users are absent from the overlap mapping. '
            'The requested confidence analysis therefore cannot assign a valid '
            'source-side correspondence to all test users.'
        )

    device = tgt2src_tensor.device
    target_ids_all = torch.as_tensor(
        test_users,
        dtype=torch.long,
        device=device,
    )
    source_ids_all = tgt2src_tensor[target_ids_all]

    if torch.any(source_ids_all < 0):
        raise ValueError('Negative source user ID found in overlap mapping.')

    confidences = []
    potc_recalls = []
    potc_hrs = []
    potc_ndcgs = []

    for start in range(0, len(test_users), int(test_batch_size)):
        end = min(start + int(test_batch_size), len(test_users))

        batch_target = target_ids_all[start:end]
        batch_source = source_ids_all[start:end]

        confidence = _analysis_user_confidence(
            model,
            batch_source,
        )
        confidences.extend(
            confidence.detach().cpu().numpy().tolist()
        )

        scores = _analysis_model_score_batch(
            model,
            batch_source,
            batch_target,
        )

        k_eff = min(int(k), int(scores.shape[1]))

        # Same exclusion logic as the existing evaluation:
        # mask train/valid positives while keeping held-out test items.
        for row, target_uid in enumerate(test_users[start:end]):
            truth = cdr_dataset.testDict[target_uid]
            seen = cdr_dataset.getUserPosItems(target_uid) - set(truth)

            if seen:
                ids = [
                    x for x in seen
                    if 0 <= int(x) < scores.shape[1]
                ]
                if ids:
                    scores[row, ids] = -float('inf')

        top_items = torch.topk(
            scores,
            k=k_eff,
            dim=1,
        ).indices

        for row, target_uid in enumerate(test_users[start:end]):
            truth = cdr_dataset.testDict[target_uid]
            ranked = top_items[row].detach().cpu().tolist()

            potc_recalls.append(
                _per_user_recall(truth, ranked, int(k))
            )
            potc_hrs.append(
                _per_user_hr(truth, ranked, int(k))
            )
            potc_ndcgs.append(
                _per_user_ndcg(truth, ranked, int(k))
            )

    confidences = np.asarray(confidences, dtype=np.float64)
    potc_recalls = np.asarray(potc_recalls, dtype=np.float64)
    potc_hrs = np.asarray(potc_hrs, dtype=np.float64)
    potc_ndcgs = np.asarray(potc_ndcgs, dtype=np.float64)

    out_dir = os.path.join(
        visualize_path,
        _model_run_tag(model, source_domain, target_domain),
    )
    os.makedirs(out_dir, exist_ok=True)

    if filename is None:
        filename = "correspondence_metrics.csv"

    output_path = os.path.join(out_dir, filename)

    rows = np.column_stack([
        np.asarray(test_users, dtype=np.int64),
        source_ids_all.detach().cpu().numpy().astype(np.int64),
        confidences,
        potc_recalls,
        potc_hrs,
        potc_ndcgs,
    ])

    np.savetxt(
        output_path,
        rows,
        delimiter=',',
        header=(
            'target_user_id,source_user_id,correspondence_confidence,'
            f'potc_recall_at_{k},'
            f'potc_hr_at_{k},'
            f'potc_ndcg_at_{k}'
        ),
        comments='',
    )

    print(f'Saved correspondence/metric data: {output_path}')

    return {
        'csv_path': output_path,
        'correspondence_confidence': confidences,
        'potc_recall': potc_recalls,
        'potc_hr': potc_hrs,
        'potc_ndcg': potc_ndcgs,
    }

# ------------------ Experiment 3: mapped source / target embeddings ------------------

@torch.no_grad()
def save_mapped_source_target_embeddings_csv(
    source_domain,
    target_domain,
    model,
    cdr_dataset,
    tgt2src_tensor,
    filename=None,
):
    """
    Record every test user's:
      - target user ID
      - source user ID
      - mapped source embedding produced by POTC
      - pretrained target embedding

    The target embedding is the frozen pretrained target-side user
    representation. The source embedding is the final mapped source
    representation produced by POTC.

    Saves one CSV file. Each embedding dimension is stored in its own
    column.
    """
    test_users = sorted(int(u) for u in cdr_dataset.testDict.keys())
    if not test_users:
        raise ValueError('cdr_dataset.testDict is empty.')

    overlap_src2tgt = _model_attr(
        model,
        'overlap_src2tgt',
        'overlap_src_to_tgt',
    )

    overlap_target_ids = {int(x) for x in overlap_src2tgt.values()}
    missing_overlap = [u for u in test_users if u not in overlap_target_ids]

    if missing_overlap:
        raise ValueError(
            f'{len(missing_overlap)} test users are absent from the overlap mapping.'
        )

    device = tgt2src_tensor.device

    target_ids = torch.as_tensor(
        test_users,
        dtype=torch.long,
        device=device,
    )
    source_ids = tgt2src_tensor[target_ids]

    if torch.any(source_ids < 0):
        raise ValueError('Negative source user ID found in overlap mapping.')

    model.eval()

    # ------------------------------------------------------------------
    # Get final mapped source embeddings using the actual POTC forward API
    # ------------------------------------------------------------------
    output = model(source_ids, target_ids)

    if not isinstance(output, (tuple, list)) or len(output) < 3:
        raise TypeError(
            'Expected POTC forward() to return '
            '(mapped_source, wd_loss, target_item_embeddings).'
        )

    mapped_source_all = output[0]

    if not torch.is_tensor(mapped_source_all) or mapped_source_all.ndim != 2:
        raise ValueError('mapped_source must be a 2-D tensor.')

    # ------------------------------------------------------------------
    # Get frozen pretrained target-side user embeddings
    # ------------------------------------------------------------------
    c = _analysis_components(model)
    target_users_all = c['target_users'][target_ids]

    if target_users_all.ndim != 2:
        raise ValueError('Pretrained target user embeddings must be 2-D.')

    if mapped_source_all.shape[0] != target_users_all.shape[0]:
        raise ValueError('Mapped source and target user counts do not match.')

    if mapped_source_all.shape[1] != target_users_all.shape[1]:
        raise ValueError(
            'Mapped source and target embedding dimensions do not match. '
            't-SNE comparison requires equal-dimensional embeddings.'
        )

    mapped_source_np = (
        mapped_source_all.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    target_np = (
        target_users_all.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    # ------------------------------------------------------------------
    # Get all prototype embeddings
    # ------------------------------------------------------------------
    source_prototypes_np = (
        model.sp.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    mapped_source_prototypes_np = (
        model.cluster_map(model.sp)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    target_prototypes_np = (
        model.tp.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    out_dir = os.path.join(
        visualize_path,
        _model_run_tag(model, source_domain, target_domain),
    )
    os.makedirs(out_dir, exist_ok=True)

    if filename is None:
        filename = "mappeds_t_embs.csv"

    output_path = os.path.join(out_dir, filename)

    # ------------------------------------------------------------------
    # Build CSV
    # ------------------------------------------------------------------
    rows = []

    for i, (target_uid, source_uid) in enumerate(
        zip(test_users, source_ids.detach().cpu().numpy())
    ):
        row = {
            'target_user_id': int(target_uid),
            'source_user_id': int(source_uid),
        }

        for d in range(mapped_source_np.shape[1]):
            row[f'mapped_source_emb_{d}'] = mapped_source_np[i, d]

        for d in range(target_np.shape[1]):
            row[f'target_emb_{d}'] = target_np[i, d]

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f'Saved mapped source/target embeddings: {output_path}')

    prototype_csv_path = os.path.join(out_dir, 'prototypes_embs.csv')

    prototype_rows = []

    num_prototypes = source_prototypes_np.shape[0]

    for i in range(num_prototypes):
        row = {
            'prototype_id': i,
        }

        for d in range(source_prototypes_np.shape[1]):
            row[f'source_prototype_emb_{d}'] = source_prototypes_np[i, d]

        for d in range(mapped_source_prototypes_np.shape[1]):
            row[f'mapped_source_prototype_emb_{d}'] = (
                mapped_source_prototypes_np[i, d]
            )

        for d in range(target_prototypes_np.shape[1]):
            row[f'target_prototype_emb_{d}'] = target_prototypes_np[i, d]

        prototype_rows.append(row)

    pd.DataFrame(prototype_rows).to_csv(
        prototype_csv_path,
        index=False,
    )

    print(f'Saved prototype embeddings: {prototype_csv_path}')

    return {
        'csv_path': output_path,
        'target_user_ids': np.asarray(test_users, dtype=np.int64),
        'source_user_ids': source_ids.detach().cpu().numpy().astype(np.int64),
        'mapped_source_embeddings': mapped_source_np,
        'target_embeddings': target_np,
    }
