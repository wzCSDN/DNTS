import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from scipy.sparse import linalg
from torch.autograd import Variable
from scipy.sparse import csr_matrix, coo_matrix

from collections import defaultdict


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class DataLoaderS(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, device, horizon, window, normalize=2):
        self.P = window
        self.h = horizon
        fin = open(file_name)
        self.rawdat = np.loadtxt(fin, delimiter=',')
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        self.scale = self.scale.to(device)
        self.scale = Variable(self.scale)

        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

        self.device = device

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            self.dat = self.rawdat / np.max(self.rawdat)

        # normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.to(self.device)
            Y = Y.to(self.device)
            yield Variable(X), Variable(Y)
            start_idx += batch_size


class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj


def focal_loss_new(preds, labels, alpha=0.25, gamma=2):
    eps = 1e-7

    # Clamping preds to avoid log(0)
    preds = torch.clamp(preds, eps, 1 - eps)

    loss_1 = -1 * alpha * torch.pow((1 - preds), gamma) * torch.log(preds) * labels
    loss_0 = -1 * (1 - alpha) * torch.pow(preds, gamma) * torch.log(1 - preds) * (1 - labels)

    loss = loss_0 + loss_1

    # Debugging prints
    if torch.any(torch.isnan(loss)):
        print("Loss contains NaN values")
    if torch.all(loss == 0):
        print("All loss values are 0")

    return torch.mean(loss)


def focal_loss_old(logits, targets, alpha=1, gamma=2):
    BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-BCE_loss)  # 原始概率
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss  # Focal loss
    return torch.mean(F_loss)


# def focal_loss(preds, labels,alpha=0.25,gamma=2):
#     """
#     preds:sigmoid的输出结果
#     labels：标签
#     """
#     # import pdb
#     # pdb.set_trace()
#     eps = 1e-7
#     loss_1 = -1 * alpha * torch.pow((1 - preds), gamma) * torch.log(preds + eps) * labels
#     loss_0 = -1 * (1 - alpha) * torch.pow(preds, gamma) * torch.log(1 - preds + eps) * (1 - labels)
#     loss = loss_0 + loss_1
#     return torch.mean(loss)

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    # mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    count = torch.sum(mask)

    loss = (torch.log(preds + 1) - torch.log(labels + 1)) ** 2
    # loss = (preds - labels) ** 2
    loss = loss * mask
    if count == 0:
        loss = torch.tensor(0.0, requires_grad=False)
        return loss
    else:
        return torch.sum(loss) / count


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=0.5):
    y = torch.log(logits + 1) + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def find_nei(data, is_transpose):
    nei_dict = defaultdict(set)
    category = ['x_train_graph', 'x_val_graph', 'x_test_graph']
    for modern in category:
        for p_id in data[modern].keys():

            nei_dict[p_id] = {}
            for t in data[modern][p_id].keys():
                nei_dict[p_id][t] = {}
                temp_adj = data[modern][p_id][t]
                sparse_matrix_coo = temp_adj.tocoo()
                if sparse_matrix_coo.nnz == 0:
                    nei_dict[p_id][t] = {}
                    continue
                else:
                    if is_transpose:
                        symmetric_data = []
                        symmetric_rows = []
                        symmetric_cols = []
                        for row, col, value in zip(sparse_matrix_coo.row, sparse_matrix_coo.col,
                                                   sparse_matrix_coo.data):
                            symmetric_data.append(value)
                            symmetric_rows.append(row)
                            symmetric_cols.append(col)
                            # 添加对称项（如果当前是上三角，确保下三角也存在）
                            if row != col and (col, row) not in zip(symmetric_rows, symmetric_cols):
                                symmetric_data.append(value)
                                symmetric_rows.append(col)
                                symmetric_cols.append(row)
                        symmetric_matrix_coo = coo_matrix((symmetric_data, (symmetric_rows, symmetric_cols)))
                    else:
                        symmetric_matrix_coo = sparse_matrix_coo
                    for r, c in zip(symmetric_matrix_coo.row, symmetric_matrix_coo.col):
                        nei_dict[p_id][t].setdefault(r, set()).add(c)
                        nei_dict[p_id][t].setdefault(c, set()).add(r)

    with open('data/nei_dict.pkl', 'wb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        pickle.dump(nei_dict, pkl_file)

    return nei_dict


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    # mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    count = torch.sum(mask)
    loss = torch.abs((torch.log(preds + 1) - torch.log(labels + 1)))

    loss = loss * mask
    if count == 0:
        loss = torch.tensor(0.0, requires_grad=False)
        return loss
    else:
        return torch.sum(loss) / count


def masked_mae_loss(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)  # 都是64 1 207 1  最后的1为任务难度  将多天切割为多个一天去学  mask意思就是不预测null_val的值
    mask = mask.float()
    # mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    count = torch.sum(mask)

    # # loss = torch.abs(preds - labels)
    loss = torch.abs((torch.log(preds + 1) - torch.log(labels + 1)))

    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

    # nan_mask = torch.isnan(preds)  # 或者根据具体情况定义预测为None的条件
    # penalty = nan_mask.float()
    # penalty_loss = penalty * 1.0
    # total_loss = loss + penalty_loss
    # if torch.all(nan_mask) or torch.all((labels == 0)):
    if count == 0:
        loss = torch.tensor(0.0, requires_grad=False)
        return loss
    else:
        return torch.sum(loss) / count


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    # mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    count = torch.sum(mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if count == 0:
        loss = torch.tensor(0.0, requires_grad=False)
        return loss
    else:
        return torch.sum(loss) / count


def metric(pred, real):
    mae = masked_mae(pred, real, 0.0).item()
    mape = masked_mape(pred, real, 0.0).item()
    rmse = masked_rmse(pred, real, 0.0).item()
    return mae, mape, rmse


def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    z = torch.tensor((x - mean) / std, dtype=torch.float)
    return z


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))



