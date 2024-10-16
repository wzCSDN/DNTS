import numpy as np
import random
import pickle
import copy


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


class DataLoader_G(object):
    def __init__(self, graph, graph_org, label, index, batch_size, pad_with_last_sample=True):

        self.batch_size = batch_size
        self.size = len(graph)
        self.current_ind = 0
        self.index = index
        if self.size % self.batch_size == 0:
            self.num_batch = int(self.size // self.batch_size)
        else:
            self.num_batch = int(self.size // self.batch_size) + 1
        self.graph = graph
        self.graph_org = graph_org
        self.label = label

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                feature = self.graph[start_ind: end_ind, ...]
                org = self.graph_org[start_ind: end_ind, ...]

                y = self.label[start_ind: end_ind, ...]
                index = self.index[start_ind: end_ind, ...]
                yield (feature, org, y, index)
                # keys_in_range = list(self.graph.keys())[start_ind:end_ind]
                # x_graph = {key: self.graph[key] for key in keys_in_range}
                # x_feature = {key: self.feature[key] for key in keys_in_range}
                # y = {key: self.label[key] for key in keys_in_range}
                # yield (x_graph,x_feature, y)
                self.current_ind += 1

        return _wrapper()


class DataLoader_C(object):
    def __init__(self, feature, graph, graph_inv, y_feature, index, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.size = len(graph)
        self.current_ind = 0
        self.index = index
        if self.size % self.batch_size == 0:
            self.num_batch = int(self.size // self.batch_size)
        else:
            self.num_batch = int(self.size // self.batch_size) + 1
        self.x_graph = graph
        self.x_graph_inv = graph_inv
        self.x_feature = feature
        self.index_set = self.unique_ordered_list(self.index)
        self.item_num = len(self.index_set)
        self.y_feature = y_feature

    def unique_ordered_list(self, lst):
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def find_first_and_last_occurrence(self, lst, value):
        """找到指定值在列表中第一次和最后一次出现的索引"""
        first_index = None
        last_index = None
        for i, v in enumerate(lst):
            if v == value:
                if first_index is None:
                    first_index = i
                last_index = i
        # 检查是否找到了该值
        if first_index is None:
            return None, None  # 如果值不在列表中，返回(None, None)
        else:
            return first_index, last_index

    def get_iterator(self):
        self.cur_i = 0
        self.start_index = self.index_set[0]

        def _wrapper():
            while self.cur_i < self.item_num:
                first_index, last_index = self.find_first_and_last_occurrence(self.index, self.start_index)
                start_ind = first_index
                end_ind = last_index + 1

                x_feature = self.x_feature[start_ind: end_ind, ...]
                x_graph_inv = self.x_graph_inv[start_ind: end_ind, ...]
                x_graph = self.x_graph[start_ind: end_ind, ...]
                y_feature = self.y_feature[start_ind: end_ind, ...]
                yield (x_feature, x_graph, x_graph_inv, y_feature, self.index_set[self.cur_i])
                self.cur_i = self.cur_i + 1
                if len(self.index_set) > self.cur_i:
                    self.start_index = self.index_set[self.cur_i]

        return _wrapper()


class DataLoader(object):
    def __init__(self, feature, label, batch_size, pad_with_last_sample=True):

        self.batch_size = batch_size
        self.size = len(feature)
        self.current_ind = 0
        if self.size % self.batch_size == 0:
            self.num_batch = int(self.size // self.batch_size)
        else:
            self.num_batch = int(self.size // self.batch_size) + 1
        self.feature = feature
        self.label = label

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        # feature, graph,label,index = self.feature[permutation], self.graph[permutation],self.label[permutation],self.index[permutation]
        feature, label = self.feature[permutation], self.label[permutation]

        self.feature = feature

        self.label = label

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))

                feature = self.feature[start_ind: end_ind, ...]
                y = self.label[start_ind: end_ind, ...]
                yield (feature, y)
                # keys_in_range = list(self.graph.keys())[start_ind:end_ind]
                # x_graph = {key: self.graph[key] for key in keys_in_range}
                # x_feature = {key: self.feature[key] for key in keys_in_range}
                # y = {key: self.label[key] for key in keys_in_range}
                # yield (x_graph,x_feature, y)
                self.current_ind += 1

        return _wrapper()


def split_dict(dict, beg_index, end_index, horizon=None):
    first_half = {k: v for i, (k, v) in enumerate(dict.items()) if i < beg_index}
    if horizon is None:
        second_half = {k: v for i, (k, v) in enumerate(dict.items()) if i >= beg_index and i < beg_index + end_index}

    else:
        second_half = {k: v for i, (k, v) in enumerate(dict.items()) if
                       i >= beg_index + horizon and i <= beg_index + horizon + 1}
    return first_half, second_half


def id_map(graph_dict, sales_dict, has_item):
    with open('data/mktid_member_net.pkl', 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        mktid_member_net = pickle.load(pkl_file)
    from scipy.sparse import coo_matrix
    max_key = 0
    id_map = {}
    sales_mapped_dict = copy.deepcopy(sales_dict)
    graph_mapped_dict = copy.deepcopy(graph_dict)

    for sample_idx, sample in sales_dict.items():
        id_map[sample_idx] = {}
        old_id = mktid_member_net[sample_idx]
        if has_item:
            old_id.add(0)
        n = len(old_id)
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(old_id))}
        if max_key == 0:
            max_key = max(id_mapping.keys())
        else:
            if max(id_mapping.keys()) > max_key:
                max_key = max(id_mapping.keys())
        id_map[sample_idx] = id_mapping
        for time, t in sample.items():

            new_id = {id_mapping[k]: v for k, v in sales_dict[sample_idx][time].items() if k in id_mapping}
            sales_mapped_dict[sample_idx][time] = new_id
            A = graph_dict[sample_idx][time].tocoo()
            rows, cols, data = [], [], []
            for row, col, value in zip(A.row, A.col, A.data):
                if row == 0 or col == 0:
                    continue
                else:
                    new_row, new_col = id_mapping[row], id_mapping[col]
                    rows.append(new_row)
                    cols.append(new_col)
                    data.append(value)
            new_A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
            graph_mapped_dict[sample_idx][time] = new_A
    if has_item:
        with open('data/id_mapped_sales.pkl', 'wb') as pkl_file:
            # 使用pickle的dump方法将字典写入文件
            pickle.dump(sales_mapped_dict, pkl_file)
        with open('data/id_mapped_graph.pkl', 'wb') as pkl_file:
            # 使用pickle的dump方法将字典写入文件
            pickle.dump(graph_mapped_dict, pkl_file)
        with open('data/id_map.pkl', 'wb') as pkl_file:
            # 使用pickle的dump方法将字典写入文件
            pickle.dump(id_map, pkl_file)
        return graph_mapped_dict, sales_mapped_dict, id_map, max_key
    else:
        with open('data/id_mapped_sales_noi.pkl', 'wb') as pkl_file:
            # 使用pickle的dump方法将字典写入文件
            pickle.dump(sales_mapped_dict, pkl_file)
        with open('data/id_mapped_graph_noi.pkl', 'wb') as pkl_file:
            # 使用pickle的dump方法将字典写入文件
            pickle.dump(graph_mapped_dict, pkl_file)
        with open('data/id_map_noi.pkl', 'wb') as pkl_file:
            # 使用pickle的dump方法将字典写入文件
            pickle.dump(id_map, pkl_file)
        return graph_mapped_dict, sales_mapped_dict, id_map, max_key


def get_key(att_dict, sales_dict, graph_dict, graph_inv_dict, has_item):
    with open('data/mktid_member_net.pkl', 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        mktid_member_net = pickle.load(pkl_file)
    from scipy.sparse import coo_matrix
    max_key = 0
    max_node_len = 0
    max_off_len = 0
    id_map = {}
    for sample_idx, sample in sales_dict.items():
        id_map[sample_idx] = {}
        old_id = mktid_member_net[sample_idx]
        if has_item:
            old_id.add(0)
        n = len(old_id)
        id_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(sorted(old_id))}
        if max_key == 0:
            max_key = max(id_mapping.keys())
        else:
            if max(id_mapping.keys()) > max_key:
                max_key = max(id_mapping.keys())

    return max_key


def id_map_cas(att_dict, sales_dict, graph_dict, graph_inv_dict, has_item):
    with open('data/mktid_member_net.pkl', 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        mktid_member_net = pickle.load(pkl_file)
    from scipy.sparse import coo_matrix
    max_key = 0
    max_node_len = 0
    max_off_len = 0
    id_map = {}
    sales_mapped_dict = copy.deepcopy(sales_dict)
    graph_mapped_dict = copy.deepcopy(graph_dict)
    att_mapped_dict = copy.deepcopy(att_dict)
    graph_mapped_inv_dict = copy.deepcopy(graph_inv_dict)
    for sample_idx, sample in sales_dict.items():
        id_map[sample_idx] = {}
        old_id = mktid_member_net[sample_idx]
        if has_item:
            old_id.add(0)
        n = len(old_id)
        id_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(sorted(old_id))}
        if max_key == 0:
            max_key = max(id_mapping.keys())
        else:
            if max(id_mapping.keys()) > max_key:
                max_key = max(id_mapping.keys())
        if max_node_len == 0:
            max_node_len = len(id_mapping)
        else:
            if len(id_mapping) > max_node_len:
                max_node_len = len(id_mapping)
        id_map[sample_idx] = id_mapping
        for time, t in sample.items():
            new_id_sales = {id_mapping[k]: v for k, v in sales_dict[sample_idx][time].items() if k in id_mapping}
            new_id_att = {id_mapping[k]: v for k, v in att_dict[sample_idx][time].items() if k in id_mapping}
            sales_mapped_dict[sample_idx][time] = new_id_sales
            att_mapped_dict[sample_idx][time] = new_id_att
            new_id_graph = {id_mapping[k]: [id_mapping[j] for j in list(v)] for k, v in
                            graph_dict[sample_idx][time].items() if k in id_mapping}
            new_id_graph_inv = {id_mapping[k]: [id_mapping[j] for j in list(v)] for k, v in
                                graph_inv_dict[sample_idx][time].items() if k in id_mapping}

            graph_mapped_dict[sample_idx][time] = new_id_graph
            graph_mapped_inv_dict[sample_idx][time] = new_id_graph_inv
    if has_item:
        with open('data/id_mapped_sales.pkl', 'wb') as pkl_file:
            # 使用pickle的dump方法将字典写入文件
            pickle.dump(sales_mapped_dict, pkl_file)
        with open('data/id_mapped_off.pkl', 'wb') as pkl_file:
            # 使用pickle的dump方法将字典写入文件
            pickle.dump(graph_mapped_dict, pkl_file)
        with open('data/id_mapped_off_inv.pkl', 'wb') as pkl_file:
            # 使用pickle的dump方法将字典写入文件
            pickle.dump(graph_mapped_inv_dict, pkl_file)
        with open('data/id_map.pkl', 'wb') as pkl_file:
            # 使用pickle的dump方法将字典写入文件
            pickle.dump(id_map, pkl_file)
        with open('data/id_mapped_att.pkl', 'wb') as pkl_file:
            # 使用pickle的dump方法将字典写入文件
            pickle.dump(att_mapped_dict, pkl_file)
        return graph_mapped_dict, sales_mapped_dict, id_map
    else:
        with open('data/id_mapped_sales_noi.pkl', 'wb') as pkl_file:
            # 使用pickle的dump方法将字典写入文件
            pickle.dump(sales_mapped_dict, pkl_file)
        with open('data/id_mapped_off_noi.pkl', 'wb') as pkl_file:
            # 使用pickle的dump方法将字典写入文件
            pickle.dump(graph_mapped_dict, pkl_file)
        with open('data/id_mapped_off_inv_noi.pkl', 'wb') as pkl_file:
            # 使用pickle的dump方法将字典写入文件
            pickle.dump(graph_mapped_inv_dict, pkl_file)
        with open('data/id_map_noi.pkl', 'wb') as pkl_file:
            # 使用pickle的dump方法将字典写入文件
            pickle.dump(id_map, pkl_file)
        with open('data/id_mapped_att_noi.pkl', 'wb') as pkl_file:
            # 使用pickle的dump方法将字典写入文件
            pickle.dump(att_mapped_dict, pkl_file)
        return graph_mapped_dict, graph_mapped_inv_dict, sales_mapped_dict, att_mapped_dict, id_map, max_key, max_node_len, max_off_len


def load_dataset(sales_dict, graph_dict, seq_in_len, seq_out_len, train_ratio=None, valid_ratio=None, test_ratio=None):
    data = {}
    product_id_list = list(sales_dict.keys())
    all_steps = len(graph_dict[product_id_list[0]])
    x_step = seq_in_len
    random.shuffle(product_id_list)
    y_step = seq_out_len
    total_product_num = len(product_id_list)
    train_size = int(total_product_num * train_ratio)
    val_size = int(total_product_num * valid_ratio)
    test_size = total_product_num - train_size - val_size

    train_ids = product_id_list[:train_size]
    val_ids = product_id_list[train_size:train_size + val_size]
    test_ids = product_id_list[train_size + val_size:]

    data['x_train_graph'] = {id: split_dict(graph_dict[id], x_step, y_step)[0] for id in train_ids}
    data['x_val_graph'] = {id: split_dict(graph_dict[id], x_step, y_step)[0] for id in val_ids}
    data['x_test_graph'] = {id: split_dict(graph_dict[id], x_step, y_step)[0] for id in test_ids}

    data['x_train_feature'] = {id: split_dict(sales_dict[id], x_step, y_step)[0] for id in train_ids}
    data['x_val_feature'] = {id: split_dict(sales_dict[id], x_step, y_step)[0] for id in val_ids}
    data['x_test_feature'] = {id: split_dict(sales_dict[id], x_step, y_step)[0] for id in test_ids}
    data['y_train'] = {id: split_dict(sales_dict[id], x_step, y_step)[1] for id in train_ids}
    data['y_val'] = {id: split_dict(sales_dict[id], x_step, y_step)[1] for id in val_ids}
    data['y_test'] = {id: split_dict(sales_dict[id], x_step, y_step)[1] for id in test_ids}

    return data, train_size, val_size, test_size


def load_cas_dataset(att_dict, sales_dict, off_dict, off_trans, seq_in_len, seq_out_len, train_ratio=None,
                     valid_ratio=None, test_ratio=None):
    data = {}
    product_id_list = list(sales_dict.keys())

    x_step = seq_in_len
    random.shuffle(product_id_list)
    y_step = seq_out_len

    total_product_num = len(product_id_list)
    train_size = int(total_product_num * train_ratio)
    val_size = int(total_product_num * valid_ratio)
    test_size = total_product_num - train_size - val_size

    train_ids = product_id_list[:train_size]
    val_ids = product_id_list[train_size:train_size + val_size]
    test_ids = product_id_list[train_size + val_size:]

    data['x_train_graph_inv'] = {id: split_dict(off_trans[id], x_step, y_step)[0] for id in train_ids}
    data['x_val_graph_inv'] = {id: split_dict(off_trans[id], x_step, y_step)[0] for id in val_ids}
    data['x_test_graph_inv'] = {id: split_dict(off_trans[id], x_step, y_step)[0] for id in test_ids}

    data['x_train_graph'] = {id: split_dict(off_dict[id], x_step, y_step)[0] for id in train_ids}
    data['x_val_graph'] = {id: split_dict(off_dict[id], x_step, y_step)[0] for id in val_ids}
    data['x_test_graph'] = {id: split_dict(off_dict[id], x_step, y_step)[0] for id in test_ids}

    data['x_train_feature'] = {id: split_dict(sales_dict[id], x_step, y_step)[0] for id in train_ids}
    data['x_val_feature'] = {id: split_dict(sales_dict[id], x_step, y_step)[0] for id in val_ids}
    data['x_test_feature'] = {id: split_dict(sales_dict[id], x_step, y_step)[0] for id in test_ids}

    data['y_train_feature'] = {id: split_dict(att_dict[id], x_step, y_step)[1] for id in train_ids}
    data['y_val_feature'] = {id: split_dict(att_dict[id], x_step, y_step)[1] for id in val_ids}
    data['y_test_feature'] = {id: split_dict(att_dict[id], x_step, y_step)[1] for id in test_ids}
    data['y_train_graph'] = {id: split_dict(off_dict[id], x_step, y_step)[1] for id in train_ids}
    data['y_val_graph'] = {id: split_dict(off_dict[id], x_step, y_step)[1] for id in val_ids}
    data['y_test_graph'] = {id: split_dict(off_dict[id], x_step, y_step)[1] for id in test_ids}

    return data, train_size, val_size, test_size


def load_dataset_single(graph_file_path, sales_file_path, horizon, seq_in_len, seq_out_len, train_ratio=None,
                        valid_ratio=None, test_ratio=None):
    with open(graph_file_path, 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        graph_dict = pickle.load(pkl_file)
    with open(sales_file_path, 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        sales_dict = pickle.load(pkl_file)
    data = {}
    product_id_list = list(sales_dict.keys())
    all_steps = len(graph_dict[product_id_list[0]])
    assert (all_steps >= seq_in_len + seq_out_len + horizon)
    x_step = seq_in_len
    random.shuffle(product_id_list)

    total_product_num = len(product_id_list)
    train_size = int(total_product_num * train_ratio)
    val_size = int(total_product_num * valid_ratio)
    test_size = total_product_num - train_size - val_size

    train_ids = product_id_list[:train_size]
    val_ids = product_id_list[train_size:train_size + val_size]
    test_ids = product_id_list[train_size + val_size:]

    data['x_train_graph'] = {id: split_dict(graph_dict[id], x_step, horizon)[0] for id in train_ids}
    data['x_val_graph'] = {id: split_dict(graph_dict[id], x_step, horizon)[0] for id in val_ids}
    data['x_test_graph'] = {id: split_dict(graph_dict[id], x_step, horizon)[0] for id in test_ids}

    data['x_train_feature'] = {id: split_dict(sales_dict[id], x_step, horizon)[0] for id in train_ids}
    data['x_val_feature'] = {id: split_dict(sales_dict[id], x_step, horizon)[0] for id in val_ids}
    data['x_test_feature'] = {id: split_dict(sales_dict[id], x_step, horizon)[0] for id in test_ids}
    data['y_train'] = {id: split_dict(sales_dict[id], x_step, horizon)[1] for id in train_ids}
    data['y_val'] = {id: split_dict(sales_dict[id], x_step, horizon)[1] for id in val_ids}
    data['y_test'] = {id: split_dict(sales_dict[id], x_step, horizon)[1] for id in test_ids}

    return data, train_size, val_size, test_size


def generate_grid_loader(data, time_train_span, time_predict_span, max_n, channel_num, batch_size):
    raw_data = data
    data = {}
    train_size = len(raw_data['x_train_feature'])
    val_size = len(raw_data['x_val_feature'])
    test_size = len(raw_data['x_test_feature'])
    with open('data/mktid_member_net.pkl', 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        mktid_member_net = pickle.load(pkl_file)
    train_index = []
    val_index = []
    test_index = []
    train_size_new = 0
    val_size_new = 0
    test_size_new = 0
    train_keys = list(raw_data['y_train'].keys())
    val_keys = list(raw_data['y_val'].keys())
    test_keys = list(raw_data['y_test'].keys())
    for key in train_keys:
        fre = int(len(mktid_member_net[key]) / max_n) + 1
        temp_idx = np.repeat(key, fre)
        train_index.extend(temp_idx)
        train_size_new = train_size_new + fre
    for key in val_keys:
        fre = int(len(mktid_member_net[key]) / max_n) + 1
        temp_idx = np.repeat(key, fre)
        val_index.extend(temp_idx)
        val_size_new = val_size_new + fre
    for key in test_keys:
        fre = int(len(mktid_member_net[key]) / max_n) + 1
        temp_idx = np.repeat(key, fre)
        test_index.extend(temp_idx)
        test_size_new = test_size_new + fre
    data['x_train_feature'] = np.zeros((train_size_new, time_train_span, max_n, channel_num))
    data['y_train'] = np.zeros((train_size_new, time_predict_span, max_n, channel_num))
    data['x_val_feature'] = np.zeros((val_size_new, time_train_span, max_n, channel_num))
    data['y_val'] = np.zeros((val_size_new, time_predict_span, max_n, channel_num))
    data['x_test_feature'] = np.zeros((test_size_new, time_train_span, max_n, channel_num))
    data['y_test'] = np.zeros((test_size_new, time_predict_span, max_n, channel_num))
    category_1 = ['x_train', 'x_val', 'x_test', 'y_train', 'y_val', 'y_test']

    for c in category_1:
        sample_rid = 0
        if 'x_' in c:
            modern = c + '_feature'
        else:
            modern = c
        end_batch = 0
        start_batch = 0
        for p_id, p in raw_data[modern].items():
            if sample_rid == 0:
                if len(mktid_member_net[p_id]) % max_n == 0:
                    end_batch = len(mktid_member_net[p_id]) // max_n
                else:
                    end_batch = len(mktid_member_net[p_id]) // max_n + 1
            else:
                if len(mktid_member_net[p_id]) % max_n == 0:
                    end_batch = start_batch + len(mktid_member_net[p_id]) // max_n
                else:
                    end_batch = start_batch + (len(mktid_member_net[p_id]) // max_n + 1)

            for t, time in p.items():
                temp_fea = raw_data[modern][p_id][t]
                org_id = list(temp_fea.keys())
                relative_node_id = [id % max_n for id in org_id]
                relative_batch_id = [id // max_n + start_batch for id in org_id]
                for i in range(len(org_id)):
                    if channel_num == 1:
                        if 'x_' in c:
                            data[modern][relative_batch_id[i], t - 1, relative_node_id[i], 0] = temp_fea[org_id[i]]
                        else:
                            data[modern][relative_batch_id[i], t - 1 - time_train_span, relative_node_id[i], 0] = \
                            temp_fea[org_id[i]]
            sample_rid = sample_rid + 1
            start_batch = end_batch
    scaler = StandardScaler(mean=data['x_train_feature'][..., 0].mean(), std=data['x_train_feature'][..., 0].std())
    data['scaler'] = scaler
    for c in category_1:
        if 'x_' in c:
            modern = c + '_feature'
        data[modern][..., 0] = scaler.transform(data[modern][..., 0])
    data['train_loader'] = DataLoader(data['x_train_feature'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val_feature'], data['y_val'], batch_size)
    data['test_loader'] = DataLoader(data['x_test_feature'], data['y_test'], batch_size)

    with open('data/grid_data_loader.pkl', 'wb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        pickle.dump(data, pkl_file)
    return data


def generate_graph_loader(data, data_nei, time_train_span, time_predict_span, node_num, channel_num, batch_size, k,
                          id_map):
    raw_data = data
    data = {}

    train_size = len(raw_data['x_train_feature'])
    val_size = len(raw_data['x_val_feature'])
    test_size = len(raw_data['x_test_feature'])
    with open('data/mktid_member_net.pkl', 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        mktid_member_net = pickle.load(pkl_file)
    train_index = []
    val_index = []
    test_index = []
    train_size_new = 0
    val_size_new = 0
    test_size_new = 0
    train_keys = list(raw_data['y_train'].keys())
    val_keys = list(raw_data['y_val'].keys())
    test_keys = list(raw_data['y_test'].keys())
    for key in train_keys:
        if len(mktid_member_net[key]) % node_num == 0:
            fre = int(len(mktid_member_net[key]) / node_num)
        else:
            fre = int(len(mktid_member_net[key]) / node_num) + 1
        temp_idx = np.repeat(key, fre)
        train_index.extend(temp_idx)
        train_size_new = train_size_new + fre
    for key in val_keys:
        if len(mktid_member_net[key]) % node_num == 0:
            fre = int(len(mktid_member_net[key]) / node_num)
        else:
            fre = int(len(mktid_member_net[key]) / node_num) + 1
        temp_idx = np.repeat(key, fre)
        val_index.extend(temp_idx)
        val_size_new = val_size_new + fre
    for key in test_keys:
        if len(mktid_member_net[key]) % node_num == 0:
            fre = int(len(mktid_member_net[key]) / node_num)
        else:
            fre = int(len(mktid_member_net[key]) / node_num) + 1
        temp_idx = np.repeat(key, fre)
        test_index.extend(temp_idx)
        test_size_new = test_size_new + fre
    data['x_train_graph'] = np.zeros((train_size_new, time_train_span, node_num, k))
    data['x_train_graph_org'] = np.zeros((train_size_new, time_train_span, node_num, k))
    data['y_train'] = np.zeros((train_size_new, time_predict_span, node_num, channel_num))
    data['x_val_graph'] = np.zeros((val_size_new, time_train_span, node_num, k))
    data['x_val_graph_org'] = np.zeros((val_size_new, time_train_span, node_num, k))
    data['y_val'] = np.zeros((val_size_new, time_predict_span, node_num, channel_num))
    data['x_test_graph'] = np.zeros((test_size_new, time_train_span, node_num, k))
    data['y_test'] = np.zeros((test_size_new, time_predict_span, node_num, channel_num))
    data['x_test_graph_org'] = np.zeros((test_size_new, time_train_span, node_num, k))

    data['train_index'] = np.array(train_index)
    data['val_index'] = np.array(val_index)
    data['test_index'] = np.array(test_index)
    category_1 = ['x_train', 'x_val', 'x_test', 'y_train', 'y_val', 'y_test']

    for c in category_1:
        sample_rid = 0
        if 'x_' in c:
            modern = c + '_graph'
        else:
            modern = c
        end_batch = 0
        start_batch = 0
        for p_id, p in raw_data[modern].items():

            temp_id_map = id_map[p_id]
            inver_maps = {value: key for key, value in temp_id_map.items()}
            inver_maps.update({0: 0})
            if sample_rid == 0:
                if len(mktid_member_net[p_id]) % node_num == 0:
                    end_batch = len(mktid_member_net[p_id]) // node_num
                else:
                    end_batch = len(mktid_member_net[p_id]) // node_num + 1
            else:
                if len(mktid_member_net[p_id]) % node_num == 0:
                    end_batch = start_batch + len(mktid_member_net[p_id]) // node_num
                else:
                    end_batch = start_batch + (len(mktid_member_net[p_id]) // node_num + 1)

            for t in p.keys():
                if "x" not in c:
                    temp_fea = raw_data[modern][p_id][t]
                    org_id = list(temp_fea.keys())
                else:
                    temp_adj = data_nei[p_id][t]
                    org_id = list(temp_adj.keys())
                    if len(org_id) == 0:
                        continue
                    nei_list = np.zeros((len(temp_adj), k))
                    relative_id = 0
                    for n in temp_adj:
                        temp_nei_list = list(temp_adj[n])
                        if len(temp_nei_list) < k - 1:
                            nei_list[relative_id][0] = n  # 如果邻居数量少于k，采样所有邻居
                            nei_list[relative_id][1:] = random.choices(temp_nei_list, k=k - 1)
                        else:
                            nei_list[relative_id][0] = n
                            nei_list[relative_id][1:] = np.random.choice(temp_nei_list, size=k - 1,
                                                                         replace=False).tolist()
                        relative_id = relative_id + 1
                    map_nei_list = [[inver_maps[i] for i in row] for row in nei_list]

                relative_node_id = [id % node_num for id in org_id]
                relative_batch_id = [id // node_num + start_batch for id in org_id]
                if 200 in relative_batch_id:
                    a = 5
                for i in range(len(org_id)):
                    if channel_num == 1:
                        if 'x_' in c:
                            data[modern + "_org"][relative_batch_id[i], t - 1, relative_node_id[i], :] = map_nei_list[i]
                            data[modern][relative_batch_id[i], t - 1, relative_node_id[i], :] = nei_list[i]

                        else:
                            data[modern][relative_batch_id[i], t - 1 - time_train_span, relative_node_id[i], 0] = \
                            temp_fea[org_id[i]]
            sample_rid = sample_rid + 1
            start_batch = end_batch
        a = 3

    data['train_loader'] = DataLoader_G(data['x_train_graph'], data['x_train_graph_org'], data['y_train'],
                                        data['train_index'], batch_size)
    data['val_loader'] = DataLoader_G(data['x_val_graph'], data['x_val_graph_org'], data['y_val'], data['val_index'],
                                      batch_size)
    data['test_loader'] = DataLoader_G(data['x_test_graph'], data['x_test_graph_org'], data['y_test'],
                                       data['test_index'], batch_size)

    return data


def generate_cas_loader(data, time_train_span, time_predict_span, node_num, channel_num, batch_size, neibor_size,
                        id_maps, item_num):
    raw_data = data
    data = {}
    # ['x_train_feature'] b*t*n
    # ['x_train_graph'] b*t*n*k
    k = neibor_size
    # 为每一个graph里的id map一下  从0开始map，后续链路预测任务要用
    train_size = len(raw_data['x_train_feature'])
    val_size = len(raw_data['x_val_feature'])
    test_size = len(raw_data['x_test_feature'])
    with open('data/mktid_member_net.pkl', 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        mktid_member_net = pickle.load(pkl_file)
    train_index = []
    val_index = []
    test_index = []
    train_size_new = 0
    val_size_new = 0
    test_size_new = 0
    train_keys = list(raw_data['x_train_feature'].keys())
    val_keys = list(raw_data['x_val_feature'].keys())
    test_keys = list(raw_data['x_test_feature'].keys())

    for key in train_keys:
        if len(mktid_member_net[key]) + 1 % node_num == 0:
            fre = int((len(mktid_member_net[key]) + 1) / node_num)
        else:
            fre = int((len(mktid_member_net[key]) + 1) / node_num) + 1
        temp_idx = np.repeat(key, fre)
        train_index.extend(temp_idx)
        train_size_new = train_size_new + fre
    for key in val_keys:
        if len(mktid_member_net[key]) + 1 % node_num == 0:
            fre = int((len(mktid_member_net[key]) + 1) / node_num)
        else:
            fre = int((len(mktid_member_net[key]) + 1) / node_num) + 1
        temp_idx = np.repeat(key, fre)
        val_index.extend(temp_idx)
        val_size_new = val_size_new + fre
    for key in test_keys:
        if len(mktid_member_net[key]) + 1 % node_num == 0:
            fre = int((len(mktid_member_net[key]) + 1) / node_num)
        else:
            fre = int((len(mktid_member_net[key]) + 1) / node_num) + 1
        temp_idx = np.repeat(key, fre)
        test_index.extend(temp_idx)
        test_size_new = test_size_new + fre

    data['x_train_feature'] = np.zeros((train_size_new, time_train_span, node_num, channel_num))
    data['x_train_graph'] = np.zeros((train_size_new, time_train_span, node_num, k))
    data['x_train_graph_inv'] = np.zeros((train_size_new, time_train_span, node_num, k))

    data['y_train_feature'] = np.zeros((train_size_new, time_predict_span, node_num, channel_num))
    data['y_val_feature'] = np.zeros((val_size_new, time_predict_span, node_num, channel_num))
    data['y_test_feature'] = np.zeros((test_size_new, time_predict_span, node_num, channel_num))

    data['x_val_feature'] = np.zeros((val_size_new, time_train_span, node_num, channel_num))
    data['x_val_graph'] = np.zeros((val_size_new, time_train_span, node_num, k))
    data['x_val_graph_inv'] = np.zeros((val_size_new, time_train_span, node_num, k))

    data['x_test_feature'] = np.zeros((test_size_new, time_train_span, node_num, channel_num))
    data['x_test_graph'] = np.zeros((test_size_new, time_train_span, node_num, k))
    data['x_test_graph_inv'] = np.zeros((test_size_new, time_train_span, node_num, k))

    category = ['x_train_feature', 'x_val_feature', 'x_test_feature',
                'x_train_graph', 'x_val_graph', 'x_test_graph',
                'x_train_graph_inv', 'x_val_graph_inv', 'x_test_graph_inv',
                'y_train_feature', 'y_val_feature', 'y_test_feature'
                ]
    for modern in category:

        sample_rid = 0
        end_batch = 0
        start_batch = 0
        for p_id, p in raw_data[modern].items():

            temp_id_map = id_maps[p_id]
            inver_maps = {value: key for key, value in temp_id_map.items()}
            inver_maps.update({0: 0})
            # inver_maps.update({-1: 0})
            if sample_rid == 0:
                if len(mktid_member_net[p_id]) % node_num == 0:
                    end_batch = len(mktid_member_net[p_id]) // node_num
                else:
                    end_batch = len(mktid_member_net[p_id]) // node_num + 1
            else:
                if len(mktid_member_net[p_id]) % node_num == 0:
                    end_batch = start_batch + len(mktid_member_net[p_id]) // node_num
                else:
                    end_batch = start_batch + (len(mktid_member_net[p_id]) // node_num + 1)

            for t in p.keys():
                relative_id = 0
                if 'feature' in modern:
                    temp_fea = raw_data[modern][p_id][t]
                    org_id = list(temp_fea.keys())

                else:
                    temp_adj = raw_data[modern][p_id][t]
                    org_id = list(temp_adj.keys())
                    if len(org_id) == 0:
                        continue
                    nei_list = np.zeros((len(temp_adj), k))
                    for n in temp_adj:
                        temp_nei_list = list(temp_adj[n])
                        if len(temp_nei_list) == 0:
                            nei_list[relative_id][0] = n
                            nei_list[relative_id][1:] = nei_list[relative_id][1:]

                        elif 0 < len(temp_nei_list) < k - 1:
                            nei_list[relative_id][0] = n  # 如果邻居数量少于k，采样所有邻居
                            nei_list[relative_id][1:] = random.choices(temp_nei_list, k=k - 1)
                        else:
                            nei_list[relative_id][0] = n
                            nei_list[relative_id][1:] = np.random.choice(temp_nei_list, size=k - 1,
                                                                         replace=False).tolist()
                        relative_id = relative_id + 1

                    map_nei_list = [[inver_maps[i] for i in row] for row in nei_list]

                relative_node_id = [id % node_num for id in org_id]
                relative_batch_id = [id // node_num + start_batch for id in org_id]

                for i in range(len(org_id)):
                    if channel_num == 1:
                        if 'graph' in modern:
                            data[modern][relative_batch_id[i], t - 1, relative_node_id[i], :] = map_nei_list[i]

                        elif 'y_' in modern:
                            data[modern][relative_batch_id[i], t - 1 - time_train_span, relative_node_id[i], 0] = \
                            temp_fea[org_id[i]]

                        else:
                            data[modern][relative_batch_id[i], t - 1, relative_node_id[i], 0] = temp_fea[org_id[i]]

            sample_rid = sample_rid + 1
            start_batch = end_batch
    scaler = StandardScaler(mean=data['x_train_feature'][..., 0].mean(), std=data['x_train_feature'][..., 0].std())
    data['scaler'] = scaler
    for c in category:
        if 'x_' in c and 'feature' in c:
            data[c][..., 0] = scaler.transform(data[c][..., 0])
    data['train_loader'] = DataLoader_C(data['x_train_feature'], data['x_train_graph'], data['x_train_graph_inv'],
                                        data['y_train_feature'], train_index, batch_size)
    data['val_loader'] = DataLoader_C(data['x_val_feature'], data['x_val_graph'], data['x_val_graph_inv'],
                                      data['y_val_feature'], val_index, batch_size)
    data['test_loader'] = DataLoader_C(data['x_test_feature'], data['x_test_graph'], data['x_test_graph_inv'],
                                       data['y_test_feature'], test_index, batch_size)
    return data


