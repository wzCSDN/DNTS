import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim,
                 use_bias=False, aggr_method="mean"):
        """聚合节点邻居
        Args:
            input_dim: 输入特征的维度
            output_dim: 输出特征的维度
            use_bias: 是否使用偏置 (default: {False})
            aggr_method: 邻居聚合方式 (default: {mean})
        """
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=3)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=3)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=3)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}"
                             .format(self.aggr_method))

        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)

class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation=F.relu,
                 aggr_neighbor_method="mean",
                 aggr_hidden_method="sum"):
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim,
                                             aggr_method=aggr_neighbor_method)
        self.b = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.b)

    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.b)

        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=-1)
        else:
            raise ValueError("Expected sum or concat, got {}"
                             .format(self.aggr_hidden))
        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

    def extra_repr(self):
        output_dim = self.hidden_dim if self.aggr_hidden_method == "sum" else self.hidden_dim * 2
        return 'in_features={}, out_features={}, aggr_hidden_method={}'.format(
            self.input_dim, output_dim, self.aggr_hidden_method)



class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim,num_node):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)
        self.num_node=num_node
        self.gcn = nn.ModuleList()
        self.gcn.append(SageGCN(input_dim, hidden_dim[0]))
        self.Embedding= nn.Embedding(self.num_node+1, self.input_dim)

        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(SageGCN(hidden_dim[index], hidden_dim[index + 1]))
        if self.num_layers>1:
            self.gcn.append(SageGCN(hidden_dim[-2], hidden_dim[-1], activation=None))
        self.reset_parameters()
    def reset_parameters(self):
        init.kaiming_uniform_(self.Embedding.weight)
        self.Embedding.weight.data[0,:]=0
    def forward(self, adj_org):
        input_shape = (adj_org.shape[0], adj_org.shape[1], adj_org.shape[2], adj_org.shape[3])
        flat_a = adj_org.reshape(adj_org.shape[0] * adj_org.shape[1] * adj_org.shape[2] * adj_org.shape[3], 1).long()

        input_x = self.Embedding(flat_a)
        input_x =input_x.reshape(adj_org.shape[0], adj_org.shape[1], adj_org.shape[2], adj_org.shape[3], self.input_dim)
        input_self = input_x[:, :, :, 0, :]
        input_nei = input_x[:, :, :, 1:, :]
        hidden_self=input_self
        hidden_nei = input_nei
        for l in range(self.num_layers):
            gcn = self.gcn[l]
            hidden_self = gcn(hidden_self, hidden_nei)
            hidden_self = torch.relu(hidden_self)
        return hidden_self

    def extra_repr(self):
        return 'in_features={}, num_neighbors_list={}'.format(
            self.input_dim, self.num_neighbors_list
        )
