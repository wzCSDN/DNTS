import argparse
import torch
import torch.nn as nn
from .graphsage import *
class TGCNCell(nn.Module):
    def __init__(self, input_dim: int, out_dim: int):
        super(TGCNCell, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim

        self.GRU_Z = nn.Sequential(
            nn.Linear(self.input_dim+self.out_dim, self.out_dim, bias=True),
            nn.Sigmoid()).double()
        # reset gate.
        self.GRU_R = nn.Sequential(
            nn.Linear(self.input_dim+self.out_dim, self.out_dim, bias=True),
            nn.Sigmoid()).double()
        # new embedding gate.
        self.GRU_H_Tilde = nn.Sequential(
            nn.Linear(self.input_dim+self.out_dim, self.out_dim, bias=True),
            nn.Tanh()).double()


    def forward(self, inputs, hidden_state):
        if inputs.dtype == torch.float:
            inputs = inputs.double()
        if hidden_state.dtype == torch.float:
            hidden_state = hidden_state.double()
        Z = self.GRU_Z(torch.cat([inputs, hidden_state], dim=2))
        R = self.GRU_R(torch.cat([inputs, hidden_state], dim=2))
        H_tilde = self.GRU_H_Tilde(torch.cat([inputs, R * hidden_state], dim=2))
        H_gru = Z * hidden_state + (1 - Z) * H_tilde

        return H_gru

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,num_node,time_train_len,time_test_len):
        super(TGCN, self).__init__()
        self.num_node=num_node
        self.node_in_dim=input_dim
        self.hidden_dim=hidden_dim
        self.out_dim=hidden_dim[-1]
        self.node_num=num_node
        self.len_train=time_train_len
        self.len_test=time_test_len
        self.gcn_cov=GraphSage(self.node_in_dim, self.hidden_dim, self.node_num)
        self.gru_cell=TGCNCell(self.out_dim,self.out_dim)
        self.emb2y = nn.Linear(self.out_dim, self.len_test, bias=True).double()
        # self.emb2logits = nn.Linear(self.hidden_dim, self.num_node, bias=True).double()


    def activation(self, x):
        # 这里定义RULE激活函数的行为
        # 作为一个示例，假设RULE激活函数是ReLU
        return torch.relu(x)

    def forward(self, tx_org):

        input_x=self.gcn_cov(tx_org)
        hidden_state = torch.zeros(tx_org.shape[0], tx_org.shape[2],self.out_dim)

        for i in range(self.len_train):

            hidden_state = self.gru_cell(input_x[:, i, :,:], hidden_state)
            self.activation(hidden_state)
        logits=self.emb2y(hidden_state.double())

        logits = logits.permute(0, 2, 1).unsqueeze(-1)
        # pred = F.log_softmax(logits, dim=-1)
        return logits

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}

