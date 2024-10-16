import argparse
import torch
import torch.nn as nn
from .graphsage import *
from .net import *
from .tgcn import  *
from utils.util import *


class CasTS(nn.Module):
    def __init__(self, hidden_dim,gcn_true,num_nodes,max_len, device, static_feat=None, dropout=0.3,
                 input_dim=40, dilation_exponential=1, conv_channels=32,
                 residual_channels=32, skip_channels=64, end_channels=128,
                 time_train_len=5, channel_num=1, time_predict_len=2, layers=None, propalpha=None,
                 tanhalpha=None, is_inv=True,layer_norm_affline=True):
        super(CasTS, self).__init__()
        self.gcn_true = gcn_true
        self.max_len=max_len
        self.time_train_len=time_train_len
        self.time_predict_len=time_predict_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)
        self.num_node = num_nodes
        self.is_inv=is_inv
        self.loss = masked_mae
        if self.is_inv:
            self.embed_2_logit = nn.Linear(self.hidden_dim[-1]*2, self.hidden_dim[-1]*2, bias=True)
            self.out_dim=self.hidden_dim[-1]*2
        else:
            self.embed_2_logit = nn.Linear(self.hidden_dim[-1]*2, self.hidden_dim[-1]*2, bias=True)
            self.out_dim = self.hidden_dim[-1]
        num_nodes=207
        self.t_conv=gtnet(gcn_true, num_nodes,
                      device,
                      dropout=dropout,
                      node_dim=input_dim,
                      dilation_exponential=dilation_exponential,
                      conv_channels=conv_channels, residual_channels=residual_channels,
                      skip_channels=skip_channels, end_channels=end_channels,
                      seq_length=time_train_len, in_dim=channel_num, out_dim=time_predict_len,
                      layers=layers, propalpha=propalpha, tanhalpha=tanhalpha, layer_norm_affline=True)
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
        self.t_gcn=GraphSage(self.input_dim, self.hidden_dim,self.num_node)

        self.is_leaf_fc = nn.Sequential(
            nn.Linear(self.out_dim, 1),
            nn.Sigmoid()
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.h_list = [nn.Parameter(torch.randn(self.out_dim)) for _ in range(self.time_predict_len)]
        self.h_act_list = [nn.Parameter(torch.randn(self.out_dim,self.out_dim)) for _ in range(self.time_predict_len)]

        self.margin=0.5
        self.p=2
        self.gru_cell = TGCNCell(self.out_dim, self.out_dim)
    def activation(self, x):
        # 这里定义RULE激活函数的行为
        # 作为一个示例，假设RULE激活函数是ReLU
        return torch.relu(x)
    def forward(self,x_feature,x_graph,x_graph_inv,y_feature,temp_y,is_leaf_dict,cur_node_num,mode):

        x_feature = x_feature.transpose(1, 3)
        grid_pred = self.t_conv(x_feature)
        loss_off_list=[]
        h = self.t_gcn(x_graph)
        h_inv = self.t_gcn(x_graph_inv)
        h_final = torch.concat((h, h_inv), dim=-1)
        time_span=h_final.shape[1]
        time_pre = len(temp_y)
        hidden_state = torch.zeros(h_final.shape[0], h_final.shape[2],self.out_dim)

        grid_preds = grid_pred.squeeze(-1).permute(1, 0, 2)

        grid_x = grid_preds.reshape(time_pre, -1)[:, :cur_node_num]
        y_feature = torch.Tensor(y_feature)
        # label = y_feature.squeeze(-1).permute(1, 0, 2).reshape(time_pre, -1)[:, :cur_node_num] #2,637
        for i in range(time_span):
            hidden_state = self.gru_cell(h_final[:, i, :,:], hidden_state)
            self.activation(hidden_state)

        h_trans = hidden_state.reshape(-1, self.out_dim)[:cur_node_num, :]
        h_trans = torch.nn.functional.normalize(h_trans, p=2.0, dim=-1, eps=1e-12, out=None)
        loss_off=0
        loss_leaf=0
        I = torch.eye(self.out_dim)
        for i, h in enumerate(self.h_list):
            h = torch.nn.functional.normalize(h, p=2.0, dim=-1, eps=1e-12, out=None)
            h_uns = h.unsqueeze(1)
            W_temp = (I - 2 * h_uns @ h_uns.t())
            if i == 0:
                W_trans = W_temp.unsqueeze(0)
            else:
                W_trans = torch.concat((W_trans, W_temp.unsqueeze(0)), 0)
        if mode=="train":
            relative_id = 0
            for cur_t, cur_y in temp_y.items():
                cur_y = torch.tensor(cur_y) #513*12
                y_leaf=torch.tensor(is_leaf_dict[cur_t])
                if len(cur_y.shape)!=0:
                    self_index = cur_y[:, 0]
                    pos_index = cur_y[:, 1]
                    neg_index = cur_y[:, 2:]
                    emb_self = h_trans.index_select(dim=0, index=self_index).unsqueeze(1)
                    emb_pos = h_trans.index_select(dim=0, index=pos_index).unsqueeze(1)
                    emb_neg = h_trans.index_select(dim=0, index=neg_index.reshape(-1))
                    emb_neg = emb_neg.view(cur_y.shape[0], cur_y.shape[1] - 2, -1)

                    a = torch.matmul(emb_pos.double(), W_trans[relative_id].double())
                    score_neg = 1-torch.einsum('abc,acd->ad', [a, emb_neg.permute(0, 2, 1)])
                    score_pos = 1-torch.einsum('abc,acd->ad', [a, emb_pos.permute(0, 2, 1)])
                    loss_offc_cur =F.relu(score_pos - score_neg + self.margin).mean()
                else:
                    loss_offc_cur =0
                loss_off = loss_off + loss_offc_cur
                # loss_off = self.triplet_loss(emb_self, emb_pos, emb_neg)
                leaf_logits = self.is_leaf_fc(h_trans.float())
                loss_leaf_cur = self.criterion(leaf_logits.squeeze(-1), y_leaf)


                loss_leaf=loss_leaf+loss_leaf_cur
                h_s = torch.matmul(h_trans.double(), W_trans[relative_id].double())  #367，128
                score_all = torch.einsum('ab,bd->ad', [h_s, h_s.permute(1, 0)]) #367，367

                grid_data = grid_x[relative_id].unsqueeze(1)
                final_sale_grid = leaf_logits * grid_data
                final_sale_grid = final_sale_grid.t().repeat_interleave(cur_node_num, dim=0)
                att_pre_cur = torch.sum(score_all * final_sale_grid, 1).unsqueeze(0)
                if relative_id==0:
                    att_pre=att_pre_cur
                else:
                    att_pre=torch.concat((att_pre,att_pre_cur),dim=0)
                relative_id=relative_id+1
        if mode=="train":
            loss_leaf=(loss_leaf)/time_pre
            loss_off = (loss_off)/time_pre
            loss_mae=masked_mae(att_pre, label, 0.0).item()
            loss_all=loss_leaf+loss_off+loss_mae
            mape = masked_mape(att_pre, label, 0.0).item()
            rmse = masked_rmse(att_pre, label, 0.0).item()
            return loss_all, mape, rmse
        else:
            return att_pre
