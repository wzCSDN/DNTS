import pickle
import random
import numpy as np
from model.tgcn import *

import argparse
from utils.load_dataset import *
from utils.util import *

import time
from trainer import *
from model.CasTS_3 import *
from model.tgcn import *

from itertools import combinations

parser = argparse.ArgumentParser()
# load

parser.add_argument('--device', type=str, default='cpu', help='')
parser.add_argument('--feature_file_path', type=str, default='data_feature.pkl', help='data path')
parser.add_argument('--graph_file_path', type=str, default='data_graph.pkl', help='data path')

parser.add_argument('--train_ratio', type=float, default='0.6')
parser.add_argument('--test_ratio', type=float, default='0.3')
parser.add_argument('--valid_ratio', type=float, default='0.1')

parser.add_argument('--is_off', type=bool, default=True, help='whether to add off')
parser.add_argument('--is_transpose', type=bool, default=True, help='whether to add inverse adj')
parser.add_argument('--adj_data', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
parser.add_argument('--gcn_true', type=bool, default=False, help='whether to add graph convolution layer')
parser.add_argument('--is_inv', type=bool, default=True, help='whether to add inverse graph')
parser.add_argument('--load_static_feature', type=bool, default=False, help='whether to load static feature')
parser.add_argument('--cl', type=bool, default=False, help='whether to do curriculum learning')  # 是否课程学习

parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=1, help='dilation exponential')

parser.add_argument('--channel_num', type=int, default=1, help='convolution channels')
parser.add_argument('--conv_channels', type=int, default=32, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=32, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=64, help='skip channels')
parser.add_argument('--end_channels', type=int, default=128, help='end channels')
parser.add_argument('--has_item', type=bool, default=False)

parser.add_argument('--clip', type=int, default=1, help='clip')  # 网络参数分开优化

parser.add_argument('--in_dim', type=int, default=128, help='inputs dimension')
# parser.add_argument('--hidden_dim',type=list,default=[64,128],help='inputs dimension')
# parser.add_argument('--layer',type=int,default=2,help='input sequence length')
parser.add_argument('--hidden_dim', type=list, default=[64], help='inputs dimension')
parser.add_argument('--layer', type=int, default=1, help='input sequence length')

parser.add_argument('--seq_in_len', type=int, default=5, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=2, help='output sequence length')
parser.add_argument('--neg_num', type=int, default=10)

# 要预测的长度
parser.add_argument('--max_n', type=int, default=200, help='max node number per epoch')
parser.add_argument('--neibor_size', type=int, default=10, help='max node number per epoch')
parser.add_argument('--layers', type=int, default=3, help='number of layers')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--step_size1', type=int, default=2500, help='step_size')

parser.add_argument('--epochs', type=int, default=10, help='')
parser.add_argument('--print_every', type=int, default=5, help='')
parser.add_argument('--seed', type=int, default=101, help='random seed')
parser.add_argument('--save', type=str, default='./save/', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')

parser.add_argument('--runs', type=int, default=1, help='number of runs')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args = parser.parse_args()

torch.set_num_threads(3)
seed_everything(args.seed)

graph_file_path = 'data/mktid_offspring.pkl'
graph_file_inv_path = 'data/mktid_offspring_transpose.pkl'
sales_file_path = 'data/mktid_self_sales_process.pkl'
att_file_path = 'data/mktid_propagete_sales.pkl'
# if not os.path.exists('data/clean_data_1.pkl'):
if not os.path.exists('data/id_mapped_off_inv_noi.pkl'):
    with open(graph_file_path, 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        old_graph_dict = pickle.load(pkl_file)
    with open(att_file_path, 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        old_att_dict = pickle.load(pkl_file)
    with open(sales_file_path, 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        old_sales_dict = pickle.load(pkl_file)
    with open(graph_file_inv_path, 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        old_graph_inv_dict = pickle.load(pkl_file)
    # max_key=get_key(old_att_dict,old_sales_dict,old_graph_dict,old_graph_inv_dict,args.has_item)
    # import pdb
    # pdb.set_trace()
    off_dict, off_trans_dict, sales_dict, att_dict, id_maps, max_key, max_len, max_off_len = id_map_cas(
        old_att_dict, old_sales_dict, old_graph_dict, old_graph_inv_dict, args.has_item)
else:
    with open("data/id_mapped_off_noi.pkl", 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        off_dict = pickle.load(pkl_file)
    with open("data/id_mapped_off_inv_noi.pkl", 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        off_trans_dict = pickle.load(pkl_file)
    with open('data/id_mapped_sales_noi.pkl', 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        sales_dict = pickle.load(pkl_file)
    with open('data/id_map_noi.pkl', 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        id_maps = pickle.load(pkl_file)
    with open('data/id_mapped_att_noi.pkl', 'rb') as pkl_file:
        # 使用pickle的dump方法将字典写入文件
        att_dict = pickle.load(pkl_file)

with open('data/mktid_member_net.pkl', 'rb') as pkl_file:
    # 使用pickle的dump方法将字典写入文件
    mktid_member_net = pickle.load(pkl_file)
# with open('data/mktid_offspring_transpose.pkl', 'rb') as pkl_file:
#     # 使用pickle的dump方法将字典写入文件
#     off_trans_dict=pickle.load(pkl_file)
# with open('data/mktid_offspring.pkl', 'rb') as pkl_file:
#     # 使用pickle的dump方法将字典写入文件
#     off_dict=pickle.load(pkl_file)

with open(graph_file_path, 'rb') as pkl_file:
    # 使用pickle的dump方法将字典写入文件
    old_graph_dict = pickle.load(pkl_file)

raw_data, train_size, val_size, test_size = load_cas_dataset(att_dict, sales_dict, off_dict, off_trans_dict,
                                                             args.seq_in_len, args.seq_out_len, args.train_ratio,
                                                             args.valid_ratio, args.test_ratio)

with open('data/clean_data_1.pkl', 'wb') as pkl_file:
    # 使用pickle的dump方法将字典写入文件
    pickle.dump(raw_data, pkl_file)

# else:
#     with open('data/clean_data_1.pkl', 'rb') as pkl_file:
#         # 使用pickle的dump方法将字典写入文件
#         raw_data = pickle.load(pkl_file)
#     with open("data/id_mapped_off_noi.pkl", 'rb') as pkl_file:
#         # 使用pickle的dump方法将字典写入文件
#         off_dict = pickle.load(pkl_file)
#     with open('data/mktid_member_net.pkl', 'rb') as pkl_file:
#         # 使用pickle的dump方法将字典写入文件
#         mktid_member_net = pickle.load(pkl_file)
#     with open('data/id_map_noi.pkl', 'rb') as pkl_file:
#         # 使用pickle的dump方法将字典写入文件
#         id_maps = pickle.load(pkl_file)
#     train_size = 600
#     val_size = 100
#     test_size = 300

num_node = args.max_n
channel_num = 1

# if not os.path.exists('data/grid__data_loader.pkl'):
#     data_loader = generate_grid_loader(data, args.seq_in_len, args.seq_out_len, args.max_n, channel_num,
#                                        args.batch_size)
# else:
#     with open('data/grid__data_loader.pkl', 'rb') as pkl_file:
#         # 使用pickle的dump方法将字典写入文件
#         data_loader = pickle.load(pkl_file)
max_key = 99999
max_len = 1086
max_off_len = 276
item_num = len(mktid_member_net)
num_node = args.max_n
channel_num = 1
device = args.device
model = CasTS(args.hidden_dim, args.gcn_true, max_key, max_len,
              args.device,
              dropout=args.dropout,
              input_dim=args.in_dim,
              dilation_exponential=args.dilation_exponential,
              conv_channels=args.conv_channels, residual_channels=args.residual_channels,
              skip_channels=args.skip_channels, end_channels=args.end_channels,
              time_train_len=args.seq_in_len, channel_num=args.channel_num,
              time_predict_len=args.seq_out_len,
              layers=args.layers, propalpha=None, tanhalpha=None, is_inv=args.is_inv,
              layer_norm_affline=True).to(device)
out_dim = model.out_dim
fc_model = FC(out_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# trainer = Trainer_our(model, args.learning_rate, args.weight_decay, args.clip,
#                       args.step_size1, args.seq_out_len,
#                       args.device, args.cl)
print("start training...", flush=True)
train_time = []
minl = 1e5

out_dim, margin, time_pre = model.get_values()
# model = nn.DataParallel(model)
# out_dim,h_list,margin,time_pre=model.module.get_values()
data_loader = generate_cas_loader(raw_data, args.seq_in_len, args.seq_out_len, args.max_n, channel_num,args.batch_size, args.neibor_size,id_maps,item_num)

scaler = data_loader['scaler']
# #
# for epoch in range(1, args.epochs + 1):
#     if epoch % 1 == 0:
#         data_loader = generate_cas_loader(raw_data, args.seq_in_len, args.seq_out_len, args.max_n, channel_num,
#                                           args.batch_size, args.neibor_size, id_maps, item_num)
#
#         scaler = data_loader['scaler']
#     loss_all = 0
#     mae_all = 0
#     mape_all = 0
#     rmse_all = 0
#     for iter, (x_feature, x_graph, x_graph_inv, y_feature, index) in enumerate(
#             data_loader['train_loader'].get_iterator()):
#         model.train()
#
#         if iter % 50 == 0:
#             optimizer.zero_grad()
#             loss_all = 0
#             mape_all = 0
#             rmse_all = 0
#             mae_all = 0
#             W_trans = []
#             time_span = [i for i in range(args.seq_in_len + 1, args.seq_in_len + args.seq_out_len + 1)]
#             for cur_i in range(len(time_span)):
#                 h_list = model.h_list
#                 h_act_list = model.h_act_list
#                 I = torch.eye(out_dim)
#                 h = h_list[cur_i]
#                 h = torch.nn.functional.normalize(h, p=2.0, dim=-1, eps=1e-12, out=None)
#                 h_uns = h.unsqueeze(1)
#                 W_trans.append((I - 2 * h_uns @ h_uns.t()))
#
#         x_feature = torch.Tensor(x_feature).to(device)
#
#         x_graph = torch.Tensor(x_graph).to(device)
#         x_graph_inv = torch.Tensor(x_graph_inv).to(device)
#         n = x_feature.shape[0]
#         cur_node_num = len(mktid_member_net[index]) + 1 #没0的
#         y_feature = torch.Tensor(y_feature)
#         label = y_feature.squeeze(-1).permute(1, 0, 2).reshape(time_pre, -1)[:, :cur_node_num]
#         neg_num = args.neg_num
#         # time_span = [i for i in range(args.seq_in_len + 1, args.seq_in_len + args.seq_out_len + 1)]
#         id_mapping = id_maps[index]
#         node_set = set(id_mapping.values())
#         # node_set = set(id_mapping.keys())
#         h_trans, grid_pred = model(x_feature, x_graph, x_graph_inv, cur_node_num)
#
#         grid_pred = scaler.inverse_transform(grid_pred)  #执行逆变换
#
#         # grid_pred = torch.Tensor(grid_pred)
#         # h_trans = torch.Tensor(h_trans,dtype=torch.float32)
#
#         grid_preds = grid_pred.squeeze(-1).permute(1, 0, 2)
#
#         grid_x = grid_preds.reshape(time_pre, -1)[:, :cur_node_num]
#         h_trans = h_trans.reshape(-1, out_dim)[:cur_node_num, :]
#         h_trans = torch.nn.functional.normalize(h_trans, p=2.0, dim=-1, eps=1e-12, out=None)
#         label_y = torch.zeros_like(label)
#         label_y[label != 0] = 1
#
#         loss_off = 0
#         loss_leaf = 0
#         i = 0
#         temp_y = {}
#         for cur_t in time_span:
#
#             temp_adj = off_dict[index][cur_t]
#             y_leaf = torch.zeros((cur_node_num))
#             is_leaf_index = [k for i, (k, v) in enumerate(temp_adj.items()) if len(v) != 0]
#             is_off = torch.zeros((cur_node_num, cur_node_num))
#             pairs_pos = np.array([(k, v) for k, vs in temp_adj.items() for v in vs])
#             if len(is_leaf_index) == 0:
#                 loss_offc_cur = 0
#
#             else:
#                 y_leaf=label_y[i]
#                 # y_leaf[is_leaf_index] = 1
#
#                 is_off[pairs_pos[:, 0], pairs_pos[:, 1]] = 1
#
#             # W_trans=W_trans.to('cuda')
#
#             # loss_off = self.triplet_loss(emb_self, emb_pos, emb_neg)
#             # is_leaf_fc=is_leaf_fc.to('cpu')
#             # criterionloss_leaf_cur.to('cpu')
#
#             leaf_logits = model.fc(h_trans.float())
#
#             # leaf_logits =gumbel_softmax_sample(leaf_logits,temperature=0.5)
#             loss_leaf_cur = criterion(leaf_logits.squeeze(-1), y_leaf)
#             leaf_logits=torch.where(leaf_logits < 0.5, torch.zeros_like(leaf_logits), leaf_logits)
#
#             # loss_leaf_cur=0
#             loss_leaf = loss_leaf + loss_leaf_cur
#
#             h_s = torch.matmul(h_trans.double(), W_trans[i].double())  # 367，128
#
#             score_all = torch.einsum('ab,bd->ad', [h_s, h_s.permute(1, 0)])  # 367，367
#
#             score_all = score_all.fill_diagonal_(0)
#
#             loss_off_cur = focal_loss_new(score_all, is_off)
#             loss_off = loss_off + loss_off_cur
#
#             score_all = gumbel_softmax_sample(score_all, temperature=0.5)
#
#             grid_data = grid_x[i].unsqueeze(1)
#             h_act=h_act_list[i]
#             h_act = torch.matmul(h_trans.double(), h_act.double())  # 367，128
#             act_all = torch.sigmoid(torch.einsum('ab,bd->ad', [h_act, h_s.permute(1, 0)]))
#             score_all = score_all * act_all
#             final_sale_grid =  grid_data
#
#             # final_sale_grid = leaf_logits * grid_data
#             final_sale_grid = final_sale_grid.t().repeat_interleave(cur_node_num, dim=0)
#             leaf_logits = leaf_logits.squeeze()
#             att_pre_cur = (leaf_logits *torch.sum(score_all * final_sale_grid, 1).squeeze()).unsqueeze(0)
#             if i == 0:
#                 att_pre = att_pre_cur
#             else:
#                 att_pre = torch.concat((att_pre, att_pre_cur), dim=0)
#             i = i + 1
#         # if iter!=0 and iter%150==0:
#         #     import pdb
#         #     pdb.set_trace()
#         loss_leaf = (loss_leaf) / time_pre
#         loss_off = (loss_off) / time_pre
#         loss_mae = masked_mae_loss(att_pre, label, 0.0).item()
#         loss_cur = loss_leaf + loss_off + loss_mae
#         # loss_cur = torch.tensor(loss_mae, requires_grad=True)
#         mae_cur=masked_mae(att_pre, label, 0.0).item()
#         mape_cur = masked_mape(att_pre, label, 0.0).item()
#         rmse_cur = masked_rmse(att_pre, label, 0.0).item()
#
#         loss_all = loss_all + loss_cur
#         mae_all = mae_cur + mae_all
#         mape_all = mape_cur + mape_all
#         rmse_all = rmse_cur + rmse_all
#         if iter != 0 and (iter + 1) % (50) == 0:
#             mae_all = mae_all / 50
#             loss_all = loss_all / 50
#             mape_all = mape_all / 50
#             rmse_all = rmse_all / 50
#             loss_all.backward()
#             optimizer.step()
#             log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE:{:.4f},Train MAPE: {:.4f}, Train RMSE: {:.4f}'
#             print(log.format(iter, loss_all, mae_all, mape_all, rmse_all), flush=True)
#             loss_all = 0
#             mae_all = 0
#             mape_all = 0
#             rmse_all = 0
#     torch.save(model.state_dict(), args.save + "exp" + str(args.expid) + "_" + ".pth")
#
#     print(epoch)

model.load_state_dict(torch.load(args.save + "exp" + str(args.expid) + "_" + ".pth"))

mae_list = []
mape_list = []
rmse_list = []

for iter, (x_feature, x_graph, x_graph_inv, y_feature, index) in enumerate(data_loader['test_loader'].get_iterator()):

    x_feature = torch.Tensor(x_feature).to(device)
    x_graph = torch.Tensor(x_graph).to(device)
    x_graph_inv = torch.Tensor(x_graph_inv).to(device)
    n = x_feature.shape[0]
    cur_node_num = len(mktid_member_net[index]) + 1
    y_feature = torch.Tensor(y_feature)

    label = y_feature.squeeze(-1).permute(1, 0, 2).reshape(time_pre, -1)[:, :cur_node_num]
    with torch.no_grad():
        h_trans, grid_pred = model(x_feature, x_graph, x_graph_inv, cur_node_num)
        h_list = model.h_list
        h_act_list = model.h_act_list
    grid_pred = scaler.inverse_transform(grid_pred)  #执行逆变换
    grid_preds = grid_pred.squeeze(-1).permute(1, 0, 2)
    grid_x = grid_preds.reshape(time_pre, -1)[:, :cur_node_num]

    h_trans = h_trans.reshape(-1, out_dim)[:cur_node_num, :]
    h_trans = torch.nn.functional.normalize(h_trans, p=2.0, dim=-1, eps=1e-12, out=None)
    time_span = [i for i in range(args.seq_in_len + 1, args.seq_in_len + args.seq_out_len + 1)]

    I = torch.eye(out_dim)
    i = 0
    temp_y = {}
    for cur_t in time_span:
        label_cur = label[i]
        h = h_list[i]
        h = torch.nn.functional.normalize(h, p=2.0, dim=-1, eps=1e-12, out=None)
        h_uns = h.unsqueeze(1)
        W_trans = (I - 2 * h_uns @ h_uns.t())
        leaf_logits = model.fc(h_trans.float())
        leaf_logits = torch.where(leaf_logits < 0.5, torch.zeros_like(leaf_logits), leaf_logits)
        # leaf_logits = gumbel_softmax_sample(leaf_logits, temperature=0.5)
        h_s = torch.matmul(h_trans.double(), W_trans.double())  # 367，128
        score_all = torch.einsum('ab,bd->ad', [h_s, h_s.permute(1, 0)])  # 367，367
        score_all = score_all.fill_diagonal_(0)
        score_all = gumbel_softmax_sample(score_all, temperature=0.5)
        grid_data = grid_x[i].unsqueeze(1)

        h_act = h_act_list[i]
        h_act = torch.matmul(h_trans.double(), h_act.double())  # 367，128
        act_all = torch.sigmoid(torch.einsum('ab,bd->ad', [h_act, h_s.permute(1, 0)]))
        score_all = score_all * act_all
        final_sale_grid = grid_data

        # final_sale_grid = leaf_logits * grid_data
        final_sale_grid = final_sale_grid.t().repeat_interleave(cur_node_num, dim=0)
        leaf_logits = leaf_logits.squeeze()
        att_pre_cur = (leaf_logits * torch.sum(score_all * final_sale_grid, 1).squeeze()).unsqueeze(0)
        if i == 0:
            att_pre = att_pre_cur
        else:
            att_pre = torch.concat((att_pre, att_pre_cur), dim=0)
        i = i + 1

    mae_cur = masked_mae(att_pre, label, 0.0).item()
    mape_cur = masked_mape(att_pre, label, 0.0).item()
    rmse_cur = masked_rmse(att_pre, label, 0.0).item()
    mae_list.append(mae_cur)
    mape_list.append(mape_cur)
    rmse_list.append(rmse_cur)
log = 'Evaluate best model on test data:Train MAE:{:.4f},Train MAPE: {:.4f}, Train RMSE: {:.4f}'
print(log.format( np.mean(mae_list), np.mean(mape_list), np.mean(rmse_list)))