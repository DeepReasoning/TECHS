import datetime

import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter

from model.time_layer import TimeEncode


class GNNModel(torch.nn.Module):
    def __init__(self, emb_dim, num_rel, act=lambda x: x, reduce='sum'):
        super(GNNModel, self).__init__()
        self.emb_dim = emb_dim
        self.num_rel = num_rel
        self.act = act  # 激活函数
        self.reduce = reduce  # sum

        self.zero_rel_emd = nn.Parameter(torch.zeros(1, emb_dim))
        self.rel_emd = nn.Embedding(2*num_rel+1, emb_dim)

        # 实例推理
        # self.line_q = nn.Linear(3*emb_dim, emb_dim)
        # self.line_a = nn.Linear(3*emb_dim, emb_dim)
        # self.att_line = nn.Linear(3*emb_dim, 1)
        #
        # # 一阶逻辑推理
        # self.att_rule = nn.Linear(emb_dim, 1)
        #
        # self.out_line = nn.Linear(emb_dim, emb_dim)

        self.Ws_attn = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wr_attn = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wqr_attn = nn.Linear(emb_dim, emb_dim)
        self.w_alpha = nn.Linear(emb_dim, 1)

        self.W_h = nn.Linear(emb_dim, emb_dim, bias=False)


    def conbine(self, t1, t2):  # 连个tensor融合 三维
        multi_ = t1 * t2
        sub_ = t1 - t2
        add_ = t1 + t2
        results = torch.cat([multi_, sub_, add_], dim=-1)
        return results

    def forward(self, query_rel, temp_neighbors_facts, tail_nodes, tail_index, hidden, ent_emds, rel_emds, time_encoder, line_time):  # X*6 b,n,h,r,t,ts  N*3 X
        b = temp_neighbors_facts[:, 0]  # batch index
        n = temp_neighbors_facts[:, 1]  # 上一层的index
        h = temp_neighbors_facts[:, 2]
        r = temp_neighbors_facts[:, 3]
        t = temp_neighbors_facts[:, 4]
        ts = temp_neighbors_facts[:, 5]



        h_e = hidden[n]  # 尾结点
        hr = self.rel_emd(r+1)
        q_r = self.rel_emd(query_rel+1)[b]

        message = h_e + hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(h_e) + self.Wr_attn(hr) + self.Wqr_attn(q_r))))
        message = alpha * message

        # rel_emds = torch.cat([self.zero_rel_emd, rel_emds], dim=0)
        # h_e = hidden[n]  # 尾结点
        # q_r = rel_emds[query_rel + 1][b]
        # h_emd = ent_emds[h]
        # t_emd = ent_emds[t]
        # r_emd = rel_emds[r + 1]
        # ts_emd = line_time(time_encoder(ts))
        #
        # message = h_e + h_emd + r_emd + t_emd + ts_emd
        # alpha = torch.sigmoid(self.w_alpha(self.act(self.Ws_attn(message) + self.Wqr_attn(q_r))))
        # message = alpha * message

        message_agg = scatter(message, index=tail_index, dim=0, reduce=self.reduce)
        new_hidden = self.act(self.W_h(message_agg))
        return new_hidden  # N*D 新的节点表示


class GLogicalLayer(torch.nn.Module):  # 规则解码相关
    def __init__(self, num_ent, num_rel, gcn_dim, emb_dim, n_layer=3, dropout=0.1, max_nodes=10, act=lambda x: x, reduce='sum', device=None):
        super(GLogicalLayer, self).__init__()
        self.gcn_dim = gcn_dim
        self.emb_dim = emb_dim
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_layer = n_layer
        self.max_nodes = max_nodes  # 每层 最大  采样数量
        self.act = act
        self.reduce = reduce
        self.device = device


        # self.ent_bias = nn.Parameter(torch.zeros(self.num_ent+1))  # E ? 每个entity作为候选的bias
        # self.empty_ent = self.get_param([1, self.emb_dim])
        # self.empty_rel = self.get_param([1, self.emb_dim])

        # self.time_encoder = TimeEncode(emb_dim)  # 时间信息嵌入
        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNModel(emb_dim, num_rel, act=self.act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        self.dropout = nn.Dropout(dropout)
        self.W_final = nn.Linear(self.emb_dim, 1, bias=False)  # hidden feature -> get score
        self.gru = nn.GRU(self.emb_dim, self.emb_dim)  # 计算(seq, batch, feature)

        # self.ent_emd = nn.Embedding(num_ent, emb_dim)
        self.line_ent = nn.Linear(gcn_dim, emb_dim)
        self.line_rel = nn.Linear(gcn_dim, emb_dim)
        self.line_time = nn.Linear(gcn_dim, emb_dim)

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def forward(self, ent_emds, rel_emds, query_head, query_rel, query_time, mask_index_tensor, dataset, time_encoder, device):
        batch_size = query_head.size(0)

        # q_time = time_encoder(query_time)  # 时间表示  # B*D
        h0 = torch.zeros((1, batch_size, self.emb_dim)).to(device)
        # h0 = self.ent_embed(q_sub).unsqueeze(0)
        hidden = torch.zeros(batch_size, self.emb_dim).to(device)

        # h0 = self.ent_emd(query_head).unsqueeze(0)
        # hidden = self.ent_emd(query_head)

        # ent_emds, rel_emds = self.line_ent(ent_emds), self.line_ent(rel_emds)
        # h0 = ent_emds[query_head].unsqueeze(0)
        # hidden = ent_emds[query_head]

        scores_all = []
        tail_nodes = None  # N*3
        # q_rel_emd = None
        for i in range(self.n_layer):
            if i == 0:
                # X*5 b,h,r,t,ts  N*3 X L
                temp_neighbors_facts, tail_nodes, tail_index, old_nodes_new_index = dataset.load_neighbors4model_1(query_head, query_time,
                                                                                                                   mask_index_tensor, device,self.max_nodes)
            else:
                if mask_index_tensor != None:
                    mask_index_tensor_ = mask_index_tensor[tail_nodes[:, 0]]  # batch index
                    temp_neighbors_facts, tail_nodes, tail_index, old_nodes_new_index = dataset.load_neighbors4model_2(tail_nodes, query_time,
                                                                                                                       mask_index_tensor_, device, self.max_nodes)
                else:
                    temp_neighbors_facts, tail_nodes, tail_index, old_nodes_new_index = dataset.load_neighbors4model_2(tail_nodes, query_time,
                                                                                                                       None, device, self.max_nodes)

            hidden = self.gnn_layers[i](query_rel, temp_neighbors_facts, tail_nodes, tail_index, hidden, ent_emds, rel_emds, time_encoder, self.line_time)  # N*D
            h0 = torch.zeros(1, tail_nodes.size(0), hidden.size(1)).to(device).index_copy_(1, old_nodes_new_index, h0)
            hidden = self.dropout(hidden)
            # hidden, h0 = self.gru(hidden.unsqueeze(0), h0)
            # hidden = hidden.squeeze(0)  # N*D
        # 最后生成候选实体及其label
        tail_ents, tail_index = torch.unique(tail_nodes[:, [0, 1]], dim=0, sorted=True, return_inverse=True)  # X*2 N
        tail_emd = scatter(hidden, index=tail_index, dim=0, reduce=self.reduce)  # X*D

        # scores = torch.sigmoid(self.W_final(tail_emd)).squeeze(-1)  # X
        scores = self.W_final(tail_emd).squeeze(-1)  # X
        # print(scores)

        # if mask_index_tensor == None:
            # print(scores)
            # print(scores.size(0) / 64)
            # print(torch.max(scores))
            # print(tail_nodes)
            # print(tail_ents, tail_index)

        scores_all = torch.zeros((batch_size, self.num_ent)).to(device)
        # scores_all = torch.ones((batch_size, self.num_ent)).to(device) * 0.0001
        # scores_all = torch.ones((batch_size, self.num_ent)).to(device)
        scores_all[[tail_ents[:, 0], tail_ents[:, 1]]] = scores
        return scores_all


if __name__ == '__main__':
    model = GLogicalLayer(10)
    ent_emds, rel_emds = torch.rand([21,10]), torch.rand([11,10])
    query_head = torch.tensor([12,3,4,5])
    query_rel = torch.tensor([3,6,8,10])
    query_time = torch.tensor([0,0,0,0])
    a = model(ent_emds, rel_emds, query_head, query_rel, query_time, None)


