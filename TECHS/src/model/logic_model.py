import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter

# from model.time_layer import TimeEncode

from model.model_utils import group_max_norm


class GNNModel(torch.nn.Module):
    def __init__(self, emb_dim, gcn_dim, num_rel, act=lambda x: x, max_nodes=50, reduce='sum'):
        super(GNNModel, self).__init__()
        self.emb_dim = emb_dim
        self.num_rel = num_rel
        self.max_nodes = max_nodes
        self.act = act  # 激活函数
        self.reduce = reduce  # sum

        self.zero_rel_emd = self.get_param([1, gcn_dim])  # 空关系表示

        self.W_message = nn.Linear(emb_dim*2, emb_dim)
        # 实例推理
        # self.att_q1 = nn.Linear(emb_dim, emb_dim)
        # self.att_m1 = nn.Linear(emb_dim, emb_dim)
        # self.att_t1 = nn.Linear(emb_dim, emb_dim)
        self.att_1 = nn.Linear(3*emb_dim, 1)

        # 一阶逻辑推理
        self.att_2 = nn.Linear(emb_dim, 1)
        # self.att_2 = nn.Linear(emb_dim, 1)

        self.W_h = nn.Linear(emb_dim+emb_dim, emb_dim)
        self.line_e_ts = nn.Linear(gcn_dim+emb_dim, emb_dim)
        self.line_r_ts = nn.Linear(gcn_dim, emb_dim)
        self.line_time = nn.Linear(emb_dim, emb_dim)

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def forward(self, query_time, query_emd, temp_neighbors_facts, ent_emds, rel_emds,
                hidden_node, hidden_rel, hidden_time, att_agg, time_emds, gru1, batch_size, device, max_nodes):  # X*6 b,n,h,r,t,ts  N*3 X
        b = temp_neighbors_facts[:, 0]  # batch index
        n = temp_neighbors_facts[:, 1]  # 上一层的index
        h = temp_neighbors_facts[:, 2]
        r = temp_neighbors_facts[:, 3]  # 存在-1
        t = temp_neighbors_facts[:, 4]
        ts = temp_neighbors_facts[:, 5]


        hidden_node_pre = hidden_node[n]
        hidden_rel_pre = hidden_rel[n]
        # hidden_time_pre = hidden_time[n]

        rel_emds = torch.cat([self.zero_rel_emd, rel_emds], dim=0)
        h_r = rel_emds[r+1]
        h_t = ent_emds[t]
        # h_ts = time_encoder(ts)
        h_ts = time_emds[ts]

        h_t = torch.cat([h_t, h_ts], dim=-1)
        h_t = self.line_e_ts(h_t)
        h_r = self.line_r_ts(h_r)
        # h_ts = line_time(h_ts)
        h_ts = self.line_time(h_ts)


        message = torch.cat([hidden_node_pre, h_r], dim=-1)
        message = self.W_message(message)

        # att1
        # att_q = self.att_q1(query_emd)[b]
        # att_m = self.att_m1(message)
        # att_t = self.att_t1(h_t)
        att_q = query_emd[b]
        att_m = message
        att_t = h_t
        att_input = torch.cat([att_q, att_m, att_t], dim=-1)
        # att1 = self.att_1(att_input)
        att1 = torch.sigmoid(self.att_1(att_input))

        # att2
        hidden_rel_new, _ = gru1(h_r.unsqueeze(0), hidden_rel_pre.unsqueeze(0))  # 仅仅是关系
        hidden_rel_new = hidden_rel_new.squeeze(0)

        # hidden_time_new, _ = gru2(h_ts.unsqueeze(0), hidden_time_pre.unsqueeze(0))
        # hidden_time_new = hidden_time_new.squeeze(0)
        # hidden_fol = torch.cat([hidden_rel_new, hidden_time_new], dim=-1)  # rel time计算一阶逻辑规则

        hidden_fol = hidden_rel_new

        # att2 = self.att_2(hidden_fol)
        att2 = torch.sigmoid(self.att_2(hidden_fol))
        # att2 = torch.sigmoid(self.att_2(hidden_rel_new))

        att1 = att1.squeeze(1)
        att2 = att2.squeeze(1)



        # 根据att2选择topk  选择一定数量的edge
        attention = (att1 + att2) / 2  # 注意力机制
        # attention = att2
        topk = max_nodes
        new_index = None
        for i in range(batch_size):
            batch_index = b == i
            temp_att = attention[batch_index]
            temp_index = torch.nonzero(batch_index).squeeze(1)  # 在原来tail_nodes中的index
            if temp_att.size(0) > topk:  # 否则不变
                topk_index = torch.topk(temp_att, topk).indices
                temp_index = temp_index[topk_index]
            if i == 0:
                new_index = temp_index
                # new_attention = tmp_att
            else:
                new_index = torch.cat([new_index, temp_index], dim=0)
        temp_neighbors_facts = temp_neighbors_facts[new_index]
        message = message[new_index]
        hidden_rel_new = hidden_rel_new[new_index]

        # hidden_time_new = hidden_time_new[new_index]
        hidden_time_new = None

        att1 = att1[new_index]
        att2 = att2[new_index]
        b = b[new_index]
        n = n[new_index]
        attention = (att1 + att2) / 2  # 注意力机制
        # attention = att2
        if att_agg != None:
            node_att_pre = att_agg[n]
            attention = attention * node_att_pre
        tail_nodes, tail_index = torch.unique(temp_neighbors_facts[:, [0, 4, 5]], dim=0, sorted=True,
                                              return_inverse=True)  # N*3 X
        # 原tail表示
        tail_e = ent_emds[tail_nodes[:, 1]]
        # tail_ts = time_encoder(tail_nodes[:, 2])
        tail_ts = time_emds[tail_nodes[:, 2]]
        tail_emd = torch.cat([tail_e, tail_ts], dim=-1)
        tail_emd = self.line_e_ts(tail_emd)

        attention = group_max_norm(attention, b, device=device)  # X  根据batch softmax
        message = attention.unsqueeze(1) * message
        message_agg = scatter(message, index=tail_index, dim=0, reduce=self.reduce)
        message_agg = torch.cat([message_agg, tail_emd], dim=-1)
        hidden_node_new = self.act(self.W_h(message_agg))
        hidden_rel_new = scatter(attention.unsqueeze(1) * hidden_rel_new, index=tail_index, dim=0, reduce=self.reduce)
        # hidden_time_new = scatter(attention.unsqueeze(1) * hidden_time_new, index=tail_index, dim=0, reduce=self.reduce)

        att_agg_new = scatter(attention, index=tail_index, dim=0, reduce='sum')  # 使用sum



        # attention = (att1 + att2) / 2  # 注意力机制
        # # attention = attention.squeeze(1)
        # if att_agg != None:
        #     node_att_pre = att_agg[n]
        #     attention = attention * node_att_pre
        # # attention = att2
        #
        # # 根据attention选择top k (每层选取个数)
        # topk = max_nodes
        # tail_nodes, tail_index = torch.unique(temp_neighbors_facts[:, [0, 4, 5]], dim=0, sorted=True,
        #                                       return_inverse=True)  # N*3 X
        # # 原tail表示
        # tail_e = ent_emds[tail_nodes[:, 1]]
        # tail_ts = time_encoder(tail_nodes[:, 2])
        # tail_emd = torch.cat([tail_e, tail_ts], dim=-1)
        # tail_emd = self.line_e_ts(tail_emd)
        #
        # attention = group_max_norm(attention, b, device=device)  # X  根据batch softmax
        # message = attention.unsqueeze(1) * message
        # message_agg = scatter(message, index=tail_index, dim=0, reduce=self.reduce)
        # message_agg = torch.cat([message_agg, tail_emd], dim=-1)
        # hidden_node_new = self.act(self.W_h(message_agg))
        # hidden_rel_new = scatter(attention.unsqueeze(1) * hidden_rel_new, index=tail_index, dim=0, reduce=self.reduce)
        # hidden_time_new = scatter(attention.unsqueeze(1) * hidden_time_new, index=tail_index, dim=0, reduce=self.reduce)
        #
        # tail_nodes_index = None
        # tail_nodes_batchs = tail_nodes[:, 0]
        # att_agg = scatter(attention, index=tail_index, dim=0, reduce='sum')  # 使用sum
        # for i in range(batch_size):
        #     batch_index = tail_nodes_batchs == i
        #     temp_att = att_agg[batch_index]
        #     temp_index = torch.nonzero(batch_index).squeeze(1)  # 在原来tail_nodes中的index
        #     if temp_att.size(0) > topk:  # 否则不变
        #         topk_index = torch.topk(temp_att, topk).indices
        #         temp_index = temp_index[topk_index]
        #     # tmp_att = F.softmax(temp_att, dim=0)  # 采样后、进行归一化
        #     # tmp_att = temp_att
        #     if i == 0:
        #         tail_nodes_index = temp_index
        #         # new_attention = tmp_att
        #     else:
        #         tail_nodes_index = torch.cat([tail_nodes_index, temp_index], dim=0)
        #         # new_attention = torch.cat([new_attention, tmp_att], dim=0)
        # att_agg_new = att_agg[tail_nodes_index]
        # tail_nodes = tail_nodes[tail_nodes_index]
        # hidden_node_new = hidden_node_new[tail_nodes_index]
        # hidden_rel_new = hidden_rel_new[tail_nodes_index]
        # hidden_time_new = hidden_time_new[tail_nodes_index]

        return tail_nodes, hidden_node_new, hidden_rel_new, hidden_time_new, att_agg_new  # N*D 新的节点表示


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

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNModel(emb_dim, gcn_dim, num_rel, act=self.act, max_nodes=self.max_nodes, reduce=self.reduce))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        self.dropout = nn.Dropout(dropout)

        self.gru1 = nn.GRU(self.emb_dim, self.emb_dim)  # 关系
        # self.gru2 = nn.GRU(self.emb_dim, self.emb_dim)  # 时间

        self.line_ent = nn.Linear(gcn_dim, emb_dim)
        self.line_rel = nn.Linear(gcn_dim, emb_dim)
        self.line_time = nn.Linear(gcn_dim, emb_dim)

        # self.line_e_ts = nn.Linear(emb_dim * 3, emb_dim)
        self.line_query = nn.Linear(emb_dim * 3, emb_dim)

        # self.line_ts = nn.Linear(gcn_dim, emb_dim)

        # self.bn = nn.BatchNorm1d(emb_dim*3)
        # self.W_final = nn.Linear(self.emb_dim * 3, 1, bias=False)  # hidden feature -> get score
        self.W_final = nn.Linear(self.emb_dim, 1, bias=False)  # hidden feature -> get score

    def forward(self, ent_emds, rel_emds, time_emds, query_head, query_rel, query_time, dataset, device, args):
        batch_size = query_head.size(0)
        sample_nodes = args.sample_nodes
        max_nodes = args.max_nodes
        sample_method = args.sample_method
        sample_ratio = args.sample_ratio
        score_method = args.score_method
        loss = args.loss

        time_emds = self.line_time(time_emds)
        query_time_emd = time_emds[query_time]  # 调整维度
        query_head_emd = ent_emds[query_head]
        query_rel_emd = rel_emds[query_rel]

        query_head_emd = self.line_ent(query_head_emd)
        query_rel_emd = self.line_rel(query_rel_emd)
        # query_time_emd = self.line_time(query_time_emd)
        query_time_emd = query_time_emd
        query_emd = self.line_query(torch.cat([query_head_emd, query_rel_emd, query_time_emd], dim=-1))

        hidden_node = query_head_emd  # 节点表示
        hidden_rel = query_rel_emd  # 关系表示
        hidden_time = query_time_emd  # 时间表示
        tail_nodes = None  # N*3
        att_agg = None
        # q_rel_emd = None
        # print(query_head, query_time, mask_index_tensor, device)
        for i in range(self.n_layer):
            if i == 0:
                # X*5 b,h,r,t,ts
                temp_neighbors_facts = dataset.load_neighbors4model_1(query_head, query_time, device, sample_method, sample_nodes, sample_ratio)
            else:
                temp_neighbors_facts = dataset.load_neighbors4model_2(tail_nodes, query_time, device, sample_method, sample_nodes, sample_ratio)

            tail_nodes, hidden_node, hidden_rel, hidden_time, att_agg = self.gnn_layers[i](query_time, query_emd, temp_neighbors_facts, ent_emds, rel_emds, hidden_node, hidden_rel,
                                        hidden_time, att_agg, time_emds, self.gru1, batch_size, device, max_nodes)  # N*D
            hidden_node = self.dropout(hidden_node)
            hidden_rel = self.dropout(hidden_rel)
            # hidden_time = self.dropout(hidden_time)
            query_time = query_time[tail_nodes[:, 0]]  # batch index

        if args.time_score == 1:
            time_scores = (tail_nodes[:,2] - query_time) * 0.1
            time_scores = torch.exp(time_scores)
            att_agg += time_scores
            # print(time_scores)

        if score_method == 'emd':
            score_emd = self.W_final(hidden_node).squeeze(1)  # 根据嵌入向量计算
            score_emd = torch.relu(score_emd)
            att_agg = score_emd
        elif score_method == 'att':
            att_agg = att_agg
        elif score_method == 'both':
            score_emd = self.W_final(hidden_node).squeeze(1)  # 根据嵌入向量计算
            score_emd = torch.relu(score_emd)
            att_agg = score_emd + att_agg

        tail_ents, tail_index = torch.unique(tail_nodes[:, [0, 1]], dim=0, sorted=True, return_inverse=True)  # X*2 N
        scores = scatter(att_agg, index=tail_index, dim=0, reduce=self.reduce)  # X
        if loss == 'bce':  # 只有BCE进行归一化
            scores = group_max_norm(scores, tail_ents[:, 0], device=device)  # X


        return scores, tail_ents



