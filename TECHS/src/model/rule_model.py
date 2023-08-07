import datetime

import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter

from model.time_layer import TimeEncode


class GNNModel(torch.nn.Module):
    def __init__(self, emb_dim, act=lambda x: x, reduce='sum'):
        super(GNNModel, self).__init__()
        self.emb_dim = emb_dim
        self.act = act  # 激活函数
        self.reduce = reduce  # sum

        # 实例推理
        self.line_q = nn.Linear(3*emb_dim, emb_dim)
        self.line_a = nn.Linear(3*emb_dim, emb_dim)
        self.att_line = nn.Linear(3*emb_dim, 1)

        # 一阶逻辑推理
        self.att_rule = nn.Linear(emb_dim, 1)

        self.out_line = nn.Linear(emb_dim, emb_dim)


    def conbine(self, t1, t2):  # 连个tensor融合 三维
        multi_ = t1 * t2
        sub_ = t1 - t2
        add_ = t1 + t2
        results = torch.cat([multi_, sub_, add_], dim=-1)
        return results

    def forward(self, q_head, q_rel, q_time, tail_nodes, tail_index, r_neighbor, t_neighbor, time_neighbor, hidden, tail_emd,
                batch_size, num_nodes=0, node_num2=0, device=None):  # 嵌入表示  第0层
        message = q_head.unsqueeze(1) + r_neighbor + time_neighbor  # B*N*D
        message = message.view(batch_size * num_nodes, -1)  # BN*D

        query_emd = torch.cat([q_head, q_rel, q_time], dim=-1)
        neighbor_emd = torch.cat([r_neighbor, t_neighbor, time_neighbor], dim=-1)
        query_emd = self.line_q(self.act(query_emd)).unsqueeze(1)  # B*1*D
        neighbor_emd = self.line_a(self.act(neighbor_emd))  # B*N*D
        q_a_emd = self.conbine(query_emd, neighbor_emd)  # B*N*3D
        att1 = torch.sigmoid(self.att_line(q_a_emd)).view(batch_size*num_nodes, -1)  # BN*1

        att2 = torch.sigmoid(self.att_rule(hidden))  # BN*1

        att = (att1 + att2) / 2  # BN*1
        message = att * message
        tail_emd = scatter(message, index=tail_index, dim=0, out=tail_emd, reduce=self.reduce)  # sum
        # tail_emd = self.act(self.out_line(tail_emd))  # X*D
        new_hidden = scatter(hidden, index=tail_index, dim=0, reduce=self.reduce)  # X*D

        # att = att.view(batch_size * num_nodes, -1)  # BN*1
        agg_att = scatter(att, index=tail_index, dim=0, reduce=self.reduce).squeeze(1)  # X

        # 生成top k邻居  根据注意力值
        # 采样tail nodes
        nodes_batch = tail_nodes[:, 0]
        nodes_tail = tail_nodes[:, 1]
        nodes_time = tail_nodes[:, 2]

        tail_list = []
        time_list = []
        emd_list = []
        hidden_list = []
        for i in range(batch_size):  # 逐个batch选择topk ent_time pair
            batch_index = nodes_batch == i
            batch_att = agg_att[batch_index]
            item_len = batch_att.size(0)
            if item_len < self.topk:  # item_len会不会为0？
                # indices = torch.arange(0, item_len).to(device)
                # sample_ = torch.randint(0, item_len, (self.topk,)).to(device)  # 导致重复
                # indices = torch.cat([indices, sample_], dim=0)[:self.topk]

                indices = torch.arange(0, item_len).to(device)
                more_len = self.topk - item_len
                new_tail = torch.tensor([-1] * more_len).to(device)
                new_time = torch.tensor([0] * more_len).to(device)
                new_emd = torch.zeros(more_len, self.emb_dim).to(device)

                temp_tail = nodes_tail[batch_index][indices]  # M
                temp_time = nodes_time[batch_index][indices]  # M
                temp_emd = tail_emd[batch_index][indices]  # M*D
                hidden_emd = new_hidden[batch_index][indices]  # M*D

                temp_tail = torch.cat([temp_tail, new_tail], dim=0)
                temp_time = torch.cat([temp_time, new_time], dim=0)
                temp_emd = torch.cat([temp_emd, new_emd], dim=0)
                hidden_emd = torch.cat([hidden_emd, new_emd], dim=0)
            else:
                indices = torch.topk(batch_att, self.topk).indices  # 取一定数量
                temp_tail = nodes_tail[batch_index][indices]  # M
                temp_time = nodes_time[batch_index][indices]  # M
                temp_emd = tail_emd[batch_index][indices]  # M*D
                hidden_emd = new_hidden[batch_index][indices]  # M*D
            tail_list.append(temp_tail)
            time_list.append(temp_time)
            emd_list.append(temp_emd)
            hidden_list.append(hidden_emd)
        tail_list = torch.stack(tail_list, dim=0)  # B*M
        time_list = torch.stack(time_list, dim=0)  # B*M
        new_nodes = torch.stack([tail_list, time_list], dim=-1)  # B*M*2
        tail_emd = torch.stack(emd_list, dim=0)  # B*M*D
        new_hidden = torch.stack(hidden_list, dim=0)  # B*M*D
        tail_emd = self.act(self.out_line(tail_emd))  # B*M*D
        return new_nodes, tail_emd, new_hidden


    def process_layer(self, q_head, q_rel, q_time, new_tail_emd, tail_nodes, tail_index, r_neighbor, t_neighbor, time_neighbor,hidden, tail_emd,
                      batch_size, num_nodes, nodes_num2, device=None):  # 后面层
        message = new_tail_emd.unsqueeze(2) + r_neighbor + time_neighbor  # B*M*N*D
        message = message.view(batch_size * num_nodes * nodes_num2, -1)  # BMN*D

        query_emd = torch.stack([q_head, q_rel, q_time], dim=1).unsqueeze(1)  # B*1*3*D
        query_emd = query_emd + new_tail_emd.unsqueeze(2)  # B*M*1*D -> B*M*3*D
        neighbor_emd = torch.cat([r_neighbor, t_neighbor, time_neighbor], dim=-1)  # B*M*N*3D
        query_emd = self.line_q(query_emd.view(batch_size, num_nodes, -1)).unsqueeze(2)  # B*M*1*D
        neighbor_emd = self.line_a(neighbor_emd).view(batch_size, num_nodes, nodes_num2, -1)  # B*M*N*D
        q_a_emd = self.conbine(query_emd, neighbor_emd)  # B*M*N*3D
        att1 = torch.sigmoid(self.att_line(q_a_emd)).view(batch_size * num_nodes * nodes_num2, 1)  # BMN*1

        att2 = torch.sigmoid(self.att_rule(hidden))  # BMN*D -> BMN*1

        att = (att1 + att2) / 2  # BMN*1
        message = att * message
        tail_emd = scatter(message, index=tail_index, dim=0, out=tail_emd, reduce=self.reduce)  # sum
        # tail_emd = self.act(self.out_line(tail_emd))  # X*D
        new_hidden = scatter(hidden, index=tail_index, dim=0, reduce=self.reduce)  # X*D

        agg_att = scatter(att, index=tail_index, dim=0, reduce=self.reduce).squeeze(1)  # X

        # 生成top k邻居  根据注意力值
        # 采样tail nodes
        nodes_batch = tail_nodes[:, 0]
        nodes_tail = tail_nodes[:, 1]
        nodes_time = tail_nodes[:, 2]

        tail_list = []
        time_list = []
        emd_list = []
        hidden_list = []
        att_list = []
        for i in range(batch_size):  # 逐个batch选择topk ent_time pair
            batch_index = nodes_batch == i
            batch_att = agg_att[batch_index]
            item_len = batch_att.size(0)
            if item_len < self.topk:  # item_len会不会为0？
                # indices = torch.arange(0, item_len).to(device)
                # sample_ = torch.randint(0, item_len, (self.topk,)).to(device)  # 导致重复
                # indices = torch.cat([indices, sample_], dim=0)[:self.topk]

                indices = torch.arange(0, item_len).to(device)
                more_len = self.topk - item_len
                new_tail = torch.tensor([-1] * more_len).to(device)
                new_time = torch.tensor([0] * more_len).to(device)
                new_emd = torch.zeros(more_len, self.emb_dim).to(device)

                temp_tail = nodes_tail[batch_index][indices]  # M
                temp_time = nodes_time[batch_index][indices]  # M
                temp_emd = tail_emd[batch_index][indices]  # M*D
                hidden_emd = new_hidden[batch_index][indices]  # M*D

                temp_tail = torch.cat([temp_tail, new_tail], dim=0)
                temp_time = torch.cat([temp_time, new_time], dim=0)
                temp_emd = torch.cat([temp_emd, new_emd], dim=0)
                hidden_emd = torch.cat([hidden_emd, new_emd], dim=0)
            else:
                indices = torch.topk(batch_att, self.topk).indices  # 取一定数量
                temp_tail = nodes_tail[batch_index][indices]  # M
                temp_time = nodes_time[batch_index][indices]  # M
                temp_emd = tail_emd[batch_index][indices]  # M*D
                hidden_emd = new_hidden[batch_index][indices]  # M*D
            tail_list.append(temp_tail)
            time_list.append(temp_time)
            emd_list.append(temp_emd)
            hidden_list.append(hidden_emd)
            # att_list.append(temp_att)
        tail_list = torch.stack(tail_list, dim=0)  # B*M
        time_list = torch.stack(time_list, dim=0)  # B*M
        new_nodes = torch.stack([tail_list, time_list], dim=-1)  # B*M*2
        tail_emd = torch.stack(emd_list, dim=0)  # B*M*D
        new_hidden = torch.stack(hidden_list, dim=0)  # B*M*D
        tail_emd = self.act(self.out_line(tail_emd))  # B*M*D
        # atts = torch.stack(att_list, dim=0)  # B*M
        return new_nodes, tail_emd, new_hidden


class GLogicalLayer(torch.nn.Module):  # 规则解码相关
    def __init__(self, emb_dim, num_ent, n_layer=3, dropout=0.1, max_nodes=10, act=lambda x: x, reduce='sum'):
        super(GLogicalLayer, self).__init__()
        self.emb_dim = emb_dim
        self.num_ent = num_ent
        self.n_layer = n_layer
        self.max_nodes = max_nodes  # 每层 最大  采样数量
        self.act = act
        self.reduce = reduce

        # self.ent_bias = nn.Parameter(torch.zeros(self.num_ent+1))  # E ? 每个entity作为候选的bias
        self.empty_ent = self.get_param([1, self.emb_dim])
        self.empty_rel = self.get_param([1, self.emb_dim])

        # self.time_encoder = TimeEncode(emb_dim)  # 时间信息嵌入
        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNModel(emb_dim, topk=self.max_nodes, act=self.act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)

        self.dropout = nn.Dropout(dropout)
        # self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)  # hidden feature -> get score
        self.gru = nn.GRU(self.emb_dim, self.emb_dim)  # 计算(seq, batch, feature)

    def out_process(self, all_ent_emd, tail_nodes, tail_emd, batch, num_nodes, device):  # B B*M*2 B*M*D B*M
        all_ent_emd = all_ent_emd.unsqueeze(0)
        return_ent_emd = all_ent_emd.expand(batch, -1, -1)  # 增加B

        batch_tensor = [[i] * num_nodes for i in range(batch)]
        batch_tensor = torch.tensor(batch_tensor).unsqueeze(2).to(device)  # B*M*1
        nodes_info = torch.cat([batch_tensor, tail_nodes], dim=2).view(batch*num_nodes, -1)  # BM*3

        tail_ents, tail_index = torch.unique(nodes_info[:, [0, 1]], dim=0, sorted=True, return_inverse=True)  # X*2 BM
        tail_emd = tail_emd.view(batch*num_nodes, -1)  # BM*D
        tail_emd = scatter(tail_emd, index=tail_index, dim=0, reduce=self.reduce)  # X*D

        # atts = atts.view(batch * num_nodes)  # BM
        # atts = scatter(atts, index=tail_index, dim=0, reduce=self.reduce)  # X

        nodes_batch = tail_ents[:, 0]
        nodes_ent = tail_ents[:, 1]

        # 增强尾结点的表示
        for i in range(batch):  # 选择ent结果
            batch_index = nodes_batch == i
            batch_emd = tail_emd[batch_index]
            batch_ent = nodes_ent[batch_index] + 1
            if self.reduce == 'sum':
                return_ent_emd[i][batch_ent] += batch_emd
            elif self.reduce == 'mean':
                return_ent_emd[i][batch_ent] = (return_ent_emd[i][batch_ent] + batch_emd) / 2
        return return_ent_emd[:,1:,:]  # 去掉第一个新加的

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def forward(self, ent_emds, rel_emds, query_head, query_rel, query_time, mask_index_tensor, dataset, time_encoder, device):
        all_ent_emd = torch.cat([self.empty_ent, ent_emds])  # 实体表示
        all_rel_emd = torch.cat([self.empty_rel, rel_emds])  # 关系表示

        q_head = all_ent_emd[query_head+1]  # B*D
        q_rel = all_rel_emd[query_rel+1]  # B*D
        q_time = time_encoder(query_time)  # 时间表示  # B*D

        batch_size = query_head.size(0)

        hidden = None
        new_nodes = None   # B*M*2
        new_tail_emd = None  # B*M*D
        atts = None  # B*M
        for i in range(self.n_layer):
            if i == 0:
                # X*5 b,h,r,t,ts  N*3 X
                temp_neighbors_facts, tail_nodes, tail_index = dataset.load_neighbors4model_1(query_head,mask_index_tensor,device,self.max_nodes)



                tail_nodes, tail_index, temp_neighbors_r_e_ts, num_nodes = dataset.load_neighbors4model_1(query_head, mask_index_tensor, device)  # X*3 BN BN*4
                r_neighbor = temp_neighbors_r_e_ts[:, 1]
                t_neighbor = temp_neighbors_r_e_ts[:, 2]
                time_neighbor = temp_neighbors_r_e_ts[:, 3]
                r_neighbor = all_rel_emd[r_neighbor+1].view(batch_size, num_nodes, -1)  # B*N*D
                t_neighbor = all_ent_emd[t_neighbor+1].view(batch_size, num_nodes, -1)
                time_neighbor = time_encoder(time_neighbor).view(batch_size, num_nodes, -1)  # B*N*D

                # 关系表示
                h0 = q_rel + q_time  # B*D
                input4rule = r_neighbor + time_neighbor  # B*N*D

                h0 = h0.unsqueeze(1).expand_as(input4rule)  # B*N*D
                h0 = h0.reshape(batch_size*num_nodes, -1).unsqueeze(0)  # 1*BN*D
                input4rule = input4rule.view(batch_size*num_nodes, -1).unsqueeze(0)  # 1*BN*D
                output, hidden = self.gru(input4rule, h0)  # 1*BN*D
                hidden = hidden.squeeze(0)  # BN*D

                tail_emd = tail_nodes[:, 1]
                tail_time = tail_nodes[:, 2]
                tail_emd = all_ent_emd[tail_emd+1]
                tail_time = time_encoder(tail_time)  # X*D
                tail_emd = tail_emd + tail_time  # X*D

                # B*M*2 B*M*D B*M*D
                new_nodes, new_tail_emd, hidden = self.gnn_layers[i](q_head, q_rel, q_time, tail_nodes, tail_index, r_neighbor, t_neighbor, time_neighbor,
                                   hidden, tail_emd, batch_size, num_nodes=num_nodes, device=device)
            else:
                # M是num_nodes
                tail_nodes, tail_index, temp_neighbors_r_e_ts, num_nodes1, num_nodes2 = dataset.load_neighbors4model_2(new_nodes, mask_index_tensor, device)  # X*3 BMN B*M*N*3
                num_nodes = num_nodes1
                nodes_num2 = num_nodes2
                temp_neighbors_r_e_ts = temp_neighbors_r_e_ts.view(batch_size*num_nodes*nodes_num2, 3)  # BMN*3

                r_neighbor = temp_neighbors_r_e_ts[:, 0]  # BMN
                t_neighbor = temp_neighbors_r_e_ts[:, 1]  # BMN
                time_neighbor = temp_neighbors_r_e_ts[:, 2]  # BMN

                r_neighbor = all_rel_emd[r_neighbor + 1].view(batch_size, num_nodes, nodes_num2, -1)  # B*M*N*D
                t_neighbor = all_ent_emd[t_neighbor + 1].view(batch_size, num_nodes, nodes_num2, -1)  # B*M*N*D
                time_neighbor = time_encoder(time_neighbor).view(batch_size, num_nodes, nodes_num2, -1)  # 时间嵌入 B*M*N*D

                # 关系表示
                input4rule = r_neighbor + time_neighbor  # B*M*N*D
                hidden = hidden.unsqueeze(2).expand_as(input4rule)  # B*M*N*D
                hidden = hidden.reshape(batch_size*num_nodes*nodes_num2, -1).unsqueeze(0)  # 1*BMN*D
                input4rule = input4rule.view(batch_size*num_nodes*nodes_num2, -1).unsqueeze(0)  # 1*BMN*D
                output, hidden = self.gru(input4rule, hidden)  # 1*BMN*D
                hidden = hidden.squeeze(0)  # BMN*D

                tail_emd = tail_nodes[:, 1]
                tail_time = tail_nodes[:, 2]
                tail_emd = all_ent_emd[tail_emd + 1]
                tail_time = time_encoder(tail_time)  # X*D
                tail_emd = tail_emd + tail_time  # X*D

                # B*M*2 B*M*D B*M*D
                new_nodes, new_tail_emd, hidden = self.gnn_layers[i].process_layer(
                    q_head, q_rel, q_time, new_tail_emd, tail_nodes, tail_index, r_neighbor, t_neighbor, time_neighbor,
                    hidden, tail_emd, batch_size, num_nodes=num_nodes, nodes_num2=nodes_num2, device=device)

        # 最后生成候选实体及其label
        # B*E*D
        return_ent_emd = self.out_process(all_ent_emd, new_nodes, new_tail_emd, batch=batch_size, num_nodes=self.max_nodes, device=device)
        return return_ent_emd


if __name__ == '__main__':
    model = GLogicalLayer(10)
    ent_emds, rel_emds = torch.rand([21,10]), torch.rand([11,10])
    query_head = torch.tensor([12,3,4,5])
    query_rel = torch.tensor([3,6,8,10])
    query_time = torch.tensor([0,0,0,0])
    a = model(ent_emds, rel_emds, query_head, query_rel, query_time, None)


