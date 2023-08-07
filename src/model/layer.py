import torch
from torch import nn
import dgl
import dgl.function as fn
import numpy as np
from dgl.nn.pytorch.softmax import edge_softmax


class CompGCNCov(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CompGCNCov, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.trans_w = self.get_param([in_channels, out_channels])
        self.loop_w = self.get_param([in_channels, out_channels])
        self.w_rel = self.get_param([in_channels, out_channels])

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.attn_h = self.get_param([in_channels, 1])
        self.attn_t = self.get_param([in_channels, 1])
        self.attn_r = self.get_param([in_channels, 1])
        self.attn_ts = self.get_param([in_channels, 1])  # 时间戳

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape), requires_grad=True)
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def message_func(self, edges):
        rel_emd = edges.data['rel_emd']
        time_emd = edges.data['time_emd']

        head_emd = edges.src['h'] + time_emd
        rel_emd = rel_emd + time_emd

        edge_data = head_emd * rel_emd  # 直接点乘
        msg = torch.matmul(edge_data, self.trans_w)  # E*D
        msg = msg * edges.data['att']  # [E, D] * [E, 1]
        return {'msg': msg}

    def forward(self, g: dgl.DGLGraph, x, rel_repr, time_emds):
        h_att = torch.matmul(x, self.attn_h)  # N1 先计算注意力
        t_att = torch.matmul(x, self.attn_t)
        r_att = torch.matmul(rel_repr, self.attn_r)
        ts_att = torch.matmul(time_emds, self.attn_ts)
        type_ids = g.edata['type']  # 边的类型
        type_e_att = r_att[type_ids]
        ts_ids = g.edata['time']
        ts_e_att = ts_att[ts_ids]
        # print(type_ids, torch.min(type_ids), torch.max(type_ids), rel_repr.size())
        # print(ts_ids, torch.min(ts_ids), torch.max(ts_ids), time_emds.size())
        g.ndata.update({'e_h': h_att, 'e_t': t_att, 'h': x})
        g.edata.update({'e_r': type_e_att, 'e_ts': ts_e_att})
        def edge_attention(edges):
            return {'e': self.leaky_relu(edges.src['e_h'] - edges.dst['e_t']
                                         + edges.data['e_r'] + edges.data['e_ts'])}
        g.apply_edges(edge_attention)
        attations = g.edata.pop('e')  # E1
        g.edata['att'] = edge_softmax(g, attations)  # attention

        g.edata['rel_emd'] = rel_repr[type_ids]
        g.edata['time_emd'] = time_emds[ts_ids]

        g.update_all(self.message_func, fn.sum(msg='msg', out='h'))
        x = g.ndata.pop('h') + torch.matmul(x, self.loop_w)

        rel_repr = torch.matmul(rel_repr, self.w_rel)
        return x, rel_repr

