import torch
from torch import nn
import dgl
from model.layer import CompGCNCov
import torch.nn.functional as F


class TGCN(nn.Module):  # 仅仅用来TKG编码
    def __init__(self, num_ent, num_rel, input_dim, gcn_dim, n_layer, conv_bias=True, gcn_drop=0.1, opn='mult',
                 act=None, device=None):
        super(TGCN, self).__init__()
        self.act = act  # 是否可以修改
        self.num_ent, self.num_rel, self.num_base = num_ent, num_rel, -1

        self.init_dim, self.gcn_dim, self.embed_dim = gcn_dim, gcn_dim, gcn_dim
        self.conv_bias = conv_bias
        self.gcn_drop = gcn_drop
        self.opn = opn
        self.n_layer = n_layer

        self.conv1 = CompGCNCov(self.init_dim, self.gcn_dim)
        self.conv2 = CompGCNCov(self.gcn_dim, self.embed_dim) if n_layer == 2 else None
        self.line_time = nn.Linear(input_dim, gcn_dim)

        self.time_ln = nn.LayerNorm(gcn_dim)
        self.ent_ln1 = nn.LayerNorm(gcn_dim)
        self.rel_ln1 = nn.LayerNorm(gcn_dim)
        self.ent_ln2 = nn.LayerNorm(gcn_dim)
        self.rel_ln2 = nn.LayerNorm(gcn_dim)

        self.drop = nn.Dropout(gcn_drop)

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))  # relu
        return param

    def forward(self, g, time_emds, ent_emds, rel_emds):
        # print('hhh'*100)
        g = g.local_var()

        # e_time = g.edata['time']
        time_emds = self.line_time(time_emds)
        time_emds = self.time_ln(time_emds)
        # g.edata['time_emd'] = e_time_emd

        x, r = ent_emds, rel_emds
        x, r = self.conv1(g, x, r, time_emds)
        x = self.ent_ln1(x)
        x = self.act(x)
        x = self.drop(x)
        r = self.rel_ln1(r)  # 关系
        r = self.act(r)
        r = self.drop(r)

        if self.n_layer == 2:
            x, r = self.conv2(g, x, r, time_emds)
            x = self.ent_ln2(x)
            x = self.act(x)
            x = self.drop(x)
            r = self.rel_ln2(r)  # 关系
            r = self.act(r)
            r = self.drop(r)

        return x, r  # 实体、关系表示
