import numpy as np
import torch
import torch.nn as nn

class TimeEncode(nn.Module):
    def __init__(self, expand_dim, factor=5):
        super(TimeEncode, self).__init__()
        # init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])

        time_dim = expand_dim
        self.factor = factor
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float())  # D
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())  # D

        # self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)

        # torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):  # N
        ts = ts.unsqueeze(1)  # N*1
        map_ts = ts * self.basis_freq.view(1, -1)  # N*D
        map_ts += self.phase.view(1, -1)  # N*D
        harmonic = torch.cos(map_ts)
        return harmonic  # N*D

    def forward1(self, ts):
        # ts: [N, L]
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        ts = ts.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = ts * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic  # self.dense(harmonic)


if __name__ == '__main__':
    emder1 = TimeEncode(100)
    print(emder1.basis_freq.size())
    print(emder1(torch.tensor([1,2,100])))
