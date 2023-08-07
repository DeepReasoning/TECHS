import math
import torch
from torch import nn

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, batch_size=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        # if batch_size is not None:
        #     return pos_emb[:, None, :].expand(-1, batch_size, -1)
        # else:
        #     return pos_emb[:, None, :]
        return pos_emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0)
        # print(pe.size())
        pe = nn.Parameter(pe, requires_grad=False)
        self.register_buffer('pe', pe)

    def forward(self, x):
        emds = torch.index_select(self.pe, 0, x)
        return emds


if __name__ == '__main__':
    emder1 = PositionalEmbedding(10)
    emder2 = PositionalEncoding(10)
    # input1 = torch.tensor([1,2,100])
    input2 = torch.tensor([100,2])
    print(emder1(input2))
    print(emder2(input2))

