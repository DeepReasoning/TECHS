import numpy as np

import torch


def segment_max(logits, segment_ids, device, keep_length=True):  # 每个类的最大值
    n_logits = len(segment_ids)
    mask = segment_ids[1:] != segment_ids[:-1]  # 要求sorted
    seg_head_ids = np.concatenate([np.array([0]),
                                   np.arange(1, n_logits)[mask],
                                   np.array([n_logits])]).astype(np.int64)
    if keep_length:
        seg_max_ind = torch.cat([(torch.argmax(logits[torch.arange(head, tail).to(torch.int64).to(device)]) + torch.tensor([head]).to(torch.int64).to(device)).repeat(tail - head) for head, tail in zip(seg_head_ids[:-1], seg_head_ids[1:])])
    else:
        seg_max_ind = torch.cat([torch.argmax(logits[torch.arange(head, tail).to(torch.int64).to(device)]) + torch.tensor([head]).to(torch.int64).to(device) for head, tail in zip(seg_head_ids[:-1], seg_head_ids[1:])])
    return logits[seg_max_ind]

def group_max_norm(logits, group_ids, device=-1):  # 组内softmax
    '''
    根据group_ids进行norm(除以每组最大值)
    :param logits: N
    :param group_ids: N 从0开始
    :param device:
    :return: N
    '''
    # if device == -1:
    #     device = torch.device('cpu')
    # else:
    #     device = torch.device('cuda:{}'.format(device))

    length = logits.size()[0]
    N_segment = max(group_ids) + 1
    group_ids = group_ids.data.cpu().numpy()

    # 计算softmax
    # logits = logits - segment_max(logits, group_ids, device, keep_length=True)
    # logits = torch.exp(logits)

    sparse_index = torch.LongTensor(np.vstack([group_ids, np.arange(length)]))
    sparse_value = torch.ones(length, dtype=torch.float)
    trans_matrix_sparse_th = torch.sparse.FloatTensor(sparse_index, sparse_value, torch.Size([N_segment, length])).to(device)
    norm_den = torch.sparse.mm(trans_matrix_sparse_th, logits.unsqueeze(1))

    sparse_index = torch.LongTensor(np.vstack([np.arange(length), group_ids]))
    sparse_value = torch.ones(length, dtype=torch.float)
    trans_matrix_sparse_th = torch.sparse.FloatTensor(sparse_index, sparse_value, torch.Size([length, N_segment])).to(device)
    den = torch.squeeze(torch.sparse.mm(trans_matrix_sparse_th, norm_den))
    res = logits / den
    res[res != res] = 0  # res != res inidcates where NaNs (0/0) are
    return res


# logits = torch.tensor([1.5, 0.5, 0.2, 0.3, 0.5])
# segment_ids = torch.tensor([0, 0, 1, 1, 1])
# print(group_max_norm(logits, segment_ids))