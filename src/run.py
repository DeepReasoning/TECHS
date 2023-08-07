from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_

from utils.dataset import tKGDataset, tkg_collate_fn, tkg_collate_fn_valid
from utils.data_process import cal_ranks, cal_performance
# from CompGCN import CompGCN_DistMult, CompGCN_ConvE
from model import CompGCN_DistMult, GLogicalLayer, TimeEncode, TGCN

# torch.cuda.set_device(0)
# device = torch.device('cuda:2')
device = torch.device('cpu')

class TLoGN(torch.nn.Module):  # 整体网络
    def __init__(self, n_ent, n_rel, emb_dim, dropout=0.1, max_nodes=10, act=lambda x: x):
        super(TLoGN, self).__init__()
        self.emb_dim = emb_dim
        self.max_nodes = max_nodes
        self.act = act  # 激活函数 torch.tanh
        self.dropout = dropout

        self.time_encoder = TimeEncode(emb_dim)  # 整体的时间信息建模
        self.gcn_layer = TGCN(n_ent, n_rel, emb_dim, num_base=50, n_layer=2, conv_bias=True, gcn_drop=dropout, opn='mult')
        self.logic_layer = GLogicalLayer(emb_dim, n_ent, n_layer=3, dropout=dropout, max_nodes=self.max_nodes)

    def forward(self, tkg_temp, query_head, query_rel, query_time, mask_index, query_answer, dataset):
        ent_emds, rel_emds = self.gcn_layer(tkg_temp, mask_index, self.time_encoder)
        # B*M B*M B*M B*M
        emd_scores, condidate_atts, labels, masks = self.logic_layer(ent_emds, rel_emds, query_head, query_rel,
                                                                     query_time, query_answer, dataset, self.time_encoder)
        return emd_scores, condidate_atts, labels, masks


class Runner(object):
    def __init__(self, params=None):
        data_dir = '../data/icews14'

        self.loss = nn.BCELoss()

        self.tkg_dataset_train = tKGDataset(data_dir, data_type='train')
        self.tkg_dataset_valid = tKGDataset(data_dir, data_type='valid')
        self.tkg_dataset_test = tKGDataset(data_dir, data_type='test')
        self.tkg_dataloder_train = DataLoader(self.tkg_dataset_train, shuffle=False, num_workers=0, collate_fn=tkg_collate_fn)
        self.tkg_dataloder_valid = DataLoader(self.tkg_dataset_valid, shuffle=False, num_workers=0, collate_fn=tkg_collate_fn_valid)
        self.tkg_dataloder_test = DataLoader(self.tkg_dataset_test, shuffle=False, num_workers=0, collate_fn=tkg_collate_fn_valid)
        self.model = CompGCN_DistMult(num_ent=self.tkg_dataset_train.n_ent, num_rel=self.tkg_dataset_train.n_rel, num_base=50, num_time=365,
                                 init_dim=200, gcn_dim=200, embed_dim=200,
                                 n_layer=2, edge_type=None, edge_norm=None,
                                 bias=True, gcn_drop=0.1, opn='mult', hid_drop=0.1).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.00001)

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def train_epoch(self):
        self.model.train()
        losses = []
        for item in tqdm(self.tkg_dataloder_train):
            query_info, mask_index, targets, tkg_temp = item  # B*4、E edge数量、B*N 实体one hot

            query_info = query_info.to(device)
            mask_index = mask_index.to(device)
            targets = targets.to(device)
            tkg_temp = tkg_temp.to(device)

            # print(query_info.size(), targets.sum(dim=1))
            query_head = query_info[:, 0]
            query_relation = query_info[:, 1]
            query_tail = query_info[:, 2]
            query_time = query_info[:, 3]

            preds = self.model(tkg_temp, query_head, query_relation, query_time[0], mask_index)  # B*E
            # preds = torch.clamp(preds, min=1e-3, max=1 - 1e-3)  # 截断 防止nan
            loss = self.calc_loss(preds, targets)

            # if torch.isnan(loss):
            #     print(preds, targets)

            # print(targets.size())
            # print(preds.size())
            self.optimizer.zero_grad()
            loss.backward()

            # clip_grad_norm_(self.model.parameters(), max_norm=5)  # 梯度截断

            self.optimizer.step()
            losses.append(loss.item())

        # print(losses)
        loss = np.mean(losses)
        print(loss)
        return loss

    def eval(self, mode='valid'):
        self.model.eval()
        eval_loader = None
        eval_dataset = None
        if mode == 'valid':
            eval_loader = self.tkg_dataloder_valid
            eval_dataset = self.tkg_dataset_valid
        elif mode == 'test':
            eval_loader = self.tkg_dataloder_test
            eval_dataset = self.tkg_dataset_test

        ranking = []
        for item in tqdm(eval_loader):
            query_info, mask_index, targets, tkg_temp = item

            query_info = query_info.to(device)
            mask_index = mask_index.to(device)
            targets = targets.to(device)
            tkg_temp = tkg_temp.to(device)

            query_head = query_info[:, 0]
            query_relation = query_info[:, 1]
            query_tail = query_info[:, 2]
            query_time = query_info[:, 3]

            preds = self.model(tkg_temp, query_head, query_relation, query_time[0])  # B*E

            scores = preds.data.cpu().numpy()  # 所有实体
            labels = targets.data.cpu().numpy()
            query_info = query_info.data.cpu().numpy()
            filters = []
            for i in range(len(labels)):
                filt = eval_dataset.filters[(query_info[i,0], query_info[i,1], query_info[i,3])]  # filter entity列表
                filt_1hot = np.zeros((eval_dataset.n_ent,))
                filt_1hot[np.array(filt)] = 1  # filter的使用1填充
                filters.append(filt_1hot)

            filters = np.array(filters)  # B×E
            ranks = cal_ranks(scores, labels, filters)
            ranking += ranks
        ranking = np.array(ranking)
        t_mrr, t_h1, t_h3, t_h10 = cal_performance(ranking)

        out_str = ''
        if mode == 'valid':
            out_str = '[VALID] MRR:%.4f H@1:%.4f H@3:%.4f H@10:%.4f\n' % (t_mrr, t_h1, t_h3, t_h10)
        elif mode == 'test':
            out_str = '[TEST] MRR:%.4f H@1:%.4f H@3:%.4f H@10:%.4f\n' % (t_mrr, t_h1, t_h3, t_h10)
        print(out_str)
        return t_mrr, t_h1, t_h3, t_h10









if __name__ == '__main__':
    runner = Runner()
    train_epochs = 500
    for epoch in range(train_epochs):
        print('Epoch:', epoch+1)
        runner.train_epoch()
        runner.eval(mode='test')






