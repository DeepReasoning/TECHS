import random
from os.path import join
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# import dgl

from utils.data_process import build_graph, build_graph_dic, build_graph_snapshots
from utils.data_utils import load_fact, load_json, load_pickle, save_pickle


# from data_process import build_graph, build_graph_snapshots
# from data_utils import load_fact, load_json, load_pickle, save_pickle

# import sys
# sys.getfilesystemencoding = lambda: 'UTF-8'


class tKGDataset:
    def __init__(self, data_dir, data_type='train', logic=False, logic_ratio=0.5, batchj_size=8):  # train valid test
        self.data_dir = data_dir
        self.data_type = data_type
        # self.max_time = -1  # 自己指向自己的时间设置
        self.logic = logic
        self.logic_ratio = logic_ratio  # logic部分用于训练的比例

        self.entity2id = load_json(join(self.data_dir, 'entity2id.json'))
        self.relation2id = load_json(join(self.data_dir, 'relation2id.json'))
        self.ts2id = load_json(join(self.data_dir, 'ts2id.json'))

        self.n_ent = len(self.entity2id)
        self.n_rel = len(self.relation2id)  # 原始rel数量、未添加逆关系
        self.n_ts = len(self.ts2id)  # timestemp数量

        train_facts = load_fact(join(self.data_dir, 'train.txt'))
        valid_facts = load_fact(join(self.data_dir, 'valid.txt'))
        test_facts = load_fact(join(self.data_dir, 'test.txt'))
        self.train_facts = self.map2ids(train_facts, self.entity2id, self.relation2id, self.ts2id)
        self.valid_facts = self.map2ids(valid_facts, self.entity2id, self.relation2id, self.ts2id)
        self.test_facts = self.map2ids(test_facts, self.entity2id, self.relation2id, self.ts2id)

        self.all_facts = np.concatenate([self.train_facts, self.valid_facts, self.test_facts], axis=0)

        self.tKG = build_graph(self.n_ent, self.train_facts)  # 构建训练KG
        # self.tKG_dic = build_graph_dic(self.n_ent, self.all_facts)  # 构建用于训练、测试的图结构dic

        self.Dic_E = self.load_neighbor_dic()

        facts4train = self.all_facts
        # facts4train = np.concatenate([np.array([[-1, -1, -1, 0]]), facts4train], axis=0)  # 时间使用0
        self.train_facts_tensor = torch.tensor(facts4train)  # 训练时用于获取邻居
        print(self.train_facts_tensor.size())

        self.h_r_ts_dic_all = self.get_h_r_ts_dic_all()  # 所有的，用于验证、测试

        if self.logic:
            max_train_time = max(self.train_facts[:, -1])
            start_time = int(max_train_time * (1 - self.logic_ratio))
            train_facts = self.train_facts
            fact_times = train_facts[:, -1]
            train_facts = train_facts[fact_times >= start_time]  # 部分用于训练
            self.train_facts_new = train_facts
            # self.new2old_index = np.nonzero(fact_times >= start_time)[0]
            # print(self.train_facts.shape)
        self.dataset_describe()

    def load_neighbor_dic(self):
        Dic_E = defaultdict(list)  # entity: fact列表 h_fact_dic = defaultdict(list)  # 头实体的邻居fact index
        # for i, fact in enumerate(self.train_facts):
        for i, fact in enumerate(self.all_facts):
            h, r, t, ts = fact
            Dic_E[h].append(i)
        for ent in Dic_E.keys():
            Dic_E[ent] = np.array(Dic_E[ent])
        return Dic_E

    def dataset_describe(self):
        print('===========Dataset Description:===========')
        print('Entity Num:{} Relation Num:{} Time Num:{}'.format(self.n_ent, self.n_rel, self.n_ts))
        print('Train Num:{} Valid Num:{} Test Num:{}'.format(len(self.train_facts), len(self.valid_facts), len(self.test_facts)))

    def map2ids(self, facts, ent2id, rel2id, ts2id, add_rev=True):  # 使用id表示事实四元组、是否增加逆关系
        fact_ids = []
        for item in facts:
            # print(item)
            # print(item)
            h = ent2id[item[0]]
            r = rel2id[item[1]]
            t = ent2id[item[2]]
            ts = ts2id[item[3]]
            fact_ids.append([h, r, t, ts])
        if add_rev:  # 叠加
            for item in facts:
                h = ent2id[item[0]]
                r = rel2id[item[1]]
                t = ent2id[item[2]]
                ts = ts2id[item[3]]
                fact_ids.append([t, r+self.n_rel, h, ts])
        return np.array(fact_ids)

    def __getitem__(self, index_):
        if self.data_type == 'train':
            if self.logic:
                train_facts = self.train_facts_new
            else:
                train_facts = self.train_facts
            h,r,t,ts = train_facts[index_]
            return h, r, t, ts
        elif self.data_type == 'valid':
            h, r, t, ts = self.valid_facts[index_]
            h_r_ts = (h, r, ts)
            all_tails = self.h_r_ts_dic_all[h_r_ts]
            y = np.zeros([self.n_ent], dtype=np.long)
            y[all_tails] = 1
            return h, r, t, ts, y
        elif self.data_type == 'test':  # 目前不考虑mask
            h, r, t, ts = self.test_facts[index_]
            h_r_ts = (h, r, ts)
            all_tails = self.h_r_ts_dic_all[h_r_ts]
            y = np.zeros([self.n_ent], dtype=np.long)
            y[all_tails] = 1
            return h, r, t, ts, y

    def __len__(self):
        # return 1000
        if self.data_type == 'train':
            if self.logic:
                train_facts = self.train_facts_new
            else:
                train_facts = self.train_facts
            return len(train_facts)
        elif self.data_type == 'valid':
            return len(self.valid_facts)
        elif self.data_type == 'test':
            return len(self.test_facts)

    def get_h_r_ts_dic_all(self):  # 训练、测试
        h_r_ts_dic = defaultdict(list)
        for item in self.all_facts:
            h,r,t,ts = item
            h_r_ts_dic[(h,r, ts)].append(t)
        return h_r_ts_dic


    def load_neighbors_by_array(self, _ent, query_time, sample_method, sample_nodes, sample_ratio, mask_2=False, pre_time=None):
        '''
        # time windows
        :param batch_ent: N sample_method 1 2 3所有 前向 后向
        :param query_time: N
        :return:
        '''
        # print(query_time)
        parent_indexs = []
        neighbors = []
        for i, ent in enumerate(_ent):
            tmp_time = query_time[i]
            neighbor_index = self.Dic_E[ent]
            neighbor_ts = self.all_facts[neighbor_index][:, -1]
            mask1 = neighbor_ts < tmp_time   # 注意是小于
            if sample_method == 1:
                mask_ = mask1
            elif sample_method == 2:
                if mask_2:
                    mask2 = neighbor_ts <= pre_time[i]  # 前向
                    mask_ = mask1 & mask2
                else:
                    mask_ = mask1
            elif sample_method == 3:
                if mask_2:
                    mask2 = neighbor_ts >= pre_time[i]  # 后向
                    # mask2 = neighbor_ts > pre_time[i]  # 后向
                    mask_ = mask1 & mask2

                    # mask2 = neighbor_ts >= pre_time[i]  # 后向
                    # mask3 = neighbor_ts >= tmp_time - 10  # 时间窗口设置
                    # mask_ = mask1 & mask2 & mask3

                else:
                    mask_ = mask1

            neighbor_index = neighbor_index[mask_]
            # 增加时间窗口的话 不进行采样
            if len(neighbor_index) > sample_nodes:
                if mask_2:
                    if sample_method == 1:
                        neighbor_ts = neighbor_ts[mask_] - pre_time[i]
                        weights = 1.0 / (np.abs(neighbor_ts)+1)
                        weights = np.power(weights, sample_ratio)
                    elif sample_method == 2:
                        neighbor_ts = neighbor_ts[mask_] - pre_time[i]
                        weights = np.exp(neighbor_ts * sample_ratio) + 1e-9
                    elif sample_method == 3:
                        neighbor_ts = neighbor_ts[mask_] - tmp_time
                        # neighbor_ts = pre_time[i] - neighbor_ts[mask_]
                        weights = np.exp(neighbor_ts * sample_ratio) + 1e-9
                        # neighbor_ts = neighbor_ts[mask_] - tmp_time
                        # weights = 1.0 / (np.abs(neighbor_ts)+1)
                else:
                    neighbor_ts = neighbor_ts[mask_] - tmp_time  # 可以增加其他参数
                    weights = np.exp(neighbor_ts * sample_ratio) + 1e-9
                weights = weights / sum(weights)
                neighbor_index = np.random.choice(neighbor_index, sample_nodes, replace=False, p=weights)
            index1 = [i] * len(neighbor_index)
            neighbors.append(neighbor_index)
            parent_indexs.extend(index1)

        parent_indexs = np.array(parent_indexs)
        neighbors = np.concatenate(neighbors, axis=0)
        return parent_indexs, neighbors


    def load_neighbors4model_1(self, batch_data, query_time, device, sample_method, sample_nodes, sample_ratio):  # 第一层 B
        # Return B*N
        batch_data_array = batch_data.data.cpu().numpy()
        batch = batch_data.size(0)

        query_time = query_time.data.cpu().numpy()
        batch_index, nonzero_values = self.load_neighbors_by_array(batch_data_array, query_time, sample_method, sample_nodes, sample_ratio)

        batch_index = torch.LongTensor(batch_index).to(device)
        nonzero_values = torch.LongTensor(nonzero_values).to(device)

        temp_neighbors_facts = self.train_facts_tensor[nonzero_values].to(device)  # X*4 h,r,t,ts
        temp_neighbors_facts = torch.cat([batch_index.unsqueeze(1), batch_index.unsqueeze(1), temp_neighbors_facts], dim=1)  # X*6 b,h,r,t,ts

        # 新增自己指向自己 fact
        batch_list = np.array(range(batch), dtype=np.long)
        h_list = batch_data_array
        r_list = np.ones(batch, dtype=np.long) * -1
        t_list = batch_data_array
        ts_list = np.zeros(batch, dtype=np.long)
        temp_array = np.stack([batch_list, batch_list, h_list, r_list, t_list, ts_list], axis=1)  # B*6
        temp_neighbors_facts = torch.cat([torch.tensor(temp_array).to(device), temp_neighbors_facts], dim=0)  # X*6 b,n,h,r,t,ts

        return temp_neighbors_facts # X*6 b,n,h,r,t,ts

    def load_neighbors4model_2(self, batch_data, query_time, device, sample_method, sample_nodes, sample_ratio):  # N*3 ent-time pair
        num_nodes, _ = batch_data.size()
        # 增加h对应的ts  用于记录新node的父node
        batch_data_array = batch_data.data.cpu().numpy()  # N*3

        query_time = query_time.data.cpu().numpy()
        pre_time = batch_data_array[:,2]
        node_index, nonzero_values = self.load_neighbors_by_array(batch_data_array[:, 1], query_time, sample_method, sample_nodes, sample_ratio, mask_2=True, pre_time=pre_time)

        node_index = torch.LongTensor(node_index).to(device)
        nonzero_values = torch.LongTensor(nonzero_values).to(device)

        temp_neighbors_facts = self.train_facts_tensor[nonzero_values].to(device)  # X*4 h,r,t,ts

        batch_index = batch_data[:,0][node_index]
        temp_neighbors_facts = torch.cat([batch_index.unsqueeze(1), node_index.unsqueeze(1), temp_neighbors_facts], dim=1)  # X*6 b,n,h,r,t,ts

        # 新增自己指向自己 fact
        batch_list = batch_data_array[:,0]  # 应该是batch
        node_list = np.arange(num_nodes)
        h_list = batch_data_array[:,1]
        r_list = np.ones(num_nodes, dtype=np.long) * -1
        t_list = batch_data_array[:,1]
        ts_list = pre_time
        temp_array = np.stack([batch_list, node_list, h_list, r_list, t_list, ts_list], axis=1)  # B*6

        # temp_neighbors_facts = torch.cat([torch.tensor(temp_array), temp_neighbors_facts], dim=0)  # X*6 b,o_ts,h,r,t,ts
        temp_neighbors_facts = torch.cat([torch.tensor(temp_array).to(device), temp_neighbors_facts], dim=0)  # X*5 b,h,r,t,ts

        return temp_neighbors_facts  # X*6 b,n,h,r,t,ts
