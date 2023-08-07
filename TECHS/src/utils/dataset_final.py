import random
from os.path import join
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# import dgl

# from utils.data_process import build_graph, build_graph_snapshots
# from utils.data_utils import load_fact, load_json, load_pickle, save_pickle

from data_process import build_graph, build_graph_snapshots
from data_utils import load_fact, load_json, load_pickle, save_pickle

# import sys
# sys.getfilesystemencoding = lambda: 'UTF-8'

class tKGDataset(Dataset):  # dataset  # 后向、前向都尝试
    def __init__(self, data_dir, data_type='train', logic=False, logic_ratio=0.5):  # train valid test
        self.data_dir = data_dir
        self.data_type = data_type
        # self.max_time = -1  # 自己指向自己的时间设置
        self.label_smooth = 0.1
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
        self.n_train = len(train_facts)  # 未添加reverse
        self.train_facts = self.map2ids(train_facts, self.entity2id, self.relation2id, self.ts2id)
        self.valid_facts = self.map2ids(valid_facts, self.entity2id, self.relation2id, self.ts2id)
        self.test_facts = self.map2ids(test_facts, self.entity2id, self.relation2id, self.ts2id)

        self.all_facts = np.concatenate([self.train_facts,self.valid_facts,self.test_facts], axis=0)
        # print(self.all_facts.shape)
        # self.n_train_fact = len(self.train_facts)

        # 根据时间戳排序train_fact
        ts_array = self.train_facts[:, -1]
        ts_index = np.argsort(ts_array)  # 小到大排序
        self.train_facts = self.train_facts[ts_index]
        self.ts_sample_weights = self.gen_ts_weighs(p=2.0)


        self.tKG = build_graph(self.n_ent, self.train_facts)  # 构建训练KG

        # self.filters = self.get_filter()  # (h,r,ts): t字典 预测结果时使用
        # for filt in self.filters:
        #     self.filters[filt] = list(self.filters[filt])  # set转换为list
        # print(len(self.filters))


        self.Dic_E = self.load_neighbor_dic()
        self.ent_neighborfact_array, self.ent_neighborfact_ts_array = None, None
        # self.ent_neighborfact_array, self.ent_neighborfact_ts_array = self.load_neighbors_data()
        # self.ent_neighborfact_list, self.e_ts_pair_neighborfact_dict = self.load_neighbors_data2()
        # self.ent_neighborfact_list, self.e_ts_pair_neighborfact_dict = self.load_neighbors_data3()


        # self.train_facts, self.valid_facts, self.test_facts, self.ent_neighborfact_array, self.e_ts_pair_index_dic, self.e_ts_pair_neighborfact_array = self.example_data()

        facts4train = self.all_facts
        facts4train = np.concatenate([np.array([[-1, -1, -1, 0]]), facts4train], axis=0)  # 时间使用0
        self.train_facts_tensor = torch.tensor(facts4train)  # 训练时用于获取邻居
        print(self.train_facts_tensor.size())

        # self.h_r_dic = self.get_h_r_dic()  # 仅仅训练
        # self.h_r_list = list(self.h_r_dic.keys())
        self.h_r_dic_all = self.get_h_r_dic_all()  # 所有的，用于验证、测试

        # self.h_r_ts_dic, self.h_r_ts_index_dic = self.get_h_r_ts_dic()  # 仅仅训练
        # self.h_r_ts_list = list(self.h_r_ts_dic.keys())
        self.h_r_ts_dic_all = self.get_h_r_ts_dic_all()  # 所有的，用于验证、测试

        # tails_len = [len(item) for item in self.h_r_ts_dic.values()]
        # print('-----'*20)
        # print(max(tails_len), np.mean(tails_len), min(tails_len))

        # print(self.train_facts.shape)
        if self.logic:
            max_train_time = max(self.train_facts[:, -1])
            train_time = int(max_train_time * (1-self.logic_ratio))
            train_facts = self.train_facts
            fact_times = train_facts[:, -1]
            train_facts = train_facts[fact_times >= train_time]  # 部分用于训练
            self.train_facts_new = train_facts
            self.new2old_index = np.nonzero(fact_times >= train_time)[0]
            # print(self.train_facts.shape)

        self.dataset_describe()

    def load_neighbor_dic(self):
        Dic_E = defaultdict(list)  # entity: fact列表 h_fact_dic = defaultdict(list)  # 头实体的邻居fact index
        # for i, fact in enumerate(self.train_facts):
        for i, fact in enumerate(self.all_facts):
            h, r, t, ts = fact
            Dic_E[h].append(i)
        # len_list = []
        # for item in Dic_E.values():
        #     len_list.append(len(item))
        # print('Node Degree:', len(Dic_E), np.sum(len_list), np.mean(len_list), np.min(len_list), np.max(len_list), np.std(len_list), np.median(len_list))
        return Dic_E

    def gen_ts_weighs(self, p=2.0):  # 生成关于时间戳的采样权重 越往后、权重越大
        # p越小 越平均 (越大 越趋向后期)
        ts_array = np.arange(self.n_ent) + 1
        # ts_array = -1.0 / ts_array
        ts_array = np.power(ts_array, p)  # 幂运算
        # ts_array = ts_array / p
        # ts_array = np.exp(ts_array) + 1e-9
        ts_array = ts_array / sum(ts_array)
        return ts_array

    def view_valid_test(self):  # 查看valid/test中在train总h top出现的比例
        train_array = self.train_facts
        valid_array = self.valid_facts
        test_array = self.test_facts
        Dic_E = self.load_neighbor_dic()

        def cal_ratio(_array):
            len_list1 = []
            len_list2 = []
            num_pos = 0
            for i, fact in enumerate(_array):  # 1 hop
                h, r, t, ts = fact
                neighbor_facts = Dic_E[h]
                len_list1.append(len(neighbor_facts))
                neighbor_facts = train_array[neighbor_facts]
                neighbor_t = list(neighbor_facts[:,2])
                len_list2.append(len(neighbor_t))
                if t in neighbor_t:
                    num_pos += 1
            print(num_pos / len(_array), np.mean(len_list1), np.mean(len_list2))

            num_pos1 = 0
            num_pos2 = 0
            # _array = _array[:100]
            for i, fact in enumerate(_array):  # 2 hop
                # print(i)
                h, r, t, ts = fact
                neighbor_facts = Dic_E[h]
                neighbor_facts = train_array[neighbor_facts]
                neighbor_t = list(neighbor_facts[:,2])
                neighbor_ts = list(neighbor_facts[:,3])
                t_list = []
                for i, temp_t in enumerate(neighbor_t):
                    temp_ts = neighbor_ts[i]
                    new_neighs = Dic_E[temp_t]
                    new_neighs = train_array[new_neighs]
                    new_t = new_neighs[:, 2]
                    new_ts = new_neighs[:, 3]
                    new_t = new_t[new_ts >= temp_ts]
                    t_list.extend(list(new_t))
                if t in t_list:
                    num_pos1 += 1  # 仅仅第二层
                t_list.extend(neighbor_t)  # 第一层
                if t in t_list:
                    num_pos2 += 1
            print(num_pos1 / len(_array), num_pos2 / len(_array))

        cal_ratio(valid_array)  # 0.6629
        cal_ratio(test_array)  # 0.6390

    def view_data(self):  # 查看valid/test中在train总h top出现的比例
        train_array = self.train_facts
        valid_array = self.valid_facts
        test_array = self.test_facts

        # list set
        e_neighbor_list, e_ts_neighbor_dic = self.ent_neighborfact_list, self.e_ts_pair_neighborfact_dict
        sample_num = 5000

        def get_sample_weight(target_ts, ts_list):
            delta_t = (ts_list - target_ts) / 24
            porbs = np.exp(delta_t) + 1e-9
            porbs = porbs / sum(porbs)

            # porbs = 1.0 / (target_ts - ts_list)  # 相减、倒数
            # porbs = porbs / sum(porbs)
            return porbs

        def cal_ratio(_array):
            num_pos1 = 0
            num_pos2 = 0
            for i, fact in enumerate(_array):  # 1 hop
                h, r, t, ts = fact
                neighbor_list = e_neighbor_list[h]
                neighbor_ts1 = train_array[neighbor_list][:, 3]
                porbs = get_sample_weight(ts, neighbor_ts1)
                if len(neighbor_list) > sample_num:
                    neighbor_list = random.choices(neighbor_list, weights=porbs, k=sample_num)
                    # neighbor_list = random.choices(neighbor_list, k=sample_num)
                neighbor_tail = list(train_array[neighbor_list][:,2])
                if t in neighbor_tail:
                    num_pos1 += 1
                # hop2_e_list = []
                # for i, temp_t in enumerate(neighbor_tail):
                #     neighbor_list2 = e_neighbor_list[temp_t]
                #     tmp_ts = neighbor_ts1[i]
                #     neighbor_ts2 = train_array[neighbor_list2][:, 3]
                #     porbs = get_sample_weight(ts, neighbor_ts2)
                #     # porbs = get_sample_weight(tmp_ts, neighbor_ts2)
                #     if len(neighbor_list2) > sample_num:
                #         neighbor_list2 = random.choices(neighbor_list2, weights=porbs, k=sample_num)
                #     neighbor_tail2 = list(train_array[neighbor_list2][:, 2])
                #     hop2_e_list.extend(neighbor_tail2)
                # if t in hop2_e_list:
                #     num_pos2 += 1

            print(num_pos1 / len(_array),num_pos2 / len(_array))
            # 后向
            # test 40 0.44902435335047647 0.6674103766449856
            # test 100 0.5208364846468008 0.7491680532445923
            # test 100 0.6135985478747542 0.8216230524882772
            # 前向
            # test 40 query ts采样 0.4512176675238239 0.6668809559824534
            # test 100 query ts采样 0.5208364846468008 0.7493571320526395
            # test 40 节点ts采样 0.4500075631523219 0.6678263500226894
            # test 100 节点ts采样 0.519739827560127 0.7480713961579186

        # cal_ratio(valid_array)  # 0.6629
        cal_ratio(test_array)  # 0.6390

        # 0.5696303262678145 0.5464755710180003 采样时没有时间权重
        # 0.5679664327569992 0.547572228104674 采样权重：相减、倒数



    def dataset_describe(self):
        print('===========Dataset Description:===========')
        print('Entity Num:{} Relation Num:{} Time Num:{}'.format(self.n_ent, self.n_rel, self.n_ts))
        print('Train Num:{} Valid Num:{} Test Num:{}'.format(len(self.train_facts), len(self.valid_facts), len(self.test_facts)))

    def map2ids(self, facts, ent2id, rel2id, ts2id, add_rev=True):  # 使用id表示事实四元组、是否增加逆关系
        fact_ids = []
        for item in facts:
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
            # h_r = self.h_r_list[index_]
            # all_tails = self.h_r_dic[h_r]
            # h, r = h_r
            # y = np.zeros([self.n_ent], dtype=np.float32)
            # y[all_tails] = 1
            # y = (1.0 - self.label_smooth) * y + (1.0 / self.n_ent)  # label_smooth

            # h_r_ts = self.h_r_ts_list[index_]
            # all_tails = self.h_r_ts_dic[h_r_ts]
            # h, r, ts = h_r_ts
            # y = np.zeros([self.n_ent], dtype=np.float32)
            # y[all_tails] = 1
            # y = (1.0 - self.label_smooth) * y + (1.0 / self.n_ent)  # label_smooth
            # mask_index = self.h_r_ts_index_dic[h_r_ts]  # list 一个或者多个
            # return np.array([index_, h, r, ts]), y, mask_index
            if self.logic:
                train_facts = self.train_facts_new
            else:
                train_facts = self.train_facts
            h,r,t,ts = train_facts[index_]
            y = np.zeros([self.n_ent], dtype=np.float32)
            y[t] = 1
            # y = (1.0 - self.label_smooth) * y + (1.0 / self.n_ent)

            if self.logic:
                mask_index = self.new2old_index[index_]
            else:
                mask_index = 0
            return np.array([index_, h, r, t, ts]), y, mask_index
        elif self.data_type == 'valid':
            # h, r, t, ts = self.valid_facts[index_]
            # y = np.zeros([self.n_ent], dtype=np.float32)
            # y[t] = 1
            # return np.array([index_, h, r, t, ts]), y

            h, r, t, ts = self.valid_facts[index_]

            # h_r = (h, r)
            # all_tails = self.h_r_dic_all[h_r]
            # y = np.zeros([self.n_ent], dtype=np.float32)
            # y[all_tails] = 1

            h_r_ts = (h, r, ts)
            all_tails = self.h_r_ts_dic_all[h_r_ts]
            y = np.zeros([self.n_ent], dtype=np.float32)
            y[all_tails] = 1

            return index_, h, r, t, ts, y
        elif self.data_type == 'test':  # 目前不考虑mask
            h, r, t, ts = self.test_facts[index_]

            # h_r = (h, r)
            # all_tails = self.h_r_dic_all[h_r]
            # y = np.zeros([self.n_ent], dtype=np.float32)
            # y[all_tails] = 1

            h_r_ts = (h, r, ts)
            all_tails = self.h_r_ts_dic_all[h_r_ts]
            y = np.zeros([self.n_ent], dtype=np.float32)
            y[all_tails] = 1

            return index_, h, r, t, ts, y

    def __len__(self):
        # return 1000
        if self.data_type == 'train':
            # return len(self.h_r_list)
            # return len(self.h_r_ts_list)
            if self.logic:
                train_facts = self.train_facts_new
            else:
                train_facts = self.train_facts
            return len(train_facts)
        elif self.data_type == 'valid':
            return len(self.valid_facts)
        elif self.data_type == 'test':
            return len(self.test_facts)

    def get_h_r_dic(self):  # 仅仅训练
        h_r_dic = defaultdict(list)
        for item in self.train_facts:
            h,r,t,ts = item
            h_r_dic[(h,r)].append(t)
        return h_r_dic

    def get_h_r_dic_all(self):  # 训练、测试
        h_r_dic = defaultdict(list)
        for item in self.train_facts:
            h,r,t,ts = item
            h_r_dic[(h,r)].append(t)
        for item in self.valid_facts:
            h,r,t,ts = item
            h_r_dic[(h,r)].append(t)
        for item in self.test_facts:
            h,r,t,ts = item
            h_r_dic[(h,r)].append(t)
        return h_r_dic

    def get_h_r_ts_dic(self):  # 仅仅训练  返回用于mask的index
        h_r_ts_dic = defaultdict(list)
        h_r_ts_index_dic = defaultdict(list)
        # for i, item in enumerate(self.train_facts[self.train_facts[:,-1] > 150]):  # 使用后续部分进行
        for i, item in enumerate(self.train_facts):
            h,r,t,ts = item
            h_r_ts_dic[(h, r, ts)].append(t)
            h_r_ts_index_dic[(h, r, ts)].append(i)  # 只需要mask自身的index
        return h_r_ts_dic, h_r_ts_index_dic

    def get_h_r_ts_dic_all(self):  # 训练、测试
        h_r_ts_dic = defaultdict(list)
        for item in self.train_facts:
            h,r,t,ts = item
            h_r_ts_dic[(h,r, ts)].append(t)
        for item in self.valid_facts:
            h,r,t,ts = item
            h_r_ts_dic[(h,r, ts)].append(t)
        for item in self.test_facts:
            h,r,t,ts = item
            h_r_ts_dic[(h,r, ts)].append(t)
        return h_r_ts_dic


    def view_neighbor_num(self):
        Dic_E = self.Dic_E
        num_list = []
        for ent in range(self.n_ent):
            num_list.append(len(Dic_E[ent]))
        print(np.min(num_list),np.max(num_list),np.median(num_list),np.mean(num_list))
        # 0 4001 2.0 17.868967452300787

    def save_neighbors(self, max_nodes=100):  # 获取邻居 4001
        Dic_E = self.Dic_E
        ent_neighborfact_array = []
        ent_neighborfact_ts_array = []  # 保存对应的时间戳
        len_list = []
        for ent in range(self.n_ent):
            temp_neighbors = []
            temp_neighbors_ts = []
            neighbors = Dic_E[ent]  # 如果长度为0？
            length = len(neighbors)
            len_list.append(length)
            if length == 0:
                temp_neighbors = [-1] * max_nodes  # [-1] 默认index
                temp_neighbors_ts = [0] * max_nodes  # [-1] 默认index
            elif length > max_nodes:  # 无放回采样 不会出现
                # 随机采样
                # n_ts = self.train_facts[neighbors][:,-1]
                # index_sample = np.random.choice(length, max_nodes)
                # temp_neighbors = list(np.array(neighbors)[index_sample])
                # temp_neighbors_ts = list(n_ts[index_sample])
                # 后k个采样
                # n_ts = self.train_facts[neighbors][:, -1][-max_nodes:]
                # temp_neighbors = list(np.array(neighbors)[-max_nodes:])
                # temp_neighbors_ts = list(n_ts)
                # 根据weight采样
                ts_ = self.train_facts[neighbors][:, -1]
                weights = self.ts_sample_weights[ts_]
                # print(len(weights))
                # print(weights)
                weights = weights / sum(weights)
                # print(weights)
                index_sample = np.random.choice(length, max_nodes, replace=False, p=weights)
                temp_neighbors = list(np.array(neighbors)[index_sample])
                temp_neighbors_ts = list(ts_[index_sample])
            else:  # -1 padding
                temp_neighbors_ts = list(self.train_facts[neighbors][:,-1])
                temp_neighbors = neighbors.copy()
                # random.shuffle(temp_neighbors)  # 不需要  保持从小到大（不一定 增加了reverse）
                temp_neighbors.extend([-1] * (max_nodes - length))
                temp_neighbors_ts.extend([0] * (max_nodes - length))
            ent_neighborfact_array.append(temp_neighbors)
            ent_neighborfact_ts_array.append(temp_neighbors_ts)
        ent_neighborfact_array = np.array(ent_neighborfact_array)
        ent_neighborfact_ts_array = np.array(ent_neighborfact_ts_array)
        # print(ent_neighborfact_array.shape)
        # print(ent_neighborfact_ts_array.shape)
        # print(ent_neighborfact_array[:10])
        # print(ent_neighborfact_ts_array[:10])

        print(np.min(len_list),np.max(len_list),np.mean(len_list),np.median(len_list))
        # 0 4001 17.868967452300787 2.0
        lower = np.quantile(len_list, 0.25, interpolation='lower')
        higher = np.quantile(len_list, 0.98, interpolation='higher')
        print(lower, higher)  # 1 144
        len_list = []
        h_list = []
        for item in self.test_facts:  # 测试集上的
            h = item[0]
            neighbors = Dic_E[h]  # 如果长度为0？
            length = len(neighbors)
            len_list.append(length)
            h_list.append(h)
        print(np.min(len_list), np.max(len_list), np.mean(len_list), np.median(len_list))
        # 0 4001 17.868967452300787 2.0
        lower = np.quantile(len_list, 0.25, interpolation='lower')
        higher = np.quantile(len_list, 0.95, interpolation='higher')
        print(lower, higher)  # 1 144
        from collections import Counter
        counter = Counter(h_list)
        print(counter)

        np.save(join(self.data_dir, 'ent_neighborfact_array.npy'), ent_neighborfact_array)
        np.save(join(self.data_dir, 'ent_neighborfact_ts_array.npy'), ent_neighborfact_ts_array)


    def load_neighbors_data(self):  # 获取邻居
        ent_neighborfact_array = np.load(join(self.data_dir, 'ent_neighborfact_array.npy'))
        ent_neighborfact_ts_array = np.load(join(self.data_dir, 'ent_neighborfact_ts_array.npy'))
        return ent_neighborfact_array, ent_neighborfact_ts_array

    def load_neighbors_data2(self):  # 获取邻居
        ent_neighborfact_list = load_pickle(join(self.data_dir, 'ent_neighborfact_list.pkl'))
        e_ts_pair_neighborfact_dict = load_pickle(join(self.data_dir, 'e_ts_pair_neighborfact_dict.pkl'))
        return ent_neighborfact_list, e_ts_pair_neighborfact_dict

    def load_neighbors_data3(self):  # 获取邻居  前向
        ent_neighborfact_list = load_pickle(join(self.data_dir, 'ent_neighborfact_list.pkl'))
        e_ts_pair_neighborfact_dict = load_pickle(join(self.data_dir, 'e_ts_pair_neighborfact_dict_prior.pkl'))
        return ent_neighborfact_list, e_ts_pair_neighborfact_dict

    def mask_neighbours(self, origin_neighbours, mask_index_tensor):  # 对邻居节点进行mask、训练时  B*M B*Y
        _neighbours = origin_neighbours.unsqueeze(-1)  # BM1
        mask_index_tensor = mask_index_tensor.unsqueeze(1)  # B1Y
        index = _neighbours == mask_index_tensor  # BMY
        index = torch.sum(index, dim=-1)  # BM
        origin_neighbours[index != 0] = -1
        return origin_neighbours

    def mask_neighbours_2(self, origin_neighbours, mask_index_tensor, num_nodes):  # 对邻居节点进行mask、训练时  BM*N B*Y
        _neighbours = origin_neighbours.unsqueeze(-1)  # BM*N*1
        batch, size1 = mask_index_tensor.size()
        mask_index_tensor = mask_index_tensor.unsqueeze(1).expand(batch, num_nodes, size1)   # B*1*Y ->  B*M*Y
        mask_index_tensor = mask_index_tensor.reshape(batch*num_nodes,-1).unsqueeze(1)  # BM*1*Y
        index = _neighbours == mask_index_tensor  # BM*N*Y
        index = torch.sum(index, dim=-1)  # BM*N
        origin_neighbours[index != 0] = -1
        return origin_neighbours

    def load_ent_neighbors(self, batch_data_array, mask=False, mask_index_array=None):
        neighbor_list = []
        if mask == False:
            for i, ent in enumerate(batch_data_array):
                temp_neighbors = self.ent_neighborfact_list[ent].copy()
                neighbor_list.append(temp_neighbors[:20])
        else:
            for i, ent in enumerate(batch_data_array):
                temp_neighbors = self.ent_neighborfact_list[ent].copy()
                temp_neighbors.remove(mask_index_array[i])
                neighbor_list.append(temp_neighbors[:20])
        batch_index = []
        neighbor_values = []
        for i in range(len(neighbor_list)):
            batch_index.extend([i] * len(neighbor_list[i]))
            neighbor_values.extend(neighbor_list[i])
        return batch_index, neighbor_values

    def load_ent_ts_neighbors(self, batch_data_array, mask=False, mask_index_array=None):
        cur_nodes = [(item[1], item[2]) for item in batch_data_array]
        neighbor_list_ = [self.e_ts_pair_neighborfact_dict[item] for item in cur_nodes]

        neighbor_list = []
        if mask == False:
            for i, ent in enumerate(batch_data_array):
                temp_neighbors = list(neighbor_list_[i])  # set -> list
                neighbor_list.append(temp_neighbors[:20])
        else:
            for i, ent in enumerate(batch_data_array):
                temp_neighbors = neighbor_list_[i].copy()  # set
                temp_neighbors = temp_neighbors - set([int(mask_index_array[i])])
                neighbor_list.append(list(temp_neighbors)[:20])  # set -> list
        batch_index = []
        neighbor_values = []
        for i in range(len(neighbor_list)):
            batch_index.extend([i] * len(neighbor_list[i]))
            neighbor_values.extend(neighbor_list[i])
        return batch_index, neighbor_values

    def load_neighbors_by_array(self, batch, query_time, query_ent, mask=False, mask_index=None):
        '''
        :param query_time: array B
        :param query_ent: array B
        :param mask_index: array B
        :return: array array
        '''
        temp_neighbors = self.ent_neighborfact_array[query_ent]  # B*N
        temp_neighbors_ts = self.ent_neighborfact_ts_array[query_ent]  # B*N
        query_time = query_time.reshape(batch, 1)  # 增加1
        ts_mask = temp_neighbors_ts > query_time  # B*N boolean
        temp_neighbors[ts_mask] = -1
        if mask:
            # mask_index = mask_index.reshape(batch, 1)  # 增加1
            pred_mask = temp_neighbors == mask_index
            temp_neighbors[pred_mask] = -1
        temp_neighbors = temp_neighbors + 1
        temp_neighbors = np.sort(temp_neighbors, axis=1)[:, -50:]  # 只取部分
        index1, index2 = np.nonzero(temp_neighbors)
        nonzero_values = temp_neighbors[index1, index2]
        return index1, nonzero_values

    def load_neighbors4model_1(self, batch_data, query_time, mask_index_tensor, device):  # 第一层 B
        # Return B*N
        batch_data_array = batch_data.data.cpu().numpy()
        batch = batch_data.size(0)

        if mask_index_tensor == None:
            query_time = query_time.data.cpu().numpy()
            batch_index, nonzero_values = self.load_neighbors_by_array(batch, query_time, batch_data_array, mask=False, mask_index=None)
        else:  # mask_index_tensor一维
            query_time = query_time.data.cpu().numpy()
            mask_index = mask_index_tensor.data.cpu().numpy()
            batch_index, nonzero_values = self.load_neighbors_by_array(batch, query_time, batch_data_array, mask=True, mask_index=mask_index)
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

    def load_neighbors4model_2(self, batch_data, query_time, mask_index_tensor, device):  # N*3 ent-time pair
        num_nodes, _ = batch_data.size()
        # 增加h对应的ts  用于记录新node的父node
        batch_data_array = batch_data.data.cpu().numpy()  # N*3


        if mask_index_tensor == None:
            query_time = query_time.data.cpu().numpy()
            batch_index, nonzero_values = self.load_neighbors_by_array(num_nodes, query_time, batch_data_array[:,1], mask=False, mask_index=None)
        else:  # mask_index_tensor一维
            query_time = query_time.data.cpu().numpy()
            mask_index = mask_index_tensor.data.cpu().numpy()
            batch_index, nonzero_values = self.load_neighbors_by_array(num_nodes, query_time, batch_data_array[:,1], mask=True, mask_index=mask_index)
        node_index = torch.LongTensor(batch_index).to(device)
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
        ts_list = batch_data_array[:,2]
        temp_array = np.stack([batch_list, node_list, h_list, r_list, t_list, ts_list], axis=1)  # B*6

        # temp_neighbors_facts = torch.cat([torch.tensor(temp_array), temp_neighbors_facts], dim=0)  # X*6 b,o_ts,h,r,t,ts
        temp_neighbors_facts = torch.cat([torch.tensor(temp_array).to(device), temp_neighbors_facts], dim=0)  # X*5 b,h,r,t,ts

        return temp_neighbors_facts  # X*6 b,n,h,r,t,ts

def tkg_collate_fn(samples):
    # index_, h, r, t, y, ts = map(list, zip(*samples))
    # # tKG = tKG[0]
    #
    # index_ = torch.LongTensor(index_)
    # h = torch.LongTensor(h)
    # r = torch.LongTensor(r)
    # t = torch.LongTensor(t)
    # ts = torch.LongTensor(ts)
    # y = torch.tensor(y)
    # return index_, h, r, t, y, ts

    triple_info, y, mask_index = map(list, zip(*samples))
    triple_info = torch.LongTensor(np.array(triple_info))
    y = torch.tensor(np.array(y))


    mask_index_tensor = torch.LongTensor(np.array(mask_index))

    # max_len = max([len(item) for item in mask_index])
    # mask_index_tensor = []
    # for item in mask_index:
    #     temp_list = item.copy()
    #     if len(temp_list) < max_len:
    #         temp_list.extend([-1] * (max_len-len(temp_list)))
    #     mask_index_tensor.append(temp_list)
    # mask_index_tensor = torch.LongTensor(np.array(mask_index_tensor))  # padding到同一维度



    # print(mask_index)
    # print(mask_index_tensor)
    # print(mask_index_tensor.size())
    return triple_info, y, mask_index_tensor

def tkg_collate_fn2(samples):
    index_, h, r, t, ts, y = map(list, zip(*samples))
    # tKG = tKG[0]

    index_ = torch.LongTensor(np.array(index_))
    h = torch.LongTensor(np.array(h))
    r = torch.LongTensor(np.array(r))
    t = torch.LongTensor(np.array(t))
    ts = torch.LongTensor(np.array(ts))
    y = torch.tensor(np.array(y))
    return index_, h, r, t, y, ts



def tkg_collate_fn_valid(samples):  # 之前版本 目前未使用
    n_ent, query_info, mask_index, tkg_temp = map(list, zip(*samples))  # 每个batch一个样本

    n_ent = n_ent[0]  # int
    query_info = torch.LongTensor(query_info[0])
    mask_index = torch.LongTensor(mask_index[0])
    tkg_temp = tkg_temp[0]

    target_tails = query_info[:, 2]
    batch = query_info.size()[0]
    targets = torch.zeros((batch, n_ent))
    for i in range(batch):
        targets[i, target_tails[i]] = 1
    return query_info, mask_index, targets, tkg_temp


if __name__ == '__main__':
    data_dir = '../../data/icews14'

    tkg_dataset = tKGDataset(data_dir)
    tkg_dataset.save_neighbors(max_nodes=100)
    # tkg_dataset.view_valid_test()
    # tkg_dataset.view_data()
    # tkg_dataset.view_neighbor_num()
    # tkg_dataset.view()
    # for item in tkg_dataset:
    #     print(item, np.sum(item[4]))
    '''
    no = 0
    for item in tkg_dataset:
        time, query_info, mask_index, kg = item
        for time_ in query_info[:, -1]:
            if time_ != time:
                no += 1
        print(np.sum(mask_index))
        print(item)
        # break
    print(no)
    '''
    # tkg_dataset.view_valid_test()
    # tkg_dataset.save_neighbors()
    # nodes = torch.tensor([1, 2, 4, 5])
    # a = tkg_dataset.load_neighbors4model_1(nodes)

    # nodes = torch.tensor([[[2,0],[2,0],[1,0]],
    #                       [[6,0],[7,0],[8,0]]])
    # a = tkg_dataset.load_neighbors4model_2(nodes)
    print('xxx')

