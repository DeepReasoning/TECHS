from collections import defaultdict
import numpy as np
from scipy.stats import rankdata

import torch
import dgl


def build_graph(num_ent, train_facts):  # 生成总的dgl graph 训练数据集上的
    # g = dgl.DGLGraph()
    # g = dgl.graph()

    # g.add_nodes(num_ent)
    # g.add_edges(train_facts[:, 0], train_facts[:, 2])

    g = dgl.graph((train_facts[:, 0], train_facts[:, 2]), num_nodes=num_ent)

    # g.edata['relation'] = train_facts[:, 1]  # E 长度
    g.edata['type'] = torch.tensor(train_facts[:, 1], dtype=torch.long)  # E 长度
    g.edata['time'] = torch.tensor(train_facts[:, 3], dtype=torch.long)

    # norm
    # in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    # norm = in_deg ** -0.5
    # norm[np.isinf(norm)] = 0
    # g.ndata['xxx'] = torch.tensor(norm)
    # g.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
    # norm = g.edata.pop('xxx')
    # g.edata['norm'] = norm

    # 仅仅考虑入度
    # in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    # norm = 1 / in_deg  # 倒数
    # norm[np.isinf(norm)] = 0  # inductive 新实体
    # g.ndata['xxx'] = torch.tensor(norm)
    # g.apply_edges(lambda edges: {'xxx': edges.dst['xxx']})
    # norm = g.edata.pop('xxx')
    # g.edata['norm'] = norm

    return g


def build_graph_from_facts(num_ent, facts):  # 生成特定dgl graph 根据特定facts
    g = dgl.graph((facts[:, 0], facts[:, 2]), num_nodes=num_ent)

    g.edata['type'] = torch.tensor(facts[:, 1], dtype=torch.long)  # E 长度
    g.edata['time'] = torch.tensor(facts[:, 3], dtype=torch.long)

    # norm
    # in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    # norm = in_deg ** -0.5
    # norm[np.isinf(norm)] = 0
    # g.ndata['xxx'] = torch.tensor(norm)
    # g.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
    # norm = g.edata.pop('xxx')
    # g.edata['norm'] = norm

    # 仅仅考虑入度
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1 / in_deg  # 倒数
    norm[np.isinf(norm)] = 0  # inductive 新实体
    g.ndata['xxx'] = torch.tensor(norm)
    g.apply_edges(lambda edges: {'xxx': edges.dst['xxx']})  # 入度
    norm = g.edata.pop('xxx')
    g.edata['norm'] = norm

    return g


def build_graph_dic(num_ent, all_facts, time_list):  # 生成总的dgl dic 用于训练测试、小于当前时间
    '''
    :param num_ent:
    :param all_facts:
    :param time_list: 包含start_time, end_time
    :return:
    '''
    graph_dic = {}
    fact_times = all_facts[:, -1]
    start_time, end_time = time_list
    for temp_time in range(start_time, end_time+1):
        temp_facts = all_facts[fact_times < temp_time]  # 不包含当前
        temp_g = build_graph_from_facts(num_ent, temp_facts)
        graph_dic[temp_time] = temp_g
    return graph_dic


def build_graph_snapshots(dgl_g, time_list):  # 时序snapshots dic time: tkg
    all_tkg = dgl_g
    all_relations = dgl_g.edata['type']
    edge_times = all_tkg.edata['time']
    all_edge_len = len(edge_times)
    edge_list = np.array(range(all_edge_len))

    time_graph_dic = defaultdict(set)
    time_graph_edge_dic = defaultdict(dict)  # 新老graph的关系dict
    for time in time_list:
        edge_index = edge_times <= time
        new_edges = edge_list[edge_index]
        new_g = all_tkg.edge_subgraph(new_edges, preserve_nodes=True)

        # norm
        in_deg = new_g.in_degrees(range(new_g.number_of_nodes())).float().numpy()
        norm = in_deg ** -0.5
        norm[np.isinf(norm)] = 0
        new_g.ndata['xxx'] = norm
        new_g.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
        norm = new_g.edata.pop('xxx').squeeze()
        new_g.edata['norm'] = norm

        # relation type
        new_g.edata['type'] = all_relations[new_g.parent_eid]
        new_g.edata['time'] = edge_times[new_g.parent_eid]
        time_graph_dic[time] = new_g

        temp_edge_dic = {}
        parent_eids = new_g.parent_eid.numpy()
        for i, parent in enumerate(parent_eids):
            temp_edge_dic[parent] = i
        time_graph_edge_dic[time] = temp_edge_dic
    return time_graph_dic, time_graph_edge_dic


def cal_ranks(scores, labels, filters):  # 输入都是B×E
    scores = scores - np.min(scores, axis=1, keepdims=True)  # 为啥减最小值
    # full_rank = rankdata(-scores, method='ordinal', axis=1)  # 从大到小的rank 尝试
    full_rank = rankdata(-scores, method='average', axis=1)  # 从大到小的rank
    filter_scores = scores * filters
    filter_rank = rankdata(-filter_scores, method='min', axis=1)
    ranks = (full_rank - filter_rank + 1) * labels      # get the ranks of multiple answering entities simultaneously
    ranks = ranks[np.nonzero(ranks)]
    return list(ranks)


def cal_ranks_new(scores, labels, filters):  # 输入都是B×E  labels必然在filters中
    max_value = 10000
    label_score_new = 1 * max_value * labels
    filter_score_new = -1 * max_value * filters
    new_score = scores + label_score_new + filter_score_new
    ranks = rankdata(-new_score, method='average', axis=1)
    ranks = ranks * labels
    ranks = ranks[np.nonzero(ranks)]
    return list(ranks)


def cal_performance(ranks):
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks<=1) * 1.0 / len(ranks)
    h_3 = sum(ranks<=3) * 1.0 / len(ranks)
    h_10 = sum(ranks<=10) * 1.0 / len(ranks)
    return mrr, h_1, h_3, h_10

