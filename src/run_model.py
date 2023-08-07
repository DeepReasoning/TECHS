import os, time
from os.path import join
import random
import argparse
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

# from utils.dataset_new import tKGDataset, tkg_collate_fn, tkg_collate_fn2
from utils.dataset_new import tKGDataset
from utils.data_process import cal_ranks, cal_performance
# from CompGCN import CompGCN_DistMult, CompGCN_ConvE
from model import CompGCN_DistMult, GLogicalLayer, TimeEncode, TGCN

# torch.cuda.set_device(0)
# device = torch.device('cuda:0')
# device = torch.device('cpu')
# num_workers = 8
# batch_size = 64

class TLoGN(torch.nn.Module):  # 整体网络
    def __init__(self, n_ent, n_rel, n_ts, act, device, args=None):
        super(TLoGN, self).__init__()
        self.input_dim = args.gcn_dim
        self.gcn_dim = args.gcn_dim  # 前两目前相同
        self.hidden_dim = args.hidden_dim  # logic layer

        self.gcn_drop = args.gcn_drop
        # self.gcn_drop = 0.3
        self.hidden_drop = args.hidden_drop

        self.gcn_layer = args.gcn_layer
        self.logic_layer = args.logic_layer
        self.max_nodes = args.max_nodes

        self.device = device
        self.act = act  # 激活函数 torch.tanh
        # self.score_type = 'distmult'  # complex distmult conve

        self.use_gcn = args.use_gcn
        # self.gcn_time = args.gcn_time
        # self.use_logic = args.use_logic
        # self.logic_time = args.logic_time
        self.n_ts = n_ts

        # self.bias = nn.Parameter(torch.zeros(n_ent))
        self.time_encoder = TimeEncode(self.input_dim)  # 整体的时间信息建模
        self.gcn_layer = TGCN(n_ent, n_rel, self.input_dim, self.gcn_dim, n_layer=self.gcn_layer, conv_bias=False,
                              gcn_drop=self.gcn_drop, opn='mult', act=self.act, device=self.device)
        self.logic_layer = GLogicalLayer(n_ent, n_rel, self.gcn_dim, self.hidden_dim, n_layer=self.logic_layer,
                                         dropout=self.hidden_drop,max_nodes=self.max_nodes, act=self.act,
                                         reduce=args.logic_reduce, device=self.device)  # mean

        # self.line_e_ts = nn.Linear(self.gcn_dim*2, self.gcn_dim)
        # self.line_r_ts = nn.Linear(self.gcn_dim*2, self.gcn_dim)

        self.ent_emds = self.get_param([n_ent, self.gcn_dim])
        self.rel_emds = self.get_param([n_rel*2, self.gcn_dim])

    def gen_time_emds(self):
        times = torch.arange(0, self.n_ts).to(self.device)
        time_emds = self.time_encoder(times)
        # print(time_emds.size())
        return time_emds

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))  # relu
        # nn.init.xavier_normal_(param, gain=nn.init.calculate_gain(self.act))  # relu
        return param

    def gen_gcn_emds(self, tkg_temp):  # 生成GCN嵌入向量
        time_emds = self.gen_time_emds()
        if tkg_temp == None:
            return self.ent_emds, self.rel_emds
        else:
            ent_emds, rel_emds = self.gcn_layer(tkg_temp, time_emds, self.ent_emds, self.rel_emds)
            # 和初始特征搞一个残差 (是否需要)
            ent_emds = self.ent_emds + ent_emds
            rel_emds = self.rel_emds + rel_emds
            return ent_emds, rel_emds, time_emds

    # def gcn_scores(self, tkg_temp, query_head, query_rel, query_time):  # 第一个score
    #     ent_emds, rel_emds = self.gen_gcn_emds(tkg_temp)
    #
    #     h_emds = ent_emds[query_head]
    #     r_emds = rel_emds[query_rel]  # relation
    #     time_emd = self.time_encoder(query_time)  # BD
    #     h_emds = torch.cat([h_emds, time_emd], dim=-1)
    #     r_emds = torch.cat([r_emds, time_emd], dim=-1)
    #     h_emds = self.line_e_ts(h_emds)
    #     r_emds = self.line_r_ts(r_emds)
    #
    #     obj_emb = h_emds * r_emds  # B*D
    #     x = torch.mm(obj_emb, ent_emds.transpose(1, 0))  # B*N
    #     # x += self.bias.expand_as(x)
    #     gcn_scores = torch.sigmoid(x)  # B*N
    #     return gcn_scores

    def logic_scores(self, ent_emds, rel_emds, time_emds, query_head, query_rel, query_time, dataset, device, args):  # 逻辑规则scores
        scores, tail_ents = self.logic_layer(ent_emds, rel_emds, time_emds, query_head, query_rel, query_time, dataset, device, args)
        return scores, tail_ents



class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.1, reduction='mean'):  # 2 0.25
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        # pt = torch.sigmoid(predict) # sigmoide获取概率
        pt = predict
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) \
               - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class Runner(object):
    def __init__(self, args=None):
        data_dir = f'../data/{args.dataset}'

        if args.loss == 'bce':
            self.loss = nn.BCELoss()
        if args.loss == 'focal':
            self.loss = BCEFocalLoss()

        # self.tkg_dataset_train = tKGDataset(data_dir, data_type='train')
        self.tkg_dataset_valid = tKGDataset(data_dir, data_type='valid')
        self.tkg_dataset_test = tKGDataset(data_dir, data_type='test')
        self.tkg_dataset_train = tKGDataset(data_dir, data_type='train', logic=True, logic_ratio=args.logic_ratio)  # 训练logic的比例

        # self.tkg_dataloder_train = DataLoader(self.tkg_dataset_train, batch_size=args.train_batch, shuffle=True,
        #                                       num_workers=args.num_workers, collate_fn=tkg_collate_fn)
        # self.tkg_dataloder_valid = DataLoader(self.tkg_dataset_valid, batch_size=args.test_batch, shuffle=False,
        #                                       num_workers=args.num_workers, collate_fn=tkg_collate_fn2)
        # self.tkg_dataloder_test = DataLoader(self.tkg_dataset_test, batch_size=args.test_batch, shuffle=False,
        #                                      num_workers=args.num_workers, collate_fn=tkg_collate_fn2)

        self.logic_batch_train = args.train_batch  # 256 16
        self.logic_batch_test = args.test_batch
        self.label_smooth = 1 - args.lable_smooth  # 0.95
        self.tkg_dataloder_train_logic = DataLoader(self.tkg_dataset_train, batch_size=self.logic_batch_train,
                                                    shuffle=True, num_workers=args.num_workers)
        self.tkg_dataloder_valid_logic = DataLoader(self.tkg_dataset_valid, batch_size=self.logic_batch_test,
                                                    shuffle=False, num_workers=args.num_workers)
        self.tkg_dataloder_test_logic = DataLoader(self.tkg_dataset_test, batch_size=self.logic_batch_test,
                                                   shuffle=False, num_workers=args.num_workers)

        self.args = args
        self.num_ent = self.tkg_dataset_train.n_ent
        self.num_ts = self.tkg_dataset_train.n_ts
        if args.gpu >= 0:
            self.device = torch.device('cuda:{}'.format(args.gpu))
        else:
            self.device = torch.device('cpu')


        if args.act == 'relu':
            act = torch.relu
        elif args.act == 'tanh':
            act = torch.tanh
        else:
            print('activation function wrong......')

        self.model = TLoGN(n_ent=self.tkg_dataset_train.n_ent, n_rel=self.tkg_dataset_train.n_rel,n_ts=self.num_ts,
                           act=act, device=self.device, args=args).to(self.device)
        total_params = sum([param.nelement() for param in self.model.parameters()])
        print('Model Parameters Num:', total_params)
        for name, param in self.model.named_parameters():
            print(name, param.size(), param.device, str(param.requires_grad))
        # self.optimizer_gcn = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.l2)
        # self.optimizer_logic = torch.optim.Adam(self.model.logic_layer.parameters(), lr=args.lr, weight_decay=args.l2)  # 规则部分
        self.optimizer_logic = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.l2)  # 规则部分
        self.tkg_temp = self.tkg_dataset_train.tKG.to(self.device)  # 所有图

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def gen_target_ont_hot(self, target_ent, pre_ents):  # 生成one-hot目标
        '''
        :param target_ent: B
        :param pre_ents: X*2
        :return:
        '''
        one_hot_label = torch.from_numpy(
            np.array([int(ent == target_ent[batch_id]) for batch_id, ent in pre_ents], dtype=np.float32)).to(self.device)
        # smooth
        one_hot_label = one_hot_label * self.label_smooth + 0.0001
        return one_hot_label

    def train_logic_epoch(self):
        self.model.train()
        losses = []
        for item in tqdm(self.tkg_dataloder_train_logic):
            h, r, t, ts = item  # B*4、E edge数量、B*N 实体one hot

            query_head = h.to(self.device)
            query_relation = r.to(self.device)
            query_tail = t.to(self.device)
            query_time = ts.to(self.device)

            # X X*2
            if self.args.use_gcn == 1:
                ent_emds, rel_emds, time_emds = self.model.gen_gcn_emds(self.tkg_temp)
            else:
                ent_emds, rel_emds, time_emds = self.model.gen_gcn_emds(None)
            scores, tail_ents = self.model.logic_scores(ent_emds, rel_emds, time_emds, query_head, query_relation, query_time,
                                                       dataset=self.tkg_dataset_train, device=self.device, args=self.args)

            if self.args.loss == 'max_min':
                batch_size = h.size()[0]
                scores_all = torch.zeros((batch_size, self.num_ent)).to(self.device)
                scores_all[[tail_ents[:, 0], tail_ents[:, 1]]] = scores
                scores = scores_all
                pos_scores = scores[[torch.arange(batch_size).to(self.device), query_tail]]
                max_n = torch.max(scores, 1, keepdim=True)[0]
                loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n), 1)))
            else:  # BCE
                target_one_hot = self.gen_target_ont_hot(query_tail, tail_ents)  # X
                loss = self.calc_loss(scores, target_one_hot)


            self.optimizer_logic.zero_grad()
            loss.backward()

            # clip_grad_norm_(self.model.parameters(), max_norm=5)  # 梯度截断 10
            clip_grad_norm_(self.model.parameters(), max_norm=5)  # 梯度截断
            self.optimizer_logic.step()
            losses.append(loss.item())
        loss = np.mean(losses)
        return loss

    def eval(self, mode='valid', model_mode='gcn'):  # gcn logic both不同得分函数的evaluation
        self.model.eval()
        eval_loader = None
        eval_dataset = None
        if model_mode=='gcn':
            if mode == 'valid':
                eval_loader = self.tkg_dataloder_valid
                eval_dataset = self.tkg_dataset_valid
            elif mode == 'test':
                eval_loader = self.tkg_dataloder_test
                eval_dataset = self.tkg_dataset_test
        elif model_mode=='logic':
            if mode == 'valid':
                eval_loader = self.tkg_dataloder_valid_logic
                eval_dataset = self.tkg_dataset_valid
            elif mode == 'test':
                eval_loader = self.tkg_dataloder_test_logic
                eval_dataset = self.tkg_dataset_test

        # tkg_temp = self.tkg_dataset_train.tKG.to(self.device)
        ranking = []
        results = dict()
        for item in tqdm(eval_loader):
            h, r, t, ts, y = item  # B*4、E edge数量、B*N 实体one hot

            query_head = h.to(self.device)
            query_relation = r.to(self.device)
            query_tail = t.to(self.device)
            query_time = ts.to(self.device)
            y = y.to(self.device)

            # B*M B*M B*M B*M B*M
            # scores, query_answer = self.model(tkg_temp, query_head, query_relation, query_time, dataset=self.tkg_dataset_train, device=device)
            # scores = self.model(tkg_temp, query_head, query_relation, query_time, None, dataset=eval_dataset, device=self.device)

            # 不同类型得分函数
            w1 = 0.5
            w2 = 0.5
            scores = None
            if model_mode == 'gcn':
                scores = self.model.gcn_scores(self.tkg_temp, query_head, query_relation, query_time)
            elif model_mode == 'logic':
                # ent_emds, rel_emds = self.model.gen_gcn_emds(None)
                # X X*2
                if self.args.use_gcn == 1:
                    ent_emds, rel_emds, time_emds = self.model.gen_gcn_emds(self.tkg_temp)
                else:
                    ent_emds, rel_emds, time_emds = self.model.gen_gcn_emds(None)
                scores, tail_ents = self.model.logic_scores(ent_emds, rel_emds, time_emds, query_head, query_relation, query_time,
                                                       dataset=eval_dataset, device=self.device, args=self.args)
            elif model_mode == 'both':
                scores1 = self.model.gcn_scores(self.tkg_temp, query_head, query_relation, query_time)
                ent_emds, rel_emds = self.model.gen_gcn_emds(None)
                scores2 = self.model.logic_scores(ent_emds, rel_emds, query_head, query_relation, query_time,
                                                 dataset=eval_dataset, device=self.device)
                scores2_max = torch.max(scores2, dim=1, keepdim=True)[0]
                scores2 = scores2 / scores2_max
                scores2 = torch.pow(scores2, w1)  # 幂 次方
                scores = scores1 + scores2 * w2

            # print(scores)
            # print(tail_ents)
            # print(scores.size())
            # print(tail_ents.size())
            # scores转化 B*E

            batch_size = y.size()[0]

            scores_all = torch.zeros((batch_size, self.num_ent)).to(self.device)
            scores_all[[tail_ents[:, 0], tail_ents[:, 1]]] = scores
            pred = scores_all

            # pred = scores

            labels = y
            obj = query_tail

            b_range = torch.arange(batch_size).to(self.device)
            target_pred = pred[b_range, obj]  # [batch_size, 1], get the predictive score of obj
            # print(target_pred)
            # label=>-1000000, not label=>pred, filter out other objects with same sub&rel pair
            pred = torch.where(labels.bool(), -1e8, pred)
            pred[b_range, obj] = target_pred  # copy predictive score of obj to new pred
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]  # get the rank of each (sub, rel, obj)
            ranks = ranks.float()
            # print(ranks)
            results['count'] = torch.numel(ranks) + results.get('count', 0)  # number of predictions
            results['MR'] = torch.sum(ranks).item() + results.get('MR', 0)
            results['MRR'] = torch.sum(1.0 / ranks).item() + results.get('MRR', 0)
            for k in [1, 3, 10]:
                results[f'Hits@{k}'] = torch.numel(ranks[ranks <= k]) + results.get(f'Hits@{k}', 0)

            # print(results)
            # break
        count = results['count']
        for key_ in results.keys():
            results[key_] = round(results[key_] / count, 5)
        return results

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # 多GPU


def save_model(path, model, args):
    state = {
        'model': model.state_dict(),
        'args': vars(args)
    }
    torch.save(state, path)


def gen_result_str(results):
    return 'MR:{}  MRR:{}  Hits@1:{}  Hits@3:{}  Hits@10:{}'.format(results['MR'], results['MRR'], results['Hits@1'], results['Hits@3'], results['Hits@10'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', default='test_run', help='Set run name for saving/restoring models')
    parser.add_argument('--dataset', dest='dataset', default='icesw14', help='Dataset to use, default: FB15k-237')  # wn18rr ../data/wn18rr
    parser.add_argument('--score_func', dest='score_func', default='conve', help='Score Function for Link prediction')
    parser.add_argument('--opn', dest='opn', default='mult', help='Composition Operation to be used in CompGCN')

    parser.add_argument('--train_batch', dest='train_batch', default=32, type=int, help='Batch size')
    parser.add_argument('--test_batch', dest='test_batch', default=32, type=int, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', dest='epoch', type=int, default=3, help='Number of epochs')
    parser.add_argument('--l2', type=float, default=0.0000, help='L2 Regularization for Optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='Starting Learning Rate')
    parser.add_argument('--lable_smooth', dest='lable_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of processes to construct batches')
    parser.add_argument('--seed', dest='seed', default=95, type=int, help='Seed for randomization')  # 121 12315

    parser.add_argument('--restore', dest='restore', action='store_true', help='Restore from the previously saved model')
    parser.add_argument('--bias', dest='bias', action='store_true', help='Whether to use bias in the model')

    parser.add_argument('--num_bases', dest='num_bases', default=-1, type=int, help='Number of basis relation vectors to use')
    parser.add_argument('--gcn_dim', dest='gcn_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', dest='hidden_dim', default=50, type=int)
    parser.add_argument('--gcn_layer', dest='gcn_layer', default=1, type=int, help='number of gcn layers')
    parser.add_argument('--logic_layer', dest='logic_layer', default=3, type=int, help='number of logic layers')
    parser.add_argument('--gcn_drop', dest='gcn_drop', default=0.1, type=float)
    parser.add_argument('--hidden_drop', dest='hidden_drop', default=0.1, type=float)
    parser.add_argument('--max_nodes', dest='max_nodes', default=40, type=int)
    parser.add_argument('--sample_nodes', dest='sample_nodes', default=15, type=int)
    parser.add_argument('--sample_method', dest='sample_method', default=1, type=int)  # 1 2 3所有 前向 后向
    parser.add_argument('--sample_ratio', dest='sample_ratio', default=0.5, type=float)  # 采样权重
    parser.add_argument('--logic_ratio', dest='logic_ratio', default=0.8, type=float)  # 训练比例
    parser.add_argument('--score_method', dest='score_method', default='emd')  # emd att both
    parser.add_argument('--loss', dest='loss', default='max_min')  # bce max_min focal

    parser.add_argument('--act', dest='act', default='relu')
    parser.add_argument('--logic_reduce', dest='logic_reduce', default='sum')  # sum mean


    parser.add_argument('--time_score', dest='time_score', default=0, type=int)

    parser.add_argument('--use_gcn', dest='use_gcn', default=1, type=int)
    # parser.add_argument('--gcn_time', dest='gcn_time', action='store_true')
    # parser.add_argument('--use_logic', dest='use_logic', action='store_true')
    # parser.add_argument('--logic_time', dest='logic_time', action='store_true')




    args = parser.parse_args()
    print(args)
    # args.input_dim = args.gcn_dim  # 初始特征
    # args.embed_dim = args.k_w * args.k_h

    set_seed(args.seed)

    data_set = args.dataset
    data_dir = join('../data', data_set)
    result_dir = '../results'
    if os.path.exists(result_dir)==False:
        os.makedirs(result_dir)

    # save_dir = join(result_dir, data_set+'gcn_{}layer'.format(args.gcn_layer))

    load_dir = join(result_dir, data_set + 'gcn_{}layer'.format(args.gcn_layer))  # 预训练的gcn模型加载
    time_str = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
    save_dir = join(result_dir, data_set + '_logic_{}'.format(time_str))

    if os.path.exists(save_dir)==False:  # 保存路径
        os.makedirs(save_dir)

    logging.basicConfig(filename=join(save_dir, 'train.log'), format='%(asctime)s: %(message)s',level=logging.INFO, datefmt='%m/%d %I:%M:%S %p')


    runner = Runner(args)


    # gcn_epochs = 500  # GCN迭代
    logic_epochs = args.epoch  # 逻辑层迭代
    valid_metric = 'Hits@1'
    best_metric = -1.0



    logging.info('==============Training================')
    for epoch in range(logic_epochs):  # logic训练
        epoch_loss = runner.train_logic_epoch()
        logging.info("Epoch Loss-[{}]:{}".format(epoch+1, epoch_loss))
        print("Epoch Loss-[{}]:{}".format(epoch + 1, epoch_loss))

        results = runner.eval(mode='valid', model_mode='logic')
        results_str = gen_result_str(results)
        logging.info('==============Validing================')
        logging.info(results_str)
        print(epoch + 1, results_str)
        if results[valid_metric] > best_metric:
            save_model(join(save_dir, 'train_model.pt'), runner.model, args)
            best_metric = results[valid_metric]
    logging.info('==============Testing================')
    state = torch.load(join(save_dir, 'train_model.pt'))
    runner.model.load_state_dict(state['model'])
    results = runner.eval(mode='test', model_mode='logic')
    results_str = gen_result_str(results)
    logging.info(results_str)
    print(results_str)








