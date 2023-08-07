import os
from os.path import join

from utils.data_utils import load_fact, load_json, load_pickle, save_pickle

dataset_name = 'icews14'

old_data_dir = '../data/'
new_data_dir = '../data_static4red/'

dataset_dir = old_data_dir + dataset_name
out_dir = new_data_dir + dataset_name
if os.path.exists(new_data_dir) == False:
    os.mkdir(new_data_dir)
if os.path.exists(out_dir) == False:
    os.mkdir(out_dir)


entity2id = load_json(join(dataset_dir, 'entity2id.json'))
relation2id = load_json(join(dataset_dir, 'relation2id.json'))


id2rel_dic = {}
for rel in relation2id:
    id2rel_dic[relation2id[rel]] = rel
id2relation = []
for i in range(len(id2rel_dic)):
    id2relation.append(id2rel_dic[i])

id2ent_dic = {}
for ent in entity2id:
    id2ent_dic[entity2id[ent]] = ent
id2entity = []
for i in range(len(id2ent_dic)):
    id2entity.append(id2ent_dic[i])

train = load_fact(join(dataset_dir, 'train.txt'))
valid = load_fact(join(dataset_dir, 'valid.txt'))
test = load_fact(join(dataset_dir, 'test.txt'))
print(len(train))
print(len(valid))
print(len(test))

def save_list(_list, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for i, item in enumerate(_list):
            # f.write('{}\t{}\n'.format(i, item))
            f.write('{}\n'.format(item))

def save_fact(_list, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for i, item in enumerate(_list):
            h, r, t, ts = item
            f.write('{}\t{}\t{}\n'.format(h, r, t))

# save_list(id2entity, join(out_dir, 'entities.dict'))
# save_list(id2relation, join(out_dir, 'relations.dict'))
#
# save_fact(train, join(out_dir, 'train.txt'))
# save_fact(valid, join(out_dir, 'valid.txt'))
# save_fact(test, join(out_dir, 'test.txt'))

save_list(id2entity, join(out_dir, 'entities.txt'))
save_list(id2relation, join(out_dir, 'relations.txt'))

import random
random.shuffle(train)
len_fact = int(len(train) / 2)
facts = train[:len_fact]
train = train[len_fact:]
save_fact(facts, join(out_dir, 'facts.txt'))
save_fact(train, join(out_dir, 'train.txt'))
save_fact(valid, join(out_dir, 'valid.txt'))
save_fact(test, join(out_dir, 'test.txt'))




