import pickle
import json

def load_fact(file_path):
    facts = []
    with open(file_path, encoding='utf8') as f:
        for line in f:
            facts.append(line.strip().split('\t'))
    return facts


def load_json(file_path):
    with open(file_path, encoding='utf8') as f:
        json_ = json.load(f)
    return json_


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        item = pickle.load(f)
    return item


def save_pickle(a, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(a, f)

