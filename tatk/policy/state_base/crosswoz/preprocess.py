import json
import os
import zipfile
import sys
from collections import Counter


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def preprocess():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, '../../../../data/crosswoz')
    processed_data_dir = os.path.join(cur_dir, 'processed_data')
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    data_key = ['train', 'val', 'test']
    data = {}
    for key in data_key:
        data[key] = read_zipped_json(os.path.join(data_dir, key + '.json.zip'), key + '.json')
        print('load {}, size {}'.format(key, len(data[key])))

    processed_data = {}

    id_dim = 5
    domain_dim = 5
    slot_dim = 57
    value_dim = 2
    expressed_dim = 2




    max_num_semantic_tuples = 0
    slot_num = []
    sys_da_intent_dim = 4

    for key in data_key:
        processed_data[key] = []
        for no, sess in data[key].items():
            last_user_state = sess['goal']
            for i, turn in enumerate(sess['messages']):
                if turn['role'] == 'sys':
                    sys_da_intent_dim += [x[0] for x in turn['dialog_act']]
                pass
    print(len(set(sys_da_intent_dim)))



if __name__ == '__main__':
    preprocess()