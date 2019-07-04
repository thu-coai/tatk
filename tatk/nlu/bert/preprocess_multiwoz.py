import json
import os
import zipfile
import sys
import pickle
from collections import Counter


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))

if __name__ == '__main__':
    mode = sys.argv[1]
    assert mode=='all' or mode=='usr' or mode=='sys'
    data_dir = '../../../../../data/multiwoz'
    processed_data_dir = 'multiwoz_{}_data'.format(mode)
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    data_key = ['val', 'test', 'train']
    data = {}
    for key in data_key:
        data[key] = read_zipped_json(os.path.join(data_dir,key+'.json.zip'), key+'.json')
        print('load {}, size {}'.format(key, len(data[key])))

    processed_data = {}
    all_da = []
    all_intent = []
    all_tag = []
    turn_num = 0
    for key in data_key:
        processed_data[key] = []
        for no, sess in data[key].items():
            for is_sys, turn in enumerate(sess['log']):
                if mode == 'usr' and is_sys % 2 == 1:
                    continue
                elif mode == 'sys' and is_sys % 2 == 0:
                    continue
                turn_num += 1
                tokens = turn["text"].split()
                dialog_act = {}
                for dacts in turn["span_info"]:
                    if dacts[0] not in dialog_act:
                        dialog_act[dacts[0]] = []
                    dialog_act[dacts[0]].append([dacts[1], " ".join(tokens[dacts[3]: dacts[4] + 1])])

                spans = turn["span_info"]
                tags = []
                for i in range(len(tokens)):
                    for span in spans:
                        if i == span[3]:
                            tags.append("B-" + span[0] + "+" + span[1])
                            break
                        if i > span[3] and i <= span[4]:
                            tags.append("I-" + span[0] + "+" + span[1])
                            break
                    else:
                        tags.append("O")

                intents = []
                for dacts in turn["dialog_act"]:
                    for dact in turn["dialog_act"][dacts]:
                        if dacts not in dialog_act or dact[0] not in [sv[0] for sv in dialog_act[dacts]]:
                            if dact[1] in ["none", "?", "yes", "no", "do nt care", "do n't care"]:
                                intents.append(dacts + "+" + dact[0] + "*" + dact[1])

                processed_data[key].append([tokens,tags,intents])
                if key == 'train':
                    all_da += [da for da in turn['dialog_act']]
                    all_intent += intents
                    all_tag += tags
    print('turn num:', turn_num)
    print('dialog act num:', len(set(all_da)))
    print('sentence label num:', len(set(all_intent)))
    print('tag num:', len(set(all_tag)))
    all_da = [x[0] for x in dict(Counter(all_da)).items() if x[1]]
    all_intent = [x[0] for x in dict(Counter(all_intent)).items() if x[1]]
    all_tag = [x[0] for x in dict(Counter(all_tag)).items() if x[1]]
    pickle.dump(processed_data, open(os.path.join(processed_data_dir, 'data.pkl'), 'wb'))
    pickle.dump(all_intent, open(os.path.join(processed_data_dir, 'intent_vocab.pkl'), 'wb'))
    pickle.dump(all_tag, open(os.path.join(processed_data_dir, 'tag_vocab.pkl'), 'wb'))
