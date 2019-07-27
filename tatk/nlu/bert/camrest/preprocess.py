"""
Preprocess camrest data for BertNLU.

Usage:
    python preprocess [mode=all|usr|sys]
    mode: which side data will be use

Require:
    - ``../../../../data/camrest/[train|val|test].json.zip`` data file

Output:
    - ``data/[mode]_data/``: processed data dir
        - ``data.pkl``: data[data_key=train|val|test] is a list of [tokens,tags,intents],
            tokens: list of words; tags: list of BIO tags(e.g. B-domain-intent+slot); intent: list of 'domain-intent+slot*value'.
        - ``intent_vocab.pkl``: list of all intents (format: 'domain-intent+slot*value')
        - ``tag_vocab.pkl``: list of all tags (format: 'O'|'B-domain-intent+slot'|'B-domain-intent+slot')
"""
import json
import os
import zipfile
import sys
import pickle
from collections import Counter
from nltk.tokenize import word_tokenize


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def phrase_in_utt(phrase, utt):
    phrase_low = phrase.lower()
    utt_low = utt.lower()
    return (' ' + phrase_low in utt_low) or utt_low.startswith(phrase_low)


def phrase_idx_utt(phrase, utt):
    phrase_low = phrase.lower()
    utt_low = utt.lower()
    if ' ' + phrase_low in utt_low or utt_low.startswith(phrase_low):
        return get_idx(phrase_low, utt_low)
    return None


def get_idx(phrase, utt):
    char_index_begin = utt.index(phrase)
    char_index_end = char_index_begin + len(phrase)
    word_index_begin = len(utt[:char_index_begin].split())
    word_index_end = len(utt[:char_index_end].split()) - 1
    return word_index_begin, word_index_end


def das2tags(sen, das, all_tag=None):
    tokens = word_tokenize(sen)
    new_sen = ' '.join(tokens)
    new_das = {}
    span_info = []
    intents = []
    for da, svs in das.items():
        new_das.setdefault(da, [])
        if da == 'inform':
            for s, v in svs:
                v = ' '.join(word_tokenize(v))
                if v != 'dontcare' and phrase_in_utt(v, new_sen):
                    word_index_begin, word_index_end = phrase_idx_utt(v, new_sen)
                    span_info.append((da, s, v, word_index_begin, word_index_end))
                else:
                    intents.append(da + '+' + s + '*' + v)
                new_das[da].append([s, v])
        else:
            for s, v in svs:
                new_das[da].append([s, v])
                intents.append(da + '+' + s + '*' + v)
    tags = []
    for i, _ in enumerate(tokens):
        for span in span_info:
            if i == span[3]:
                tag = "B-" + span[0] + "+" + span[1]
                if all_tag and tag not in all_tag:
                    tags.append('O')
                else:
                    tags.append(tag)
                break
            if span[3] < i <= span[4]:
                tag = "I-" + span[0] + "+" + span[1]
                if all_tag and tag not in all_tag:
                    tags.append('O')
                else:
                    tags.append(tag)
                break
        else:
            tags.append("O")

    return tokens, tags, intents, new_das


def preprocess(mode):
    assert mode == 'all' or mode == 'usr' or mode == 'sys'
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, '../../../../data/camrest')
    processed_data_dir = os.path.join(cur_dir, 'data/{}_data'.format(mode))
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    data_key = ['train', 'val', 'test']
    data = {}
    for key in data_key:
        data[key] = read_zipped_json(os.path.join(data_dir, key + '.json.zip'), key + '.json')
        print('load {}, size {}'.format(key, len(data[key])))

    processed_data = {}
    all_da = []
    all_intent = []
    all_tag = []
    for key in data_key:
        processed_data[key] = []
        for dialog in data[key]:
            no = dialog['dialogue_id']
            for turn in dialog['dial']:
                if mode == 'usr' or mode == 'all':
                    if key == 'train':
                        tokens, tags, intents, new_das = das2tags(turn['usr']['transcript'], turn['usr']['dialog_act'])
                    else:
                        tokens, tags, intents, new_das = das2tags(turn['usr']['transcript'], turn['usr']['dialog_act'],
                                                                  all_tag)
                    processed_data[key].append([tokens, tags, intents, new_das])
                    if key == 'train':
                        all_da += [da for da in turn['usr']['dialog_act']]
                        all_intent += intents
                        all_tag += tags
                if mode == 'sys' or mode == 'all':
                    if key == 'train':
                        tokens, tags, intents, new_das = das2tags(turn['sys']['sent'], turn['sys']['dialog_act'])
                    else:
                        tokens, tags, intents, new_das = das2tags(turn['sys']['sent'], turn['sys']['dialog_act'],
                                                                  all_tag)

                    processed_data[key].append([tokens, tags, intents, new_das])
                    if key == 'train':
                        all_da += [da for da in turn['sys']['dialog_act']]
                        all_intent += intents
                        all_tag += tags

        if key == 'train':
            all_da = [x[0] for x in dict(Counter(all_da)).items() if x[1]]
            all_intent = [x[0] for x in dict(Counter(all_intent)).items() if x[1]]
            all_tag = [x[0] for x in dict(Counter(all_tag)).items() if x[1]]

    for key in data_key:
        print('loaded {}, size {}'.format(key, len(processed_data[key])))
    print('dialog act num:', len(all_da))
    print('sentence label num:', len(all_intent))
    print('tag num:', len(all_tag))
    pickle.dump(processed_data, open(os.path.join(processed_data_dir, 'data.pkl'), 'wb'))
    pickle.dump(all_intent, open(os.path.join(processed_data_dir, 'intent_vocab.pkl'), 'wb'))
    pickle.dump(all_tag, open(os.path.join(processed_data_dir, 'tag_vocab.pkl'), 'wb'))


if __name__ == '__main__':
    mode = sys.argv[1]
    preprocess(mode)
