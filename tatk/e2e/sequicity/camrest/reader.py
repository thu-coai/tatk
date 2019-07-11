import json
import os
import random
import re
from nltk.tokenize import word_tokenize

from tatk.e2e.sequicity.reader import _ReaderBase, clean_replace
from tatk.e2e.sequicity.config import global_config as cfg


class CamRest676Reader(_ReaderBase):
    def __init__(self):
        super().__init__()
        self._construct(cfg.data, cfg.db)
        self.result_file = ''

    def _get_tokenized_data(self, raw_data, db_data, construct_vocab):
        tokenized_data = []
        vk_map = self._value_key_map(db_data)
        for dial_id, dial in enumerate(raw_data):
            tokenized_dial = []
            for turn in dial['dial']:
                turn_num = turn['turn']
                constraint = []
                requested = []
                for slot in turn['usr']['slu']:
                    if slot['act'] == 'inform':
                        s = slot['slots'][0][1]
                        if s not in ['dontcare', 'none']:
                            constraint.extend(word_tokenize(s))
                    else:
                        requested.extend(word_tokenize(slot['slots'][0][1]))
                degree = len(self.db_search(constraint))
                requested = sorted(requested)
                constraint.append('EOS_Z1')
                requested.append('EOS_Z2')
                user = word_tokenize(turn['usr']['transcript']) + ['EOS_U']
                response = word_tokenize(self._replace_entity(turn['sys']['sent'], vk_map, constraint)) + ['EOS_M']
                tokenized_dial.append({
                    'dial_id': dial_id,
                    'turn_num': turn_num,
                    'user': user,
                    'response': response,
                    'constraint': constraint,
                    'requested': requested,
                    'degree': degree,
                })
                if construct_vocab:
                    for word in user + response + constraint + requested:
                        self.vocab.add_item(word)
            tokenized_data.append(tokenized_dial)
        return tokenized_data

    def _replace_entity(self, response, vk_map, constraint):
        response = re.sub('[cC][., ]*[bB][., ]*\d[., ]*\d[., ]*\w[., ]*\w', 'postcode_SLOT', response)
        response = re.sub('\d{5}\s?\d{6}', 'phone_SLOT', response)
        constraint_str = ' '.join(constraint)
        for v, k in sorted(vk_map.items(), key=lambda x: -len(x[0])):
            start_idx = response.find(v)
            if start_idx == -1 \
                    or (start_idx != 0 and response[start_idx - 1] != ' ') \
                    or (v in constraint_str):
                continue
            if k not in ['name', 'address']:
                response = clean_replace(response, v, k + '_SLOT', forward=True, backward=False)
            else:
                response = clean_replace(response, v, k + '_SLOT', forward=False, backward=False)
        return response

    def _value_key_map(self, db_data):
        requestable_keys = ['address', 'name', 'phone', 'postcode', 'food', 'area', 'pricerange']
        value_key = {}
        for db_entry in db_data:
            for k, v in db_entry.items():
                if k in requestable_keys:
                    value_key[v] = k
        return value_key

    def _get_encoded_data(self, tokenized_data):
        encoded_data = []
        for dial in tokenized_data:
            encoded_dial = []
            prev_response = []
            for turn in dial:
                user = self.vocab.sentence_encode(turn['user'])
                response = self.vocab.sentence_encode(turn['response'])
                constraint = self.vocab.sentence_encode(turn['constraint'])
                requested = self.vocab.sentence_encode(turn['requested'])
                degree = self._degree_vec_mapping(turn['degree'])
                turn_num = turn['turn_num']
                dial_id = turn['dial_id']

                # final input
                encoded_dial.append({
                    'dial_id': dial_id,
                    'turn_num': turn_num,
                    'user': prev_response + user,
                    'response': response,
                    'bspan': constraint + requested,
                    'u_len': len(prev_response + user),
                    'm_len': len(response),
                    'degree': degree,
                })
                # modified
                prev_response = response
            encoded_data.append(encoded_dial)
        return encoded_data

    def _split_data(self, encoded_data, split):
        """
        split data into train/dev/test
        :param encoded_data: list
        :param split: tuple / list
        :return:
        """
        total = sum(split)
        dev_thr = len(encoded_data) * split[0] // total
        test_thr = len(encoded_data) * (split[0] + split[1]) // total
        train, dev, test = encoded_data[:dev_thr], encoded_data[dev_thr:test_thr], encoded_data[test_thr:]
        return train, dev, test

    def _construct(self, data_json_path, db_json_path):
        """
        construct encoded train, dev, test set.
        :param data_json_path:
        :param db_json_path:
        :return:
        """
        construct_vocab = False
        if not os.path.isfile(cfg.vocab_path):
            construct_vocab = True
            print('Constructing vocab file...')
        raw_data_json = open(data_json_path)
        raw_data = json.loads(raw_data_json.read().lower())
        db_json = open(db_json_path)
        db_data = json.loads(db_json.read().lower())
        self.db = db_data
        tokenized_data = self._get_tokenized_data(raw_data, db_data, construct_vocab)
        if construct_vocab:
            self.vocab.construct(cfg.vocab_size)
            self.vocab.save_vocab(cfg.vocab_path)
        else:
            self.vocab.load_vocab(cfg.vocab_path)
        encoded_data = self._get_encoded_data(tokenized_data)
        self.train, self.dev, self.test = self._split_data(encoded_data, cfg.split)
        random.shuffle(self.train)
        random.shuffle(self.dev)
        random.shuffle(self.test)
        raw_data_json.close()
        db_json.close()

    def db_search(self, constraints):
        match_results = []
        for entry in self.db:
            entry_values = ' '.join(entry.values())
            match = True
            for c in constraints:
                if c not in entry_values:
                    match = False
                    break
            if match:
                match_results.append(entry)
        return match_results
