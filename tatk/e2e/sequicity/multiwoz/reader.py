import csv
import json
import os
import random
import re

from tatk.e2e.sequicity.reader import _ReaderBase, clean_replace
from tatk.e2e.sequicity.config import global_config as cfg


class MultiWozReader(_ReaderBase):
    def __init__(self):
        super().__init__()
        self._construct(cfg.train, cfg.dev, cfg.test, cfg.db)
        self.result_file = ''

    def _get_tokenized_data(self, raw_data, db_data, construct_vocab):
        requestable_keys = ['addr', 'area', 'fee', 'name', 'phone', 'post', 'price', 'type', 'department', 'internet',
                            'parking', 'stars', 'food', 'arrive', 'day', 'depart', 'dest', 'leave', 'ticket', 'id']

        tokenized_data = []
        vk_map = self._value_key_map(db_data)
        for dial_id, dial in enumerate(raw_data):
            tokenized_dial = []
            for turn in dial['dial']:
                turn_num = turn['turn']
                constraint = []
                requested = []
                for slot_act in turn['usr']['slu']:
                    if slot_act == 'inform':
                        slot_values = turn['usr']['slu'][slot_act]
                        for v in slot_values:
                            s = v[1]
                            if s not in ['dont_care', 'none']:
                                constraint.append(s)
                    elif slot_act == 'request':
                        slot_values = turn['usr']['slu'][slot_act]
                        for v in slot_values:
                            s = v[0]
                            if s in requestable_keys:
                                requested.append(s)
                degree = len(self.db_search(constraint))
                requested = sorted(requested)
                constraint.append('EOS_Z1')
                requested.append('EOS_Z2')
                user = turn['usr']['transcript'].split() + ['EOS_U']
                response = self._replace_entity(turn['sys']['sent'], vk_map, constraint).split() + ['EOS_M']
                response_origin = turn['sys']['sent'].split()
                tokenized_dial.append({
                    'dial_id': dial_id,
                    'turn_num': turn_num,
                    'user': user,
                    'response': response,
                    'response_origin': response_origin,
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
            response = clean_replace(response, v, k + '_SLOT')
        return response

    def _value_key_map(self, db_data):
        def normal(string):
            string = string.lower()
            string = re.sub(r'\s*-\s*', '', string)
            string = re.sub(r' ', '_', string)
            string = re.sub(r',', '_,', string)
            string = re.sub(r'\'', '_', string)
            string = re.sub(r'\.', '_.', string)
            string = re.sub(r'_+', '_', string)
            string = re.sub(r'children', 'child_-s', string)
            return string

        requestable_dict = {'address': 'addr',
                            'area': 'area',
                            'entrance fee': 'fee',
                            'name': 'name',
                            'phone': 'phone',
                            'postcode': 'post',
                            'pricerange': 'price',
                            'type': 'type',
                            'department': 'department',
                            'internet': 'internet',
                            'parking': 'parking',
                            'stars': 'stars',
                            'food': 'food',
                            'arriveBy': 'arrive',
                            'day': 'day',
                            'departure': 'depart',
                            'destination': 'dest',
                            'leaveAt': 'leave',
                            'price': 'ticket',
                            'trainId': 'id'}
        value_key = {}
        for db_entry in db_data:
            for k, v in db_entry.items():
                if k in requestable_dict:
                    value_key[normal(v)] = requestable_dict[k]
        return value_key

    def _get_encoded_data(self, tokenized_data):
        encoded_data = []
        for dial in tokenized_data:
            encoded_dial = []
            prev_response = []
            for turn in dial:
                user = self.vocab.sentence_encode(turn['user'])
                response = self.vocab.sentence_encode(turn['response'])
                response_origin = ' '.join(turn['response_origin'])
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
                    'response_origin': response_origin,
                    'bspan': constraint + requested,
                    'u_len': len(prev_response + user),
                    'm_len': len(response),
                    'degree': degree,
                })
                # modified
                prev_response = response
            encoded_data.append(encoded_dial)
        return encoded_data

    def _get_clean_db(self, raw_db_data):
        for entry in raw_db_data:
            for k, v in list(entry.items()):
                if not isinstance(v, str) or v == '?':
                    entry.pop(k)

    def _construct(self, train_json_path, dev_json_path, test_json_path, db_json_path):
        """
        construct encoded train, dev, test set.
        :param train_json_path:
        :param dev_json_path:
        :param test_json_path:
        :param db_json_path: list
        :return:
        """
        construct_vocab = False
        if not os.path.isfile(cfg.vocab_path):
            construct_vocab = True
            print('Constructing vocab file...')
        with open(train_json_path) as f:
            train_raw_data = json.loads(f.read().lower())
        with open(dev_json_path) as f:
            dev_raw_data = json.loads(f.read().lower())
        with open(test_json_path) as f:
            test_raw_data = json.loads(f.read().lower())
        db_data = list()
        for domain_db_json_path in db_json_path:
            with open(domain_db_json_path) as f:
                db_data += json.loads(f.read().lower())
        self._get_clean_db(db_data)
        self.db = db_data

        train_tokenized_data = self._get_tokenized_data(train_raw_data, db_data, construct_vocab)
        dev_tokenized_data = self._get_tokenized_data(dev_raw_data, db_data, construct_vocab)
        test_tokenized_data = self._get_tokenized_data(test_raw_data, db_data, construct_vocab)
        if construct_vocab:
            self.vocab.construct(cfg.vocab_size)
            self.vocab.save_vocab(cfg.vocab_path)
        else:
            self.vocab.load_vocab(cfg.vocab_path)
        self.train = self._get_encoded_data(train_tokenized_data)
        self.dev = self._get_encoded_data(dev_tokenized_data)
        self.test = self._get_encoded_data(test_tokenized_data)
        random.shuffle(self.train)
        random.shuffle(self.dev)
        random.shuffle(self.test)

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

    def wrap_result(self, turn_batch, gen_m, gen_z, eos_syntax=None, prev_z=None):
        """
        wrap generated results
        :param gen_z:
        :param gen_m:
        :param turn_batch: dict of [i_1,i_2,...,i_b] with keys
        :return:
        """

        results = []
        if eos_syntax is None:
            eos_syntax = {'response': 'EOS_M', 'user': 'EOS_U', 'bspan': 'EOS_Z2'}
        batch_size = len(turn_batch['user'])
        for i in range(batch_size):
            entry = {}
            if prev_z is not None:
                src = prev_z[i] + turn_batch['user'][i]
            else:
                src = turn_batch['user'][i]
            for key in turn_batch:
                entry[key] = turn_batch[key][i]
                if key in eos_syntax:
                    entry[key] = self.vocab.sentence_decode(entry[key], eos=eos_syntax[key])
            if gen_z:
                entry['generated_bspan'] = self.vocab.sentence_decode(gen_z[i], eos='EOS_Z2')
            else:
                entry['generated_bspan'] = ''
            if gen_m:
                entry['generated_response'] = self.vocab.sentence_decode(gen_m[i], eos='EOS_M')
                constraint_request = entry['generated_bspan'].split()
                constraints = constraint_request[:constraint_request.index('EOS_Z1')] if 'EOS_Z1' \
                                                                                         in constraint_request else constraint_request
                for j, ent in enumerate(constraints):
                    constraints[j] = ent.replace('_', ' ')
                degree = self.db_search(constraints)
                # print('constraints',constraints)
                # print('degree',degree)
                venue = random.sample(degree, 1)[0] if degree else dict()
                l = [self.vocab.decode(_) for _ in gen_m[i]]
                if 'EOS_M' in l:
                    l = l[:l.index('EOS_M')]
                l_origin = []
                for word in l:
                    if 'SLOT' in word:
                        word = word[:-5]
                        if word in venue.keys():
                            value = venue[word]
                            if value != '?':
                                l_origin.append(value.replace(' ', '_'))
                    else:
                        l_origin.append(word)
                entry['generated_response_origin'] = ' '.join(l_origin)
            else:
                entry['generated_response'] = ''
                entry['generated_response_origin'] = ''
            results.append(entry)
        write_header = False
        if not self.result_file:
            self.result_file = open(cfg.result_path, 'w')
            self.result_file.write(str(cfg))
            write_header = True

        field = ['dial_id', 'turn_num', 'user', 'generated_bspan', 'bspan', 'generated_response', 'response', 'u_len',
                 'm_len', 'supervised', 'generated_response_origin', 'response_origin']
        for result in results:
            del_k = []
            for k in result:
                if k not in field:
                    del_k.append(k)
            for k in del_k:
                result.pop(k)
        writer = csv.DictWriter(self.result_file, fieldnames=field)
        if write_header:
            self.result_file.write('START_CSV_SECTION\n')
            writer.writeheader()
        writer.writerows(results)
        return results
