"""
Preprocess multiwoz dataset
Usage:
    python preprocess
Require:
    - `data/data.json` from original multiwoz zip file (https://www.repository.cam.ac.uk/handle/1810/280608)
    - `data/delex.json` from multiwoz baseline preprocess code (clone https://github.com/budzianowski/multiwoz
        and run `python create_delex_data.py`, you will get this under `data/multi-woz/delex.json`)
    - `../../../../data/multiwoz/[train|val|test].json.zip` data file
    - `../../../../data/multiwoz/db` database dir
Output:
    - `data/entities.json`
    - `data/[train|val|test].json`
"""
import json
import os
import zipfile
from collections import Counter
import sys
import re
from copy import deepcopy


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


def extract_entities(db):
    entity_attr_list = {
        "[attraction_address]",
        "[restaurant_address]",
        "[attraction_phone]",
        "[restaurant_phone]",
        "[hotel_address]",
        "[restaurant_postcode]",
        "[attraction_postcode]",
        "[hotel_phone]",
        "[hotel_postcode]",
        "[hospital_phone]"
    }

    entity_type_list = {
        # attr values (relations)
        "[value_area]",  # restaurant, hotel, attraction
        "[value_pricerange]",  # restaurant, hotel, attraction
        "[value_day]",  # train
        "[value_food]",  # restaurant
        "[value_time]",  # train
        "[value_price]",  # train
        "[value_place]",  # train
        # entity names
        "[hotel_name]",
        "[train_id]",
        "[restaurant_name]",
        "[attraction_name]",
        "[hospital_department]"
    }

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

    def module(item, key1, key2):
        if key2 in item:
            value = normal(item[key2])
            if value not in entity_dict[key1]:
                entity_dict[key1].append(value)

    entity_dict = {}
    entity_set = entity_attr_list | entity_type_list
    for key in entity_set:
        entity_dict[key] = []

    for item in db['restaurant']:
        module(item, "[restaurant_address]", 'address')
        module(item, "[restaurant_phone]", 'phone')
        module(item, "[restaurant_postcode]", 'postcode')
        module(item, "[restaurant_name]", 'name')
        module(item, "[value_area]", 'area')
        module(item, "[value_pricerange]", 'pricerange')
        module(item, "[value_food]", 'food')

    for item in db['hotel']:
        module(item, "[hotel_address]", 'address')
        module(item, "[hotel_phone]", 'phone')
        module(item, "[hotel_postcode]", 'postcode')
        module(item, "[hotel_name]", 'name')
        module(item, "[value_area]", 'area')
        module(item, "[value_pricerange]", 'pricerange')

    for item in db['attraction']:
        module(item, "[attraction_address]", 'address')
        module(item, "[attraction_phone]", 'phone')
        module(item, "[attraction_postcode]", 'postcode')
        module(item, "[attraction_name]", 'name')
        module(item, "[value_area]", 'area')
        module(item, "[value_pricerange]", 'pricerange')
    entity_dict["[value_pricerange]"].remove('?')

    for item in db['hospital']:
        module(item, "[hospital_phone]", 'phone')
        module(item, "[hospital_department]", 'department')

    for item in db['train']:
        module(item, '[train_id]', 'trainID')
        module(item, '[value_day]', 'day')
        module(item, '[value_place]', 'departure')
        module(item, '[value_place]', 'destination')
        module(item, '[value_time]', 'arriveBy')
        module(item, '[value_time]', 'leaveAt')
        module(item, '[value_price]', 'price')
    for i, j in enumerate(entity_dict['[value_price]']):
        j = j.replace('_', '')
        entity_dict['[value_price]'][i] = j[:-7]

    return entity_dict


def simple(delex_data):
    # since the reference numbers have nothing to do with context, so we just treat them as symbols
    # since the taxi type and taxi phone number is random generated, and have no relation with the user's requests. (in this dataset, users never require certain color or type of taxies)
    # there is only one police station, so once the type of this slot is predicted, we get the true value.
    # there is only one hospital, so once the type of this slot is predicted, we get the true value
    symbol_list = {
        "[hotel_reference]",
        "[train_reference]",
        "[restaurant_reference]",
        "[attraction_reference]",
        "[hospital_reference]",
        "[taxi_type]",
        "[taxi_phone]",
        "[police_address]",
        "[police_phone]",
        "[police_postcode]",
        "[police_name]",
        "[hospital_postcode]",
        "[hospital_address]",
        "[hospital_name]",
        "[value_time]",
        "[value_price]",
        "[value_place]",
        "[value_day]",
        "[value_count]",
        "[train_id]"
    }

    # all the values in "entity_attr_list" almost only appear in system's utterence, so we remove them from user's utterence
    # for which we only predict the value's type and corresponding entity, and then query from the database
    entity_attr_list = {
        "[attraction_address]",
        "[restaurant_address]",
        "[attraction_phone]",
        "[restaurant_phone]",
        "[hotel_address]",
        "[restaurant_postcode]",
        "[attraction_postcode]",
        "[hotel_phone]",
        "[hotel_postcode]",
        "[hospital_phone]"
    }

    # the entities which should be queried from KG, it contains both relations and entities. Here we treat these relations (in fact the attributes of entities) also as entities.
    entity_type_list = {
        # attr values (relations)
        "[value_area]",
        "[value_pricerange]",
        "[value_food]",
        # entity names
        "[hotel_name]",
        "[restaurant_name]",
        "[attraction_name]",
        "[hospital_department]"
    }

    results = {}
    checkset = symbol_list | entity_attr_list | entity_type_list
    logset = set()

    def check(text, checkset):
        newset = set()

        for attr in checkset:
            if attr in text:
                newset.add(attr)
        checkset = checkset - newset
        return checkset

    def log(text, logset, key):
        if key in text:
            keylist = text.split(key)
            for i in range(1, len(keylist)):
                if not keylist[i].split():
                    continue
                word = keylist[i].split()[0]
                logset.add(word)
        return logset

    def normal(text):
        text = text.replace('_rd', '_road')
        text = text.replace('city_centre_north_bed_and_breakfast', 'city_centre_north_b_and_b')
        text = text.replace('alpha_milton_guest_house', 'alphamilton_guest_house')
        text = text.replace('gallery_at_12_a_high_street', 'gallery_at_twelve_a_high_street')
        text = text.replace('restaurant_17', 'restaurant_one_seven')
        text = text.replace('restaurant_2_two', 'restaurant_two_two')
        return text

    for title, item in delex_data.items():
        logs = item['log']
        turns = []
        for i, diag in enumerate(logs):
            text = diag['text'].strip()
            checkset = check(text, checkset)
            logset = log(text, logset, '[value_place]')
            text = normal(text)
            if not i % 2:
                turn = {'turn': i // 2}
                turn['usr'] = text
            else:
                turn['sys'] = text
                turns.append(turn)
        assert (i % 2)
        results[title] = turns

    return results


def lemmatize(raw_data, delex_data):
    def _replace(_data, _delex, _from, _to):
        def clean_replace_single(s, r, t, forward, backward, sidx=0):
            idx = s[sidx:].find(r)
            if idx == -1:
                return s, -1
            idx += sidx
            idx_r = idx + len(r)
            if backward:
                while idx > 0 and s[idx - 1]:
                    idx -= 1
            elif idx > 0 and s[idx - 1] != ' ':
                return s, -1

            if forward:
                while idx_r < len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
                    idx_r += 1
            elif idx_r != len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
                return s, -1
            return s[:idx] + t + s[idx_r:], idx_r

        p = re.compile(_to)
        index = 0
        for m in p.finditer(_data):
            _delex, index = clean_replace_single(_delex, _from, _from + '::' + m.group(), True, False, index)
        return _delex

    for title in raw_data.keys():
        data_diags = raw_data[title]
        delex_diags = delex_data[title]
        for i in range(len(data_diags)):
            data_turns = data_diags[i]
            delex_turns = delex_diags[i]
            delex_turns['usr'] = _replace(data_turns['usr'], delex_turns['usr'], '[value_price]',
                                          r'([0-9]{1,}[.][0-9]+)')
            delex_turns['sys'] = _replace(data_turns['sys'], delex_turns['sys'], '[value_price]',
                                          r'([0-9]{1,}[.][0-9]+)')
            delex_turns['usr'] = _replace(data_turns['usr'], delex_turns['usr'], '[value_time]',
                                          r'([0-1]?[0-9]|2[0-4]):([0-9]?[0-9])')
            delex_turns['sys'] = _replace(data_turns['sys'], delex_turns['sys'], '[value_time]',
                                          r'([0-1]?[0-9]|2[0-4]):([0-9]?[0-9])')
            delex_diags[i] = delex_turns

    return delex_data


def toSequicity(data, da):
    request_pair = {
        'addr': 'address',
        'area': 'area',
        'arrive': 'time',
        'car': 'car',
        'fee': 'price',
        'food': 'food',
        'id': 'id',
        'internet': 'boolean',
        'leave': 'time',
        'parking': 'boolean',
        'phone': 'phone',
        'post': 'postcode',
        'price': 'pricerange',
        'ref': 'reference',
        'stars': 'count',
        'time': 'time',
        'type': 'type'
    }
    sessList = set(da.keys())

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

    def read_act(da_session, turn):
        dialog_act = eval(str(da_session[turn]['dialog_act']).lower())
        for k, v in dialog_act.items():
            l = len(v)
            for j in range(l):
                dialog_act[k][j][0] = normal(v[j][0])
                dialog_act[k][j][1] = normal(v[j][1])
                if k.split('-')[-1] == 'request':
                    if dialog_act[k][j][0] in request_pair:
                        dialog_act[k][j][0] = request_pair[dialog_act[k][j][0]]
        return dialog_act

    def init_brief_state():
        return dict({'inform': [], 'request': []})

    def update_state(brief_state, dialog_act, delete_act, question_slot, current_domain):
        new_domain = current_domain
        for k, v in dialog_act.items():
            key = k.split('-')[0].lower()
            if key == 'general' or key == 'booking':
                continue
            if current_domain is None:
                current_domain = new_domain = key
                continue
            if key != current_domain:
                new_domain = key

        if new_domain != current_domain:
            current_domain = new_domain
            brief_state = init_brief_state()
            question_slot = []

        if delete_act:
            for k, v in delete_act.items():
                l = len(v)
                for j in range(l):
                    key = request_pair[v[j][0]] if v[j][0] in request_pair else v[j][0]
                    if key in question_slot:
                        question_slot.remove(key)
                        flag = False
                        v_ = brief_state['request']
                        l_ = len(v_)
                        for j_ in range(l_):
                            if v_[j_][0] == key:
                                v_.pop(j_)
                                flag = True
                                break
                        assert (flag)

        for k, v in dialog_act.items():
            domain = k.split('-')[0].lower()
            if domain != current_domain:
                continue
            key = k.split('-')[1].lower()
            if key in brief_state:
                l = len(v)
                for j in range(l):
                    if v[j] not in brief_state[key] and v[j][0] != 'none':
                        brief_state[key].append(v[j])
                        if key == 'request':
                            question_slot.append(v[j][0])

        return brief_state, question_slot, current_domain

    ff = []
    for n in data.keys():
        if n.split('.')[0] not in sessList:
            continue
        session = dict()
        session['title'] = n
        dials = data[n]
        dialogs = []
        try:
            da_session = da[n.split('.')[0]]['log']
        except:
            print(n)
            da_session = None
        brief_state = init_brief_state()
        question_slot = []
        current_domain = None
        for i in range(len(dials)):
            single_dialog = dict()
            single_dialog['turn'] = i
            single_dialog['usr'] = dict()
            single_dialog['sys'] = dict()
            usr = dials[i]['usr'].split()
            for j, word in enumerate(usr):
                if '::' in word:
                    if 'reference' in word:
                        usr[j] = word.split('::')[0]
                    else:
                        usr[j] = word.split('::')[1]
            single_dialog['usr']['transcript'] = ' '.join(usr)
            if da_session:
                dialog_act = read_act(da_session, 2 * i)
                delete_act = read_act(da_session, 2 * i - 1) if i > 0 else None
                brief_state, question_slot, current_domain = update_state(brief_state, dialog_act, delete_act,
                                                                          question_slot, current_domain)
                single_dialog['usr']['slu'] = deepcopy(brief_state)
            else:
                single_dialog['usr']['slu'] = dict()
            sys = dials[i]['sys'].split()
            for j, word in enumerate(sys):
                if '::' in word:
                    if 'reference' in word:
                        sys[j] = word.split('::')[0]
                    else:
                        sys[j] = word.split('::')[1]
            single_dialog['sys']['sent'] = ' '.join(sys)
            dialogs.append(single_dialog)
        session['dial'] = dialogs
        ff.append(session)
    return ff

if __name__ == '__main__':
    processed_data_dir = 'data/'
    raw_data = json.load(open(os.path.join(processed_data_dir, 'data.json')))
    delex_data = json.load(open(os.path.join(processed_data_dir, 'delex.json')))
    data_dir = '../../../../data/multiwoz'
    db_dir = os.path.join(data_dir, 'db')
    data_key = ['val', 'test', 'train']
    raw_split_data = {}
    for key in data_key:
        raw_split_data[key] = read_zipped_json(os.path.join(data_dir, key + '.json.zip'), key + '.json')
        print('load {}, size {}'.format(key, len(raw_split_data[key])))

    db = {
        'attraction': json.load(open(os.path.join(db_dir,'attraction_db.json'))),
        'hotel': json.load(open(os.path.join(db_dir,'hotel_db.json'))),
        'restaurant': json.load(open(os.path.join(db_dir,'restaurant_db.json'))),
        'police': json.load(open(os.path.join(db_dir,'police_db.json'))),
        'hospital': json.load(open(os.path.join(db_dir,'hospital_db.json'))),
        'taxi': json.load(open(os.path.join(db_dir,'taxi_db.json'))),
        'train': json.load(open(os.path.join(db_dir,'train_db.json')))
    }
    entity_dict = extract_entities(db)
    json.dump(entity_dict, open(os.path.join(processed_data_dir, 'entities.json'), 'w'), indent=4)

    simpled_raw_data = simple(raw_data)
    simpled_delex_data = simple(delex_data)
    lemmatized_data = lemmatize(simpled_raw_data, simpled_delex_data)
    for key in data_key:
        json.dump(toSequicity(lemmatized_data, raw_split_data[key]),
                  open(os.path.join(processed_data_dir, key+'.json'), 'w'),
                  indent=4)
