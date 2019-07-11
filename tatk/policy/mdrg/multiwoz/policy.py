#!/usr/bin/env python
# coding: utf-8
from __future__ import division, print_function, unicode_literals

import argparse
import json
import os
import shutil
import time
import re

import numpy as np
import torch

from tatk.policy.policy import Policy
from tatk.policy.mdrg.multiwoz.utils import delexicalize, util, dbPointer
from tatk.policy.mdrg.multiwoz.utils.nlp import normalize
from tatk.policy.mdrg.multiwoz.evaluator import evaluateModel
from tatk.policy.mdrg.multiwoz.model import Model
from tatk.policy.mdrg.multiwoz.create_delex_data import delexicaliseReferenceNumber, get_dial

from tatk.util.multiwoz.state import default_state

parser = argparse.ArgumentParser(description='S2S')
parser.add_argument('--no_cuda', type=util.str2bool, nargs='?', const=True, default=True, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--no_models', type=int, default=20, help='how many models to evaluate')
parser.add_argument('--original', type=str, default='model/model/', help='Original path.')

parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--use_emb', type=str, default='False')

parser.add_argument('--beam_width', type=int, default=10, help='Beam width used in beamsearch')
parser.add_argument('--write_n_best', type=util.str2bool, nargs='?', const=True, default=False, help='Write n-best list (n=beam_width)')

parser.add_argument('--model_path', type=str, default='model/model/translate.ckpt', help='Path to a specific model checkpoint.')
parser.add_argument('--model_dir', type=str, default='data/multi-woz/model/model/')
parser.add_argument('--model_name', type=str, default='translate.ckpt')

parser.add_argument('--valid_output', type=str, default='model/data/val_dials/', help='Validation Decoding output dir path')
parser.add_argument('--decode_output', type=str, default='model/data/test_dials/', help='Decoding output dir path')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")


def load_config(args):
    config = util.unicode_to_utf8(
        json.load(open('%s.json' % args.model_path, 'rb')))
    for key, value in args.__args.items():
        try:
            config[key] = value.value
        except:
            config[key] = value

    return config


def loadModelAndData(num):
    # Load dictionaries
    with open('data/input_lang.index2word.json') as f:
        input_lang_index2word = json.load(f)
    with open('data/input_lang.word2index.json') as f:
        input_lang_word2index = json.load(f)
    with open('data/output_lang.index2word.json') as f:
        output_lang_index2word = json.load(f)
    with open('data/output_lang.word2index.json') as f:
        output_lang_word2index = json.load(f)

    # Reload existing checkpoint
    model = Model(args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index)
    if args.load_param:
        model.loadModel(iter=num)

    # Load data
    if os.path.exists(args.decode_output):
        shutil.rmtree(args.decode_output)
        os.makedirs(args.decode_output)
    else:
        os.makedirs(args.decode_output)

    if os.path.exists(args.valid_output):
        shutil.rmtree(args.valid_output)
        os.makedirs(args.valid_output)
    else:
        os.makedirs(args.valid_output)

    # Load validation file list:
    with open('data/val_dials.json') as outfile:
        val_dials = json.load(outfile)

    # Load test file list:
    with open('data/test_dials.json') as outfile:
        test_dials = json.load(outfile)
    return model, val_dials, test_dials


def addBookingPointer(task, turn, pointer_vector):
    """Add information about availability of the booking option."""
    # Booking pointer
    rest_vec = np.array([1, 0])
    if True:
        # if turn['metadata']['restaurant'].has_key("book"):
        if "book" in turn['metadata']['restaurant']:
            # if turn['metadata']['restaurant']['book'].has_key("booked"):
            if "booked" in turn['metadata']['restaurant']['book']:
                if turn['metadata']['restaurant']['book']["booked"]:
                    if "reference" in turn['metadata']['restaurant']['book']["booked"][0]:
                        rest_vec = np.array([0, 1])

    hotel_vec = np.array([1, 0])
    if True:
        # if turn['metadata']['hotel'].has_key("book"):
        if "book" in turn['metadata']['hotel']:
            # if turn['metadata']['hotel']['book'].has_key("booked"):
            if "booked" in turn['metadata']['hotel']['book']:
                if turn['metadata']['hotel']['book']["booked"]:
                    if "reference" in turn['metadata']['hotel']['book']["booked"][0]:
                        hotel_vec = np.array([0, 1])

    train_vec = np.array([1, 0])
    if True:
        # if turn['metadata']['train'].has_key("book"):
        if "book" in turn['metadata']['train']:
            # if turn['metadata']['train']['book'].has_key("booked"):
            if "booked" in turn['metadata']['train']['book']:
                if turn['metadata']['train']['book']["booked"]:
                    if "reference" in turn['metadata']['train']['book']["booked"][0]:
                        train_vec = np.array([0, 1])

    pointer_vector = np.append(pointer_vector, rest_vec)
    pointer_vector = np.append(pointer_vector, hotel_vec)
    pointer_vector = np.append(pointer_vector, train_vec)

    return pointer_vector

def addDBPointer(turn):
    """Create database pointer for all related domains."""
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    pointer_vector = np.zeros(6 * len(domains))
    for domain in domains:
        num_entities = dbPointer.queryResult(domain, turn)
        pointer_vector = dbPointer.oneHotVector(num_entities, domain, pointer_vector)

    return pointer_vector


def decodeWrapper():
    # Load config file
    with open(args.model_path + '.config') as f:
        add_args = json.load(f)
        for k, v in add_args.items():
            setattr(args, k, v)

        args.mode = 'test'
        args.load_param = True
        args.dropout = 0.0
        assert args.dropout == 0.0

    # Start going through models
    args.original = args.model_path
    for ii in range(1, args.no_models + 1):
        print(70 * '-' + 'EVALUATING EPOCH %s' % ii)
        args.model_path = args.model_path + '-' + str(ii)
        try:
            decode(ii)
        except:
            print('cannot decode')

        args.model_path = args.original


def createDelexData(dialogue):
    """Main function of the script - loads delexical dictionary,
    goes through each dialogue and does:
    1) data normalization
    2) delexicalization
    3) addition of database pointer
    4) saves the delexicalized data
    """

    # create dictionary of delexicalied values that then we will search against, order matters here!
    dic = delexicalize.prepareSlotValuesIndependent()
    delex_data = {}

    # fin1 = open('data/multi-woz/data.json', 'r')
    # data = json.load(fin1)

        # dialogue = data[dialogue_name]
    dial = dialogue['cur']
    idx_acts = 1

    for idx, turn in enumerate(dial['log']):
        # print(idx)
        # print(turn)
        # normalization, split and delexicalization of the sentence
        sent = normalize(turn['text'])

        words = sent.split()
        sent = delexicalize.delexicalise(' '.join(words), dic)

        # parsing reference number GIVEN belief state
        sent = delexicaliseReferenceNumber(sent, turn)

        # changes to numbers only here
        digitpat = re.compile('\d+')
        sent = re.sub(digitpat, '[value_count]', sent)
        # print(sent)

        # delexicalized sentence added to the dialogue
        dial['log'][idx]['text'] = sent

        if idx % 2 == 1:  # if it's a system turn
            # add database pointer
            pointer_vector = addDBPointer(turn)
            # add booking pointer
            pointer_vector = addBookingPointer(dial, turn, pointer_vector)

            # print pointer_vector
            dial['log'][idx - 1]['db_pointer'] = pointer_vector.tolist()

        idx_acts += 1

    dial = get_dial(dial)

    if dial:
        dialogue = {}
        dialogue['usr'] = []
        dialogue['sys'] = []
        dialogue['db'] = []
        dialogue['bs'] = []
        for turn in dial:
            # print(turn)
            dialogue['usr'].append(turn[0])
            dialogue['sys'].append(turn[1])
            dialogue['db'].append(turn[2])
            dialogue['bs'].append(turn[3])

    delex_data['cur'] = dialogue

    return delex_data


def decode(data, model):
    # model, val_dials, test_dials = loadModelAndData(num)

    for ii in range(1):
        if ii == 0:
            print(50 * '-' + 'GREEDY')
            model.beam_search = False
        else:
            print(50 * '-' + 'BEAM')
            model.beam_search = True

        # VALIDATION
        val_dials_gen = {}
        valid_loss = 0
        # for name, val_file in val_dials.items():
        for i in range(1):
            val_file = data['cur']
            input_tensor = [];  target_tensor = [];bs_tensor = [];db_tensor = []
            input_tensor, target_tensor, bs_tensor, db_tensor = util.loadDialogue(model, val_file, input_tensor, target_tensor, bs_tensor, db_tensor)
            # create an empty matrix with padding tokens
            input_tensor, input_lengths = util.padSequence(input_tensor)
            target_tensor, target_lengths = util.padSequence(target_tensor)
            bs_tensor = torch.tensor(bs_tensor, dtype=torch.float, device=device)
            db_tensor = torch.tensor(db_tensor, dtype=torch.float, device=device)

            output_words, loss_sentence = model.predict(input_tensor, input_lengths, target_tensor, target_lengths,
                                                        db_tensor, bs_tensor)

            valid_loss += 0
            return output_words[-1]


def loadModel(num):
    # Load dictionaries
    with open('data/input_lang.index2word.json') as f:
        input_lang_index2word = json.load(f)
    with open('data/input_lang.word2index.json') as f:
        input_lang_word2index = json.load(f)
    with open('data/output_lang.index2word.json') as f:
        output_lang_index2word = json.load(f)
    with open('data/output_lang.word2index.json') as f:
        output_lang_word2index = json.load(f)

    # Reload existing checkpoint
    model = Model(args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index)
    # print(model.model_name)
    # print(model.model_dir)
    if args.load_param:
        model.loadModel(iter=num)

    return model




class MDRGWordPolicy(Policy):
    def __init__(self, num=1):
        with open(os.path.join(os.path.dirname(__file__), args.model_path + '.config'), 'r') as f:
            add_args = json.load(f)
            for k, v in add_args.items():
                setattr(args, k, v)

            args.mode = 'test'
            args.load_param = True
            args.dropout = 0.0
            assert args.dropout == 0.0

        # Start going through models
        args.original = args.model_path
        args.model_path = args.original
        self.model = loadModel(num)
        self.dial = {"cur": {"log": []}}


    def predict(self, state):
        last_usr_uttr = state['history'][-1][-1]
        usr_turn = {"text": last_usr_uttr, "metadata": ""}
        sys_turn = {"text": "placeholder " * 50, "metadata": state['belief_state']}
        self.dial['cur']['log'].append(usr_turn)
        self.dial['cur']['log'].append(sys_turn)
        # print(self.dial)

        self.normalized_dial = createDelexData(self.dial)
        response = decode(self.normalized_dial, self.model)
        self.dial['cur']['log'][-1]['text'] = response

        return response

    def init_session(self):
        self.dial = {"cur": {"log": []}}

def main():
    s = default_state()
    s['history'] = [['null', 'I want a korean restaurant in the centre.']]
    s['belief_state']['attraction']['semi']['area'] = 'centre'
    s['belief_state']['restaurant']['semi']['area'] = 'centre'
    s['belief_state']['restaurant']['semi']['food'] = 'korean'
    testPolicy = MDRGWordPolicy()
    print(testPolicy.predict(s))


if __name__ == '__main__':
    main()
