import os
import random

ACTION_NUM = 10
TMP_PATH = 'data/tmp'

def read_sents():
    """Load sents from raw data."""
    data_dir = 'data/negotiate'
    file_names = ['train', 'val', 'test']
    sents = []
    for name in file_names:
        file = os.path.join(data_dir, name+'.txt')
        with open(file) as f:
            lines = f.readlines()
        for line in lines:
            # print(line.strip())
            start = line.index('<dialogue>') + 10
            end = line.index('</dialogue>')
            session = line[start:end].strip().split('<eos>')
            session = [sent.strip() for sent in session]
            for sent in session:
                if sent != '<selection>' and ":" in sent:
                    sent = sent.split(':')[1].strip()
                # print('\t', sent)
                sents.append(sent)
    return sents

def _preprocess(sent):
    pass

def _round1(sent):
    """Special sentence pattern, using simple string matching."""
    if sent == '<selection>':
        return 'select'
    if sent in ['yes .', 'yes', 'deal', 'deal .']:
        return 'ok'
    return None

def _convert(sent):
    """Return the action label of sent."""
    action = _round1(sent)
    if action is not None:
        return action


def convert_sents(sents):
    """Convert the action label of sents."""
    return list(map(_convert, sents))


sents = read_sents()
actions = convert_sents(sents)

def result_stat(sents, actions):
    unsolved_sents = []
    for sent, action in zip(sents, actions):
        if action is None:
            unsolved_sents.append(sent)
    unsolved_sents = [random.choice(unsolved_sents) for _ in range(100)]
    with open(os.path.join(TMP_PATH, 'unsolved_sents.txt'), 'w+') as f:
        for usent in unsolved_sents:
            f.write(usent + '\n')
        f.close()

result_stat(sents, actions)