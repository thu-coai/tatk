import argparse
import pickle
import os
import json
from convlab.modules.nlu.multiwoz.bert.dataloader import Dataloader
from convlab.modules.nlu.multiwoz.bert.model import BertNLU
import torch
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np

torch.manual_seed(9102)
random.seed(9102)
np.random.seed(9102)


parser = argparse.ArgumentParser(description="Test a model.")
parser.add_argument('--config_path',
                    default='configs/multiwoz.json',
                    help='path to config file')


if __name__ == '__main__':
    args = parser.parse_args()
    config = json.load(open(args.config_path))
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    log_dir = config['log_dir']
    DEVICE = config['DEVICE']

    data = pickle.load(open(os.path.join(data_dir,'data.pkl'),'rb'))
    intent_vocab = pickle.load(open(os.path.join(data_dir,'intent_vocab.pkl'),'rb'))
    tag_vocab = pickle.load(open(os.path.join(data_dir,'tag_vocab.pkl'),'rb'))
    for key in data:
        print('{} set size: {}'.format(key,len(data[key])))
    print('intent num:', len(intent_vocab))
    print('tag num:', len(tag_vocab))

    dataloader = Dataloader(data, intent_vocab, tag_vocab)

    best_model_path = best_model_path = os.path.join(output_dir, 'bestcheckpoint.tar')
    checkpoint = torch.load(best_model_path)

    model = BertNLU(config['model'], dataloader.intent_dim, dataloader.tag_dim, DEVICE=DEVICE,
                    intent_weight=dataloader.intent_weight)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    batch_size = 1 # config['batch_size']

    batch_num = len(dataloader.data['test']) // batch_size + 1
    for i in range(batch_num):
        batch_data = dataloader.data['test'][i * batch_size:(i + 1) * batch_size]
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor = dataloader._pad_batch(batch_data)
        intent_logits, tag_logits = model.forward(word_seq_tensor, word_mask_tensor)
        ori_word_seq = batch_data[0][0]
        new2ori = batch_data[0][-4]
        intents = dataloader.recover_intent(intent_logits,tag_logits,tag_mask_tensor,ori_word_seq,new2ori)
        print(batch_data)
        print(intent_logits)
        print(intents)
