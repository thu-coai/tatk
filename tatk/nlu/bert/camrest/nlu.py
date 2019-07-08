import os
import zipfile
import json
import pickle
import torch

from tatk.util.file_util import cached_path
from tatk.nlu.nlu import NLU
from tatk.nlu.bert.dataloader import Dataloader
from tatk.nlu.bert.model import BertNLU
from tatk.nlu.bert.camrest.postprocess import recover_intent


class BERTNLU(NLU):
    def __init__(self, config_file, model_file):
        config = json.load(open(config_file))
        DEVICE = config['DEVICE']
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(root_dir, config['data_dir'])
        output_dir = os.path.join(root_dir, config['output_dir'])

        data = pickle.load(open(os.path.join(data_dir, 'data.pkl'), 'rb'))
        intent_vocab = pickle.load(open(os.path.join(data_dir, 'intent_vocab.pkl'), 'rb'))
        tag_vocab = pickle.load(open(os.path.join(data_dir, 'tag_vocab.pkl'), 'rb'))

        dataloader = Dataloader(data, intent_vocab, tag_vocab, config['model']["pre-trained"])

        best_model_path = os.path.join(output_dir, 'bestcheckpoint.tar')
        if not os.path.exists(best_model_path):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print('Load from model_file param')
            archive_file = cached_path(os.path.join(root_dir, model_file))
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(root_dir)
            archive.close()
        print('Load from', best_model_path)
        checkpoint = torch.load(best_model_path)
        print('train step', checkpoint['step'])

        model = BertNLU(config['model'], dataloader.intent_dim, dataloader.tag_dim,
                        DEVICE=DEVICE,
                        intent_weight=dataloader.intent_weight)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        model.to(DEVICE)
        model.eval()

        self.model = model
        self.dataloader = dataloader
        print("BERTNLU loaded")

    def predict(self, utterance):
        ori_word_seq = utterance.split()
        ori_tag_seq = ['O'] * len(ori_word_seq)
        intents = []
        da = {}

        word_seq, tag_seq, new2ori = self.dataloader.bert_tokenize(ori_word_seq, ori_tag_seq)
        batch_data = [[ori_word_seq, ori_tag_seq, intents, da, new2ori, word_seq, self.dataloader.seq_tag2id(tag_seq),
                       self.dataloader.seq_intent2id(intents)]]
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor = self.dataloader._pad_batch(
            batch_data)
        intent_logits, tag_logits = self.model.forward(word_seq_tensor, word_mask_tensor)
        intent = recover_intent(self.dataloader, intent_logits[0], tag_logits[0], tag_mask_tensor[0],
                                batch_data[0][0], batch_data[0][4])
        dialog_act = {}
        for act, slot, value in intent:
            dialog_act.setdefault(act, [])
            dialog_act[act].append([slot, value])
        return dialog_act


if __name__ == '__main__':
    nlu = BERTNLU(config_file='configs/camrest_usr.json', model_file='output/usr/bert_camrest_usr.zip')
    test_utterances = [
        "What type of accommodations are they. No , i just need their address . Can you tell me if the hotel has internet available ?",
        "What type of accommodations are they.",
        "No , i just need their address .",
        "Can you tell me if the hotel has internet available ?"
        "you're welcome! enjoy your visit! goodbye.",
        "yes. it should be moderately priced.",
        "i want to book a table for 6 at 18:45 on thursday",
        "i will be departing out of stevenage.",
        "What is the Name of attraction ?",
        "Can I get the name of restaurant?",
        "Can I get the address and phone number of the restaurant?",
        "do you have a specific area you want to stay in?"
    ]
    for utt in test_utterances:
        print(utt)
        print(nlu.predict(utt))
