"""
SVMNLU build a classifier for each semantic tuple (intent-slot-value) based on n-gram features. It's first proposed by Mairesse et al. (2009). We adapt the implementation from pydial.
For more information, please refer to ``tatk/nlu/svm/camrest/README.md``
Trained models can be download on:
- https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_camrest_all.zip
- https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_camrest_sys.zip
- https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_camrest_usr.zip
"""
import configparser
import os
import zipfile

from tatk.util.file_util import cached_path
from tatk.nlu.svm import Classifier
from tatk.nlu import NLU


class SVMNLU(NLU):
    def __init__(self, mode, model_file):
        """
        SVM NLU initialization.

        Args:
            mode (str): can be either `'usr'`, `'sys'` or `'all'`, representing which side of data the model was trained on.
            model_file (str): trained model path or url, should be coherent with mode.

        Example usage: nlu = SVMNLU(mode='usr', model_file='https://tatk-data.s3-ap-northeast-1.amazonaws.com/svm_camrest_usr.zip')
        """
        assert mode == 'usr' or mode == 'sys' or mode == 'all'
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/camrest_{}.cfg'.format(mode))
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.c = Classifier.classifier(self.config)
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  self.config.get("train", "output"))
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_path):
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            print('Load from model_file param')
            archive_file = cached_path(model_file)
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            archive.close()
        self.c.load(model_path)

    def predict(self, utterance):
        sentinfo = {
            "turn-id": 0,
            "asr-hyps": [
                    {
                        "asr-hyp": utterance,
                        "score": 0
                    }
                ]
            }
        slu_hyps = self.c.decode_sent(sentinfo, self.config.get("decode", "output"))
        act_list = []
        for hyp in slu_hyps:
            if hyp['slu-hyp']:
                act_list = hyp['slu-hyp']
                break
        dialog_act = {}
        for act in act_list:
            intent = act['act']
            if intent == 'request':
                slot = act['slots'][0][1]
                dialog_act.setdefault(intent,[])
                dialog_act[intent].append([slot, '?'])
            else:
                dialog_act.setdefault(intent, [])
                dialog_act[intent].append(act['slots'][0])
        return dialog_act


if __name__ == "__main__":
    nlu = SVMNLU(mode='usr',
                 model_file='model/svm_camrest_usr.zip')
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
