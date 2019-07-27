from tatk.e2e.rnn_rollout.rnnrollout import RNNRolloutAgent
from tatk.e2e.rnn_rollout.models.rnn_model import RnnModel
from tatk.e2e.rnn_rollout.models.selection_model import SelectionModel
import tatk.e2e.rnn_rollout.utils as utils
from tatk.e2e.rnn_rollout.domain import get_domain
from tatk import get_root_path
import os
import zipfile
from tatk.util.file_util import cached_path

class DealornotAgent(RNNRolloutAgent):
    """The Rnn Rollout model for DealorNot dataset."""
    def __init__(self, name, args, sel_args, train=False, diverse=False, max_total_len=100,
                 model_url='https://tatk-data.s3-ap-northeast-1.amazonaws.com/rnnrollout_dealornot.zip'):
        self.config_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'configs')

        if not os.path.exists(self.config_path):
            os.mkdir(self.config_path)
        _model_path = os.path.join(self.config_path, 'models')
        if not os.path.exists(_model_path):
            os.mkdir(_model_path)
        if not os.path.exists(os.path.join(_model_path, 'rnn_model_state_dict.th')):
            # download and unzip pretrained models
            dir = cached_path(model_url, _model_path)
            print(dir)
        self.data_path = os.path.join(get_root_path(), args.data)
        domain = get_domain(args.domain)
        corpus = RnnModel.corpus_ty(domain, self.data_path, freq_cutoff=args.unk_threshold, verbose=True,
                                    sep_sel=args.sep_sel)

        model = RnnModel(corpus.word_dict, corpus.item_dict_old,
                         corpus.context_dict, corpus.count_dict, args)
        state_dict = utils.load_model(os.path.join(self.config_path, args.model_file))  # RnnModel
        model.load_state_dict(state_dict)

        sel_model = SelectionModel(corpus.word_dict, corpus.item_dict_old,
                                   corpus.context_dict, corpus.count_dict, sel_args)
        sel_state_dict = utils.load_model(os.path.join(self.config_path, sel_args.selection_model_file))
        sel_model.load_state_dict(sel_state_dict)

        super(DealornotAgent, self).__init__(model, sel_model, args, name, train, diverse, max_total_len)
        self.vis = args.visual

def get_context_generator(context_file):
    return utils.ContextGenerator(os.path.join(get_root_path(), context_file))
