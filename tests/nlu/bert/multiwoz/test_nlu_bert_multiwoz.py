import os
import subprocess
from tatk.nlu.bert.multiwoz.nlu import BERTNLU
from tests.nlu.test_nlu import BaseTestNLU


class TestBERTNLU(BaseTestNLU):
    def test_usr(self):
        assert 0 == subprocess.call("cd tatk/nlu/bert/multiwoz && python preprocess.py {mode}".format(mode='usr'),
                                    shell=True)
        model_file = self.model_urls['bert_multiwoz_usr']
        project_dir = '.'
        config_dir = os.path.join(project_dir, 'tatk/nlu/bert/multiwoz/configs')
        config_file = os.path.join(config_dir, 'multiwoz_usr.json')
        self.nlu = BERTNLU(config_file, model_file)
        super()._test_predict(self.usr_utterances)

    def test_sys(self):
        assert 0 == subprocess.call("cd tatk/nlu/bert/multiwoz && python preprocess.py {mode}".format(mode='sys'),
                                    shell=True)
        model_file = self.model_urls['bert_multiwoz_sys']
        project_dir = '.'
        config_dir = os.path.join(project_dir, 'tatk/nlu/bert/multiwoz/configs')
        config_file = os.path.join(config_dir, 'multiwoz_sys.json')
        self.nlu = BERTNLU(config_file, model_file)
        super()._test_predict(self.sys_utterances)

    def test_all(self):
        assert 0 == subprocess.call("cd tatk/nlu/bert/multiwoz && python preprocess.py {mode}".format(mode='all'),
                                    shell=True)
        model_file = self.model_urls['bert_multiwoz_all']
        project_dir = '.'
        config_dir = os.path.join(project_dir, 'tatk/nlu/bert/multiwoz/configs')
        config_file = os.path.join(config_dir, 'multiwoz_all.json')
        self.nlu = BERTNLU(config_file, model_file)
        super()._test_predict(self.all_utterances)
