import os
from tatk.nlu.svm.multiwoz.nlu import SVMNLU
from tests.nlu.test_nlu import BaseTestNLU


class TestSVMNLU(BaseTestNLU):
    def test_usr(self):
        model_file = self.model_urls['svm_multiwoz_usr']
        project_dir = '.'
        config_dir = os.path.join(project_dir, 'tatk/nlu/svm/multiwoz/configs')
        config_file = os.path.join(config_dir, 'multiwoz_usr.cfg')
        self.nlu = SVMNLU(config_file, model_file)
        super()._test_predict(self.usr_utterances)

    def test_sys(self):
        model_file = self.model_urls['svm_multiwoz_sys']
        project_dir = '.'
        config_dir = os.path.join(project_dir, 'tatk/nlu/svm/multiwoz/configs')
        config_file = os.path.join(config_dir, 'multiwoz_sys.cfg')
        self.nlu = SVMNLU(config_file, model_file)
        super()._test_predict(self.sys_utterances)

    def test_all(self):
        model_file = self.model_urls['svm_multiwoz_all']
        project_dir = '.'
        config_dir = os.path.join(project_dir, 'tatk/nlu/svm/multiwoz/configs')
        config_file = os.path.join(config_dir, 'multiwoz_all.cfg')
        self.nlu = SVMNLU(config_file, model_file)
        super()._test_predict(self.all_utterances)
