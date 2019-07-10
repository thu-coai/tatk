import os
from tatk.nlu.svm.camrest.nlu import SVMNLU
from tests.nlu.test_nlu import BaseTestNLUCamrest


class TestSVMNLU(BaseTestNLUCamrest):
    def test_usr(self):
        model_file = self.model_urls['svm_camrest_usr']
        project_dir = '.'
        config_dir = os.path.join(project_dir, 'tatk/nlu/svm/camrest/configs')
        config_file = os.path.join(config_dir, 'camrest_usr.cfg')
        self.nlu = SVMNLU(config_file, model_file)
        super()._test_predict(self.usr_utterances)

    def test_sys(self):
        model_file = self.model_urls['svm_camrest_sys']
        project_dir = '.'
        config_dir = os.path.join(project_dir, 'tatk/nlu/svm/camrest/configs')
        config_file = os.path.join(config_dir, 'camrest_sys.cfg')
        self.nlu = SVMNLU(config_file, model_file)
        super()._test_predict(self.sys_utterances)

    def test_all(self):
        model_file = self.model_urls['svm_camrest_all']
        project_dir = '.'
        config_dir = os.path.join(project_dir, 'tatk/nlu/svm/camrest/configs')
        config_file = os.path.join(config_dir, 'camrest_all.cfg')
        self.nlu = SVMNLU(config_file, model_file)
        super()._test_predict(self.all_utterances)
