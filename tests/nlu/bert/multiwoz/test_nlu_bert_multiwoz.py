from tatk.nlu.bert.multiwoz.nlu import BERTNLU
from tests.nlu.test_nlu import BaseTestNLU


class TestBERTNLU(BaseTestNLU):
    def test_usr(self):
        model_file = self.model_urls['bert_multiwoz_usr']
        self.nlu = BERTNLU('usr', model_file)
        super()._test_predict(self.usr_utterances)

    def test_sys(self):
        model_file = self.model_urls['bert_multiwoz_sys']
        self.nlu = BERTNLU('sys', model_file)
        super()._test_predict(self.sys_utterances)

    def test_all(self):
        model_file = self.model_urls['bert_multiwoz_all']
        self.nlu = BERTNLU('all', model_file)
        super()._test_predict(self.all_utterances)
