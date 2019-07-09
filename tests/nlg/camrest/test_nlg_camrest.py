from tests.nlg.test_nlg import BaseTestNLG
from tatk.nlg.camrest.template_nlg.camrest_template_nlg import CamrestTemplateNLG


class TestCamrestNLG(BaseTestNLG):
    def test_nlg(self):
        self._test_nlg(CamrestTemplateNLG)
