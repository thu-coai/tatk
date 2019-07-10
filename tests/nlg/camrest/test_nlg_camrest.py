from tests.nlg.test_nlg import BaseTestNLGCamrest
from tatk.nlg.camrest.template_nlg.camrest_template_nlg import CamrestTemplateNLG


class TestNLGCamrest(BaseTestNLGCamrest):
    def test_nlg(self):
        self._test_nlg(CamrestTemplateNLG)
