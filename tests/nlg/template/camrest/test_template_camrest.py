from tests.nlg.test_nlg import BaseTestNLGCamrest
from tatk.nlg.template.camrest.nlg import TemplateNLG


class TestNLGCamrest(BaseTestNLGCamrest):
    def test_nlg(self):
        self._test_nlg(TemplateNLG)
