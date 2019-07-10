from tests.nlg.test_nlg import BaseTestNLGMultiwoz
from tatk.nlg.template_nlg.multiwoz.nlg import TemplateNLG


class TestNLGMultiwoz(BaseTestNLGMultiwoz):
    def test_nlg(self):
        self._test_nlg(TemplateNLG)
