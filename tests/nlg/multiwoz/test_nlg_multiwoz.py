from tests.nlg.test_nlg import BaseTestNLGMultiwoz
from tatk.nlg.multiwoz.template_nlg.multiwoz_template_nlg import MultiwozTemplateNLG


class TestNLGMultiwoz(BaseTestNLGMultiwoz):
    def test_nlg(self):
        self._test_nlg(MultiwozTemplateNLG)
