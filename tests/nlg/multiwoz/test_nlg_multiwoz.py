from tests.nlg.test_nlg import BaseTestNLG
from tatk.nlg.multiwoz.template_nlg.multiwoz_template_nlg import MultiwozTemplateNLG


class TestMultiwozNLG(BaseTestNLG):
    def test_nlg(self):
        self._test_nlg(MultiwozTemplateNLG)
