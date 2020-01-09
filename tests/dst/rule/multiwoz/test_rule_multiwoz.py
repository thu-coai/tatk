from tests.dst.test_state_tracker import BaseTestMultiwozTracker
from tatk.dst.rule.multiwoz.dst import RuleDST


class TestRuleDST(BaseTestMultiwozTracker):
    def test_update(self):
        self.tracker = RuleDST()
        self._test_update(self.usr_acts)
