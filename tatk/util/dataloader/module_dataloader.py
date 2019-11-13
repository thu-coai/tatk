from abc import ABC, abstractmethod
from pprint import pprint
from tatk.util.dataloader.dataset_dataloader import DatasetDataloader, MultiWOZDataloader


class ModuleDataloader(ABC):
    def __init__(self, dataset_dataloader:DatasetDataloader):
        self.dataset_dataloader = dataset_dataloader

    @abstractmethod
    def load_data(self, *args, **kwargs):
        return self.dataset_dataloader.load_data(*args, **kwargs)


class SingleTurnNLUDataloader(ModuleDataloader):
    def load_data(self, *args, **kwargs):
        kwargs.setdefault('utterance', True)
        kwargs.setdefault('dialog_act', True)
        return self.dataset_dataloader.load_data(*args, **kwargs)


class MultiTurnNLUDataloader(ModuleDataloader):
    def load_data(self, *args, **kwargs):
        kwargs.setdefault('utterance', True)
        kwargs.setdefault('dialog_act', True)
        kwargs.setdefault('context', True)
        kwargs.setdefault('context_window_size', 3)
        return self.dataset_dataloader.load_data(*args, **kwargs)


if __name__ == '__main__':
    d = SingleTurnNLUDataloader(dataset_dataloader=MultiWOZDataloader())
    data = d.load_data(data_key='val', role='user')
    pprint(data['val']['utterance'][:5])
    pprint(data['val']['dialog_act'][:5])
