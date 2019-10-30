from abc import ABCMeta, abstractmethod
from pprint import pprint
from tatk.util.dataloader.dataset_dataloader import DatasetDataloader, MultiWOZDataloader


class ModuleDataloader(metaclass=ABCMeta):
    def __init__(self, dataset_dataloader:DatasetDataloader):
        self.dataset_dataloader = dataset_dataloader

    @abstractmethod
    def load_data(self, *args, **kwargs):
        return self.dataset_dataloader.load_data(*args, **kwargs)


class SingleTurnNLUDataloader(ModuleDataloader):
    def load_data(self, *args, **kwargs):
        kwargs['utterance'] = True
        kwargs['dialog_act'] = True
        return self.dataset_dataloader.load_data(*args, **kwargs)


if __name__ == '__main__':
    d = SingleTurnNLUDataloader(dataset_dataloader=MultiWOZDataloader())
    data = d.load_data(data_key='val', role='user')
    pprint(data['val']['utterance'][:5])
    pprint(data['val']['dialog_act'][:5])
