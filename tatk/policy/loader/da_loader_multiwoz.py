import os
import pickle
import torch
import torch.utils.data as data
from tatk.util.multiwoz.state import default_state
from tatk.policy.loader.dataset import ActDataset
from tatk.policy.vector.vector_multiwoz import MultiWozVector
from tatk.util.dataloader.module_dataloader import ActPolicyDataloader

class ActPolicyDataLoaderMultiWoz():
    
    def __init__(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
        voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
        self.vector = MultiWozVector(voc_file, voc_opp_file)
        
        processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data')
        if os.path.exists(processed_dir):
            print('Load processed data file')
            self._load_data(processed_dir)
        else:
            print('Start preprocessing the dataset')
            self._build_data(root_dir, processed_dir)
        
    def _build_data(self, root_dir, processed_dir):
        raw_data = {}
        for part in ['train', 'val', 'test']:
            raw_data[part] = ActPolicyDataloader(data_key=part, role='sys')
        
        self.data = {}
        for part in ['train', 'val', 'test']:
            self.data[part] = []
            
            for turn in raw_data[part]:
                state = default_state()
                state['user_action'] = turn['context_dialog_act'][-1]
                state['system_action'] = turn['context_dialog_act'][-2]
                state['terminal'] = turn['terminal']
                action = turn['dialog_act']
                self.data[part].append([self.vector.state_vectorize(state),
                         self.vector.action_vectorize(action)])
        
        os.makedirs(processed_dir)
        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}.pkl'.format(part)), 'wb') as f:
                pickle.dump(self.data[part], f)

    def _load_data(self, processed_dir):
        self.data = {}
        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}.pkl'.format(part)), 'rb') as f:
                self.data[part] = pickle.load(f)
                
    def create_dataset(self, part, batchsz):
        print('Start creating {} dataset'.format(part))
        s = []
        a = []
        for item in self.data[part]:
            s.append(torch.Tensor(item[0]))
            a.append(torch.Tensor(item[1]))
        s = torch.stack(s)
        a = torch.stack(a)
        dataset = ActDataset(s, a)
        dataloader = data.DataLoader(dataset, batchsz, True)
        print('Finish creating {} dataset'.format(part))
        return dataloader
