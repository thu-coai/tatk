import os
import pickle
import torch
import torch.utils.data as data
from tatk.util.multiwoz.state import default_state
from tatk.policy.vector.dataset import ActDataset
from tatk.util.dataloader.dataset_dataloader import MultiWOZDataloader
from tatk.util.dataloader.module_dataloader import ActPolicyDataloader

class ActMLEPolicyDataLoader():
    
    def __init__(self):
        self.vector = None
        
    def _build_data(self, root_dir, processed_dir):        
        self.data = {}
        data_loader = ActPolicyDataloader(dataset_dataloader=MultiWOZDataloader())
        for part in ['train', 'val', 'test']:
            self.data[part] = []
            raw_data = data_loader.load_data(data_key=part, role='system')[part]
            
            for turn in raw_data:
                state = default_state()
                state['belief_state'] = turn['belief_state']
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
