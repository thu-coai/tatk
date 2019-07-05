import os
import json
import pickle
import zipfile
import torch
import torch.utils.data as data
from tatk.util.multiwoz.state import default_state
from tatk.policy.vector.vector_multiwoz import MultiWozVector

class PolicyDataLoaderMultiWoz():
    
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
            archive = zipfile.ZipFile(os.path.join(root_dir, 'data/multiwoz/{}.json.zip'.format(part)), 'r')
            with archive.open('{}.json'.format(part), 'r') as f:
                raw_data[part] = json.load(f)
        
        self.data = {}
        for part in ['train', 'val', 'test']:
            self.data[part] = []
            
            for key in raw_data[part]:
                sess = raw_data[part][key]['log']
                state = default_state()
                action = {}
                for i, turn in enumerate(sess):
                    if i % 2 == 0:
                        state['user_action'] = turn['dialog_act']
                        if i + 2 == len(sess):
                            state['terminal'] = True
                    else:
                        state['belief_state'] = turn['metadata']
                        action = turn['dialog_act']
                        self.data[part].append([self.vector.state_vectorize(state),
                                 self.vector.action_vectorize(action)])
                        state['system_action'] = turn['dialog_act']
        
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
        dataset = Dataset(s, a)
        dataloader = data.DataLoader(dataset, batchsz, True)
        print('Finish creating {} dataset'.format(part))
        return dataloader

class Dataset(data.Dataset):
    def __init__(self, s_s, a_s):
        self.s_s = s_s
        self.a_s = a_s
        self.num_total = len(s_s)
    
    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        return s, a
    
    def __len__(self):
        return self.num_total

