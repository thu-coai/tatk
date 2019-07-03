import os
import json
import zipfile
from tatk.policy.multiwoz.vector_multiwoz import MultiWozVector

class PolicyDataLoaderMultiWoz():
    
    def __init__(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        voc_file = os.path.join(root_dir, 'data/multiwoz/sys_da_voc.txt')
        voc_opp_file = os.path.join(root_dir, 'data/multiwoz/usr_da_voc.txt')
        self.vector = MultiWozVector(voc_file, voc_opp_file)
        
        processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data')
        if os.path.exists(processed_dir):
            print('Load processed data file')
        else:
            print('Start preprocessing the dataset')
            self._build_data(root_dir)
        
    def _build_data(self, root_dir):
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
                state = {'last_action':{}, 'action':{}, 'belief_state':{}, 'terminal':False}
                action = {}
                for turn in sess:
                    if i % 2 == 0:
                        state['action'] = raw_data['']
