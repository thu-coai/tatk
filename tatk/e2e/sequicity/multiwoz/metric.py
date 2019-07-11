import json
import os

from tatk.e2e.sequicity.metric import GenericEvaluator, report


class MultiWozEvaluator(GenericEvaluator):
    def __init__(self, result_path):
        super().__init__(result_path)
        self.entities = []
        self.entity_dict = {}

    def run_metrics(self):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        with open(os.path.join(data_dir, 'test.json')) as f:
            raw_data = json.loads(f.read().lower())
        with open(os.path.join(data_dir, 'entities.json')) as f:
            raw_entities = json.loads(f.read().lower())
        self.get_entities(raw_entities)
        data = self.read_result_data()
        for i, row in enumerate(data):
            data[i]['response'] = self.clean(data[i]['response'])
            data[i]['generated_response'] = self.clean(data[i]['generated_response'])
        bleu_score = self.bleu_metric(data,'bleu')
        success_f1 = self.success_f1_metric(data, 'success')
        match = self.match_metric(data, 'match', raw_data=raw_data)
        self._print_dict(self.metric_dict)
        return -success_f1[0]

    def get_entities(self, entity_data):
        for k in entity_data:
            k_attr = k.split('_')[1][:-1]
            self.entities.extend(entity_data[k])
            for item in entity_data[k]:
                self.entity_dict[item] = k_attr

    def _extract_constraint(self, z):
        z = z.split()
        if 'EOS_Z1' not in z:
            s = set(z)
        else:
            idx = z.index('EOS_Z1')
            s = set(z[:idx])
        if 'moderately' in s:
            s.discard('moderately')
            s.add('moderate')
        #print(self.entities)
        #return s
        return s.intersection(self.entities)
        #return set(z).difference(['name', 'address', 'postcode', 'phone', 'area', 'pricerange'])

    def _extract_request(self, z):
        z = z.split()
        return set(z).intersection(['address', 'postcode', 'phone', 'area', 'pricerange','food'])

    @report
    def match_metric(self, data, sub='match',raw_data=None):
        dials = self.pack_dial(data)
        match,total = 0,1e-8
        # find out the last placeholder and see whether that is correct
        # if no such placeholder, see the final turn, because it can be a yes/no question or scheduling dialogue
        for dial_id in dials:
            truth_req, gen_req = [], []
            dial = dials[dial_id]
            gen_bspan, truth_cons, gen_cons = None, None, set()
            truth_turn_num = -1
            truth_response_req = []
            for turn_num,turn in enumerate(dial):
                if 'SLOT' in turn['generated_response']:
                    gen_bspan = turn['generated_bspan']
                    gen_cons = self._extract_constraint(gen_bspan)
                if 'SLOT' in turn['response']:
                    truth_cons = self._extract_constraint(turn['bspan'])
                gen_response_token = turn['generated_response'].split()
                response_token = turn['response'].split()
                for idx, w in enumerate(gen_response_token):
                    if w.endswith('SLOT') and w != 'SLOT':
                        gen_req.append(w.split('_')[0])
                    if w == 'SLOT' and idx != 0:
                        gen_req.append(gen_response_token[idx - 1])
                for idx, w in enumerate(response_token):
                    if w.endswith('SLOT') and w != 'SLOT':
                        truth_response_req.append(w.split('_')[0])
            if not gen_cons:
                gen_bspan = dial[-1]['generated_bspan']
                gen_cons = self._extract_constraint(gen_bspan)
            if truth_cons:
                if gen_cons == truth_cons:
                    match += 1
                else:
                    pass
#                    print(gen_cons, truth_cons)
                total += 1

        return match / total

    @report
    def success_f1_metric(self, data, sub='successf1'):
        dials = self.pack_dial(data)
        tp,fp,fn = 0,0,0
        for dial_id in dials:
            truth_req, gen_req = set(),set()
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                gen_response_token = turn['generated_response'].split()
                response_token = turn['response'].split()
                for idx, w in enumerate(gen_response_token):
                    if w.endswith('SLOT') and w != 'SLOT':
                        gen_req.add(w.split('_')[0])
                for idx, w in enumerate(response_token):
                    if w.endswith('SLOT') and w != 'SLOT':
                        truth_req.add(w.split('_')[0])

            gen_req.discard('name')
            truth_req.discard('name')
            for req in gen_req:
                if req in truth_req:
                    tp += 1
                else:
                    fp += 1
            for req in truth_req:
                if req not in gen_req:
                    fn += 1
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1, precision, recall
