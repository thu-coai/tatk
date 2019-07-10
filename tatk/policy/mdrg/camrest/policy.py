def predict(model, prev_state, prev_active_domain, state, dic):
    start_time = time.time()
    model.beam_search = False
    input_tensor = [];
    bs_tensor = [];
    db_tensor = []

    usr = state['history'][-1][-1]

    prev_state = deepcopy(prev_state['belief_state'])
    state = deepcopy(state['belief_state'])

    mark_not_mentioned(prev_state)
    mark_not_mentioned(state)

    words = usr.split()
    usr = delexicalize.delexicalise(' '.join(words), dic)

    # parsing reference number GIVEN belief state
    usr = delexicaliseReferenceNumber(usr, state)

    # changes to numbers only here
    digitpat = re.compile('\d+')
    usr = re.sub(digitpat, '[value_count]', usr)
    # dialogue = fixDelex(dialogue_name, dialogue, data2, idx, idx_acts)

    # add database pointer
    pointer_vector, top_results, num_results = addDBPointer(state)
    # add booking pointer
    pointer_vector = addBookingPointer(state, pointer_vector)
    belief_summary = get_summary_bstate(state)

    tensor = [model.input_word2index(word) for word in normalize(usr).strip(' ').split(' ')] + [util.EOS_token]
    input_tensor.append(torch.LongTensor(tensor))
    bs_tensor.append(belief_summary)  #
    db_tensor.append(pointer_vector)  # db results and booking
    # bs_tensor.append([0.] * 94) #
    # db_tensor.append([0.] * 30) # db results and booking
    # create an empty matrix with padding tokens
    input_tensor, input_lengths = util.padSequence(input_tensor)
    bs_tensor = torch.tensor(bs_tensor, dtype=torch.float, device=device)
    db_tensor = torch.tensor(db_tensor, dtype=torch.float, device=device)

    output_words, loss_sentence = model.predict(input_tensor, input_lengths, input_tensor, input_lengths,
                                                db_tensor, bs_tensor)
    active_domain = get_active_domain(prev_active_domain, prev_state, state)
    if active_domain is not None and active_domain in num_results:
        num_results = num_results[active_domain]
    else:
        num_results = 0
    if active_domain is not None and active_domain in top_results:
        top_results = {active_domain: top_results[active_domain]}
    else:
        top_results = {}
    response = populate_template(output_words[0], top_results, num_results, state)
    return response, active_domain

def loadModel(num):
    # Load dictionaries
    with open(os.path.join(DATA_PATH, 'input_lang.index2word.json')) as f:
        input_lang_index2word = json.load(f)
    with open(os.path.join(DATA_PATH, 'input_lang.word2index.json')) as f:
        input_lang_word2index = json.load(f)
    with open(os.path.join(DATA_PATH, 'output_lang.index2word.json')) as f:
        output_lang_index2word = json.load(f)
    with open(os.path.join(DATA_PATH, 'output_lang.word2index.json')) as f:
        output_lang_word2index = json.load(f)

    # Reload existing checkpoint
    model = Model(args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index)
    model.loadModel(iter=num)

    return model


DEFAULT_CUDA_DEVICE = -1
DEFAULT_DIRECTORY = "models"
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "milu.tar.gz")


class MDRGWordPolicy(SysPolicy):
    def __init__(self,
                 archive_file=DEFAULT_ARCHIVE_FILE,
                 cuda_device=DEFAULT_CUDA_DEVICE,
                 mdoel_file=None):

        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for MDRG is specified!")
            archive_file = cached_path(model_file)

        temp_path = tempfile.mkdtemp()
        zip_ref = zipfile.ZipFile(archive_file, 'r')
        zip_ref.extractall(temp_path)
        zip_ref.close()

        self.dic = pickle.load(open(os.path.join(temp_path, 'mdrg/svdic.pkl'), 'rb'))
        # Load dictionaries
        with open(os.path.join(temp_path, 'mdrg/input_lang.index2word.json')) as f:
            input_lang_index2word = json.load(f)
        with open(os.path.join(temp_path, 'mdrg/input_lang.word2index.json')) as f:
            input_lang_word2index = json.load(f)
        with open(os.path.join(temp_path, 'mdrg/output_lang.index2word.json')) as f:
            output_lang_index2word = json.load(f)
        with open(os.path.join(temp_path, 'mdrg/output_lang.word2index.json')) as f:
            output_lang_word2index = json.load(f)
        self.response_model = Model(args, input_lang_index2word, output_lang_index2word, input_lang_word2index,
                                    output_lang_word2index)
        self.response_model.loadModel(os.path.join(temp_path, 'mdrg/mdrg'))

        shutil.rmtree(temp_path)

        self.prev_state = init_state()
        self.prev_active_domain = None


    def predict(self, state):
        try:
            response, active_domain = predict(self.response_model, self.prev_state, self.prev_active_domain, state,
                                              self.dic)
        except Exception as e:
            print('Response generation error', e)
            response = 'What did you say?'
            active_domain = None
        self.prev_state = deepcopy(state)
        self.prev_active_domain = active_domain
        return response




if __name__ == '__main__':
    dic = pickle.load(open(os.path.join(DATA_PATH, 'svdic.pkl'), 'rb'))
    state = {'user_action':{}, 'system_action':{}, 'belief_state':{}, 'terminal':False}
    state['belief_state'] = {'address': '',
                             'area': '',
                             'food': '',
                             'name': '',
                             'phone': '',
                             'pricerange': ''
                             }

    m = loadModel(15)

    s = deepcopy(state)
    s['history'] = [['null', 'I want a korean restaurant in the centre.']]
    s['belief_state']['area'] = 'centre'
    s['belief_state']['food'] = 'korean'
    predict(m, state, s, dic)
