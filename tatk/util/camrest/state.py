def default_state(da_voc, informable):
    state = {}
    for da in da_voc:
        i, s, v = da.lower().split('-')
        if i in informable and s not in ['none', 'choice', 'ref']:
            state[s] = ""
    return state