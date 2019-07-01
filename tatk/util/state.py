def default_state(da_voc, informable):
    state = {}
    for da in da_voc:
        d, i, s, v = da.lower().split('-')
        if d in ['general', 'booking']:
            continue
        if d not in state:
            state[d] = {}
        if i.capitalize() in informable and s not in ['none', 'choice', 'ref']:
            state[d][s] = ""
    return state