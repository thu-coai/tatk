from copy import deepcopy

def delexicalize_da(meta, requestable):
    meta = deepcopy(meta)
    for k, v in meta.items():
        domain, intent = k.split('-')
        if intent.lower() in requestable:
            for pair in v:
                pair.insert(1, '?')
        else:
            counter = {}
            for pair in v:
                if pair[0] == 'none':
                    pair.insert(1, 'none')
                else:
                    if pair[0] in counter:
                        counter[pair[0]] += 1
                    else:
                        counter[pair[0]] = 1
                    pair.insert(1, str(counter[pair[0]]))
    return meta

def flat_da(meta):
    meta = deepcopy(meta)
    flaten = []
    for k, v in meta.items():
        for pair in v:
            flaten.append('-'.join((k, pair[0], str(pair[1]))))
    return flaten

def deflat_da(meta):
    meta = deepcopy(meta)
    dialog_act = {}
    for da in meta:
        d, i, s, v = da.split('-')
        k = '-'.join((d, i))
        if k not in dialog_act:
            dialog_act[k] = []
        dialog_act[k].append([s, v])
    return dialog_act

def lexicalize_da(meta, entities, state, requestable):
    meta = deepcopy(meta)
    
    for k, v in meta.items():
        domain, intent = k.split('-')
        if intent.lower() in requestable:
            for pair in v:
                v[1] = '?'
        elif intent.lower() in ['nooffer', 'nobook']:
            for pair in v:
                if v[0] in state[domain]['semi']:
                    v[1] = state[domain]['semi'][v[0]]
                else:
                    v[1] = 'none'
        else:
            for pair in v:
                if v[1] == 'none':
                    continue
                elif v[0].lower() == 'choice':
                    v[1] = str(len(entities[domain]))
                else:
                    n = int(v[1]) - 1
                    if len(entities[domain]) > n and v[0] in entities[domain][n]:
                        v[1] = entities[domain][n][v[0]]
                    else:
                        v[1] = 'none'    
