def is_slot_da(da):
    tag_da = {'inform'}
    not_tag_slot = {'dontcare'}
    if da[0].split('-')[1] in tag_da and da[1] not in not_tag_slot:
        return True
    return False
