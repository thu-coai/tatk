def is_slot_da(da):
    tag_da = {'Inform', 'Select', 'Recommend', 'NoOffer', 'NoBook', 'OfferBook', 'OfferBooked', 'Book'}
    not_tag_slot = {'Internet', 'Parking', 'none'}
    if da[0].split('-')[1] in tag_da and da[1] not in not_tag_slot:
        return True
    return False
