

def extract_entities_metamap(txt):
    entities = None
    """
    This function uses a third party tool i.e. MetaMap to extract medical events
    The entities should adopt the following format:
    [
        {"begin": "9", "end": "17", "CUI": "C0809949", "orgi_term": "admitted", "pref_term": "Admission activity", "sem_type": "hlca"}, 
        {"begin": "22", "end": "41", "CUI": "C5206332", "orgi_term": "high calcium levels", "pref_term": "High Calcium Level", "sem_type": "fndg"},
        ...
    ]
    You should install MetaMap locally and then implement this function by yourself
    """
    # TODO: Apply MetaMap to extract events' information
    return entities
