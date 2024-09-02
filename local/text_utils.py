import json
import os

def read_mfa_obj(obj) -> list:
    if os.path.isfile(obj):
        obj = open(obj)
        try:
            obj = json.load(obj)
        except:
            raise TypeError('only support json obj')
    elif isinstance(obj, dict):
        pass
    else:
        raise TypeError('only support dict obj or files')
    
    words_info = obj['tiers']['words']['entries']
    phones_info = obj['tiers']['phones']['entries']
    return words_info, phones_info