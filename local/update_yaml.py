import yaml
import copy
import argparse
from utils import isdigit
from yamlinclude import YamlIncludeConstructor
YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader)

def update_dict_value(d, keys, value):

    if isdigit(value):
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
        
    if len(keys) == 1:
        d[keys[0]] = value
    else:
        if keys[0] not in d:
            d[keys[0]] = {}
        update_dict_value(d[keys[0]], keys[1:], value)
    return d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exist_config', required=True,
        help='exist_config yaml config file'
    )
    parser.add_argument(
        '--dest_config', required=True,
        help='target config yaml files'
    )
    parser.add_argument(
        '--update_info', required=True,
        help="""what you want update e.g. in org config
        data_config:{sph_config: {trim_win:1}, feats_config: fbank}
        Your info should be: data_config:sph_config:1;data_config:feats_config:mfcc;
        """
    )
    args = parser.parse_args()

    org_config = yaml.load(open(args.exist_config), Loader=yaml.FullLoader) 
    new_config = copy.deepcopy(org_config)
    update_info = args.update_info.split(";")
    for items in update_info:
        if len(items) == 0:
            continue
        items = items.split(":")
        value = items[-1]
        keys = items[:-1]
        new_config = update_dict_value(new_config, keys, value)

    df = open(args.dest_config, 'w')
    yaml.dump(new_config, df)
    df.close()
    

