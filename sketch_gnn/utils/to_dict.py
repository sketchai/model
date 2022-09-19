import os
from typing import Dict
import collections.abc
import yaml


def yaml_to_dict(file_path: str) -> Dict:
    with open(file_path, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        return data
    return None

def parse_config(file_path: str) -> Dict:
    conf = yaml_to_dict(file_path)

    # update paths
    main_dir = conf.get('experiment').get('dir')
    conf['logger']['save_dir'] = os.path.join(main_dir, conf['logger']['save_dir'])
    conf['prep_parms_path'] = os.path.join(main_dir, conf['file_prep_parms'])
    conf['train_data']['path_data'] = os.path.join(main_dir, conf['train_data']['file_data'])    
    conf['train_data']['path_weights'] = os.path.join(main_dir, conf['train_data']['file_weights'])   
    conf['val_data']['path_data'] = os.path.join(main_dir, conf['val_data']['file_data'])
    conf['test_data']['path_data'] = os.path.join(main_dir, conf['test_data']['file_data'])
    return conf

def stack_hparams(conf):
    hparams = {}
    keys_to_save = conf['logger'].get('hparams_to_save')
    subkeys = set()
    for key in keys_to_save:
        value = conf[key]
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                name = f'{key}/{subkey}' if subkey in subkeys else subkey
                hparams[name] = subvalue
                subkeys.add(subkey)
    return hparams