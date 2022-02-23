import json
import argparse
import copy
import pdb
def replace_list_with_string_in_a_dict(dictionary):
    # dict_keys = []
    for key in dictionary.keys():
        if isinstance(dictionary[key], list):
            dictionary[key] = str(dictionary[key])
        if isinstance(dictionary[key], dict):
            dictionary[key] = replace_list_with_string_in_a_dict(dictionary[key])
    return dictionary

def restore_string_to_list_in_a_dict(dictionary):
    for key in dictionary.keys():
        try:
            evaluated = eval(dictionary[key])
            if isinstance(evaluated, list):
                dictionary[key] = evaluated
        except:
            pass
        if isinstance(dictionary[key], dict):
            dictionary[key] = restore_string_to_list_in_a_dict(dictionary[key])
    return dictionary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.json', 
                        help='JSON file for configuration')
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    # pdb.set_trace()
    config_string = replace_list_with_string_in_a_dict(copy.deepcopy(config))
    print('The configuration is:')
    print(json.dumps(config_string, indent=4))

    config_restore = restore_string_to_list_in_a_dict(config_string)
    print(config_restore == config)
    pdb.set_trace()
