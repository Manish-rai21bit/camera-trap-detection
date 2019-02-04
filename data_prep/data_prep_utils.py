"""module for helper functions needed throughout """

import json

def get_label_map_from_json(json_file):
    with open(json_file, 'r') as f:
        label_map = json.load(f)
        
    return label_map