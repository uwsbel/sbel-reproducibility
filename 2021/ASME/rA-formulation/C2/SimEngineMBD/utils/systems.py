"""
Functions that all system_*.py classes share and make use of
"""

import json as js

def read_model_file(file_path):
    """Opens a json file and returns the top-level 'bodies' and 'constraints' as a tuple """

    with open(file_path, 'r') as model_file:
        model_data = js.load(model_file)

        model_bodies = model_data['bodies']
        model_constraints = model_data['constraints']

    return (model_bodies, model_constraints)