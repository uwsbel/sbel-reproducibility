"""
Functions that all system_*.py classes share and make use of
"""

import os
import json as js

def read_model_file(file_name):
    """Opens a json file and returns the top-level 'bodies' and 'constraints' as a tuple """
    
    if file_name.startswith('models'):
        # Handle paths to ../models separately
        module_dir = os.path.dirname(__file__)
        file_path = os.path.join(module_dir, '..', file_name)
    else:
        # Assume the user has given us an absolute path
        file_path = file_name

    with open(file_path, 'r') as model_file:
        model_data = js.load(model_file)

        model_bodies = model_data['bodies']
        model_constraints = model_data['constraints']

    return (model_bodies, model_constraints)