import pickle
import os

import numpy as np
import matplotlib.pyplot as plt

from SimEngineMBD.example_models.single_pendulum import time_single_pendulum
from SimEngineMBD.example_models.double_pendulum import time_double_pendulum
from SimEngineMBD.example_models.four_link import time_four_link
from SimEngineMBD.example_models.slider_crank import time_slider_crank

# Testing
# forms = ['rA', 'reps']
# model_fns = [time_single_pendulum]
# modes = ["kinematics", "dynamics"]
# num_runs = 2
# end_time = 1

# Production
forms = ['rp', 'rA', 'reps']
model_fns = [time_single_pendulum, time_double_pendulum, time_four_link, time_slider_crank]
modes = ["kinematics", "dynamics"]
num_runs = 5
end_time = 3

time_dict = {}

for form in forms:
    for model_fn in model_fns:
        # time_some_model_name -> Some_Model_Name
        model = '_'.join([word.capitalize() for word in model_fn.__name__[5:].split('_')])

        for mode in modes:
            if mode == "kinematics" and model_fn == time_double_pendulum:
                continue

            args = ['--form', form, '--mode', mode, '--end_time', str(end_time)]

            time_dict[(form, model, mode)] = [model_fn(args) for _ in range(0, num_runs)]
            
for key, times in time_dict.items():
    print('{} {} {}'.format(form, model, mode))

    for time in times:
        print(time)
