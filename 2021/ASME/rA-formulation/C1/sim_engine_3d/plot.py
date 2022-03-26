#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from mechanisms.four_link import four_link
from mechanisms.slider_crank import slider_crank
from mechanisms.double_pendulum import double_pendulum
from mechanisms.single_pendulum import single_pendulum

def run_model(args):
    form, model_fn, num_bodies = args
    pretty_name = ' '.join([word.capitalize() for word in model_fn.__name__.split('_')])

    pos_data, vel_data, acc_data, _, grid = model_fn(['--form', form, '--mode', 'kinematics', '--tol', '1e-6',
                                             '--step_size', '0.001', '-t', '5'])


tasks = []
for model_fn in [slider_crank]:
    num_bodies = 2 if model_fn.__name__ == 'double_pendulum' else 3
    for form in ['rp']:
        tasks.append((form, model_fn, num_bodies))

for task in tasks:
    run_model(task)
