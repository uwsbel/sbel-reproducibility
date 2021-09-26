import sys
import pathlib as pl
src_folder = pl.Path('/src/')
sys.path.append(str(src_folder))

import matplotlib.pyplot as plt
import argparse as arg
import logging

import cProfile
import pstats
import io
from pstats import SortKey

from rA_sim_engine_3d import rASimEngine3D
from rp_sim_engine_3d import rpSimEngine3D
from reps_sim_engine_3d import repsSimEngine3D

profiler = cProfile.Profile()

def standard_setup(parser, model_files, args=None):
    """
    Initializes common command-line options for multi-body simulation
    """

    # Setup command-line options
    parser.add_argument('--form', choices=['rp', 'rA', 'reps'], default='rp')
    parser.add_argument('--mode', choices=['kin', 'dyn', 'kinematics', 'dynamics'], default='kinematics')
    parser.add_argument('--tol', type=float)
    parser.add_argument('-t', '--end_time', type=float, default=3, dest='t_end')
    parser.add_argument('--step_size', type=float, default=1e-3, dest='h')

    parser.add_argument('-l', '--log', choices=['debug', 'info', 'warning', 'error'], default='info')
    parser.add_argument('-o', '--output')

    parser.add_argument('--plot', default=False, action='store_true')
    parser.add_argument('--no-plot', dest='plot', action='store_false')

    parser.add_argument('--save_data', default=False, action='store_true')
    parser.add_argument('--read_data', default=False, action='store_true')

    out_args = parser.parse_args() if args is None else parser.parse_args(args)

    # Determine which formulation to use
    if out_args.form == 'rp':
        sys = rpSimEngine3D(model_files)
    elif out_args.form == 'rA':
        sys = rASimEngine3D(model_files)
    elif out_args.form == 'reps':
        sys = repsSimEngine3D(model_files)
    else:
        raise ValueError('Unmapped formulation {} encountered'.format(out_args.form))

    logging.basicConfig(filename=out_args.output, level=getattr(logging, out_args.log.upper()), format='%(message)s')

    return sys, out_args

def print_profiling(profiler):
    """
    Prints out profiling information, based on suggestions here https://docs.python.org/3/library/profile.html#module-cProfile
    """
    s = io.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    #ps.strip_dirs()
    ps.print_stats()
    print(s.getvalue())