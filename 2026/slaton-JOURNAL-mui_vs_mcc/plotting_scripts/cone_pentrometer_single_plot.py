import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from penetrometer_folderParser import collect_experiments, extract_force_data
from penetrometer_plotting import plot_depth_vs_cone_pressure

time_step_dict = {
    0.001: 2e-5,
    0.0005: 1e-5,
    0.002: 5e-5
}


if __name__ == "__main__":
    root_directory = "./DEMO_OUTPUT/FSI_ConePenetrometer_GRC1"

    df_experiments = collect_experiments(root_directory)

    query_dict = {
        "penetrationDepth": 0.18,
        "youngsModulus": 1e6,
        "density": 1600,
        # "mu_s": 0.5727,
        # "mu_2": 0.5727,
        "cohesion": 0,
        "boundaryType": 'adami',
        "viscosityType": 'artificial_unilateral',
        "kernelType": 'wendland',
        "proximitySteps": 1,
        "initialSpacing": 0.001,
        "artificialViscosity": 0.5,
        "completed": True,
        "d0": 1.3,
    }
    force_df = extract_force_data(
        df_experiments, root_directory, query_dict)

    # Plot for
    plot_depth_vs_cone_pressure(force_df, density=query_dict["density"])