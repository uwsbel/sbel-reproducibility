import matplotlib.pyplot as plt

import os
import argparse
import pandas as pd
import numpy as np

def check_i(target_i, base_dir="data/traj"):
    failed_count = 0
    good_values = []

    for j in range(200):  # j = 0..99
        filename = os.path.join(base_dir, f"l2_{target_i}_{j}.csv")

        if not os.path.isfile(filename):
            print(f"[WARN] File not found: {filename}")
            continue

        # Assuming no header; if you have a header, add header=0 and adjust
        df = pd.read_csv(filename, header=None)
        value = df.iloc[-1, 0]  # last row, first column

        if value > 39.5:
            failed_count += 1
            print(f"[FAILED] i={target_i}, j={j}, value={value:.4f} (file: {filename})")
        else:
            good_values.append(value)

    print(f"\nSummary for i={target_i}")
    print(f"  Failed cases: {failed_count}")

    if len(good_values) > 0:
        mean_val = float(np.mean(good_values))
        print(f"  Mean of non-failed data[0,-1]: {mean_val:.6f}")
        print(f"  Number of non-failed samples: {len(good_values)}")
    else:
        print("  No non-failed samples to compute mean.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("i", type=int, help="Target i (1-4)")
    args = parser.parse_args()

    check_i(args.i)
