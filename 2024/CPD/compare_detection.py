import time
# from metrics.measures import *
from metrics.detection_comparison import *

import argparse
import sys
import os


def main(args):
    # save the configuration to a bash script
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    run_file = open(os.path.join(args.output_path, "run.sh"), "w")
    run_file.write("#!/usr/bin/env bash\n")
    command = "python3 "
    for a in sys.argv:
        command += a + " "
    run_file.write(command + "\n")
    run_file.close()

    # create the class to perform object detection comparison
    comparitor = DetectionComparitor(output_path=args.output_path, write=args.save)

    # set the datasets for comparison
    comparitor.SetDatasets(
        args.a_path,
        args.b_path,
        args.a_preds,
        args.b_preds,
        args.a_name,
        args.b_name,
        threads=1,
        max_samples=args.n,
        shuffle_a=args.shuffle_a,
        shuffle_b=args.shuffle_b,
    )

    #run comparison
    comparitor.PatchComparison(
        args=args
    )

def parseargs():
    # === DEFAULTS === #
    test_samples = -1
    parser = argparse.ArgumentParser(description="Object detection CPD.")

    # patch comparison parameters
    parser.add_argument(
        "--patch_size",
        "-ps",
        type=int,
        nargs="+",
        default=[64, 64],
        help="size of the patch for comparison centered around the object",
    )
    parser.add_argument(
        "--mask_factor",
        "-mf",
        type=float,
        nargs="+",
        default=[1.5, 1.5],
        help="size of the patch for comparison centered around the object",
    )
    parser.add_argument(
        "--patch_threshold",
        "-pt",
        type=float,
        default=0.9,
        help="threshold of patch similarity to be considered as part of the same batch",
    )
    parser.add_argument(
        "--min-batch-size",
        "-bs",
        type=int,
        default=1,
        help="minimum number of samples to consider a batch",
    )
    parser.add_argument("--skvd", action="store_true", help="calculate skvd metric")

    # information about the datasets being compared
    parser.add_argument(
        "--a_name",
        type=str,
        default="Dataset A",
        help="Name of first dataset in the comparison",
    )
    parser.add_argument(
        "--a_path",
        type=str,
        default="output_a",
        help="Path to first dataset's metrics files",
    )
    parser.add_argument(
        "--a_preds",
        type=str,
        default="output_a",
        help="Path to predictions on dataset A",
    )

    parser.add_argument(
        "--b_name",
        type=str,
        default="Dataset B",
        help="Name of second dataset in the comparison",
    )
    parser.add_argument(
        "--b_path",
        type=str,
        default="output_b",
        help="Path to second dataset's metrics files",
    )
    parser.add_argument(
        "--b_preds",
        type=str,
        default="output_b",
        help="Path to predictions on dataset B",
    )

    # miscellaneous parameters
    parser.add_argument(
        "--display", action="store_true", help="Display the comparisons"
    )
    parser.add_argument("--save", action="store_true", help="Save the comparisons")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print out debug information and progress",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default="output_comparison",
        help="Output Directory for Comparisons",
    )

    parser.add_argument(
        "--n",
        "-n",
        type=int,
        default=test_samples,
        help="number of samples to load for testing",
    )

    parser.add_argument(
        "--shuffle-a", action="store_true", help="whether to shuffle image list A"
    )
    parser.add_argument(
        "--shuffle-b", action="store_true", help="whether to shuffle image list B"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    main(args)
