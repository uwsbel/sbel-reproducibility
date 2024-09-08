import time
from metrics.measures import *
from metrics.segmentation_comparison import *

import argparse
import sys
import os

def main(args):
    #save the configuration to a bash script
    if(not os.path.exists(args.output_path)):
        os.makedirs(args.output_path, exist_ok=True)

    run_file = open(os.path.join(args.output_path,"run.sh"),'w')
    run_file.write("#!/usr/bin/env bash\n")
    command = "python3 "
    for a in sys.argv:
        command += a + " "
    run_file.write(command + "\n")
    run_file.close()

    #create class for segmentation comparison
    comparitor = SegmentationComparitor(args.output_path,write=args.save,nameA=args.a_name,nameB=args.b_name)

    if(args.mode == "generate_patches"):
        comparitor.load_dataset(args.a_path,args.a_preds,args.n)
        comparitor.generate_patches(patch_size=args.patch_size,patches_per_img=args.ppi,output_path=args.output_path)
    elif(args.mode == "compare_patches"):
        pass
        comparitor.load_patches(args.a_path,args.b_path,args.n,shuffle=args.shuffle)
        comparitor.compare_patches(args) # threshold=args.patch_threshold,vis=args.vis,save=args.save,output_dir=args.output_path,sample_max=args.save_max)
    elif(args.mode == "total_performance"):
        comparitor.load_dataset(args.a_path,args.a_preds,args.n)
        comparitor.evaluate_dataset_performance(args)
    else:
        raise NotImplementedError("must select implemented mode")

def parseargs():
    # === DEFAULTS === #
    test_samples = -1
    output_path = "output_comparison"

    parser = argparse.ArgumentParser(description='Segmentation Comparison. \
        Requires testing has been completed and tested predictions, labels, \
        and images have been saved in their output size')

    # information about the network
    mode_choices =["generate_patches", "compare_patches", "total_performance"]
    parser.add_argument('--mode',choices=mode_choices, help='mode for comparison')

    perf_choices =["ACC","IOU"]
    parser.add_argument('--perf-measure',"-pm",default=perf_choices[0],choices=perf_choices, help='performance measure for comparison')

    #patch comparison parameters
    parser.add_argument('--patch_size', "-ps", type=int, nargs="+", default=[64,64],
                        help="size of the patch for comparison centered around the object")
    parser.add_argument('--patch_threshold', "-pt", type=float, default=0.9,
                        help="threshold of patch similarity to be considered as part of the same batch")
    parser.add_argument('-ppi', type=int, default=8,
                        help="numpy of patches per image")

    parser.add_argument('--min-batch-size','-bs', type=int, default=1,
                        help="mimimum batch size")
    parser.add_argument("--save-max","-sm",type=int,default=100,help="max number of samples to save")
    parser.add_argument("--weight","-w",action="store_true",help="weight to prevent overcomparion of similar regions")
    parser.add_argument("--reduce-gt","-rgt",action="store_true",help="reduce size of GT for faster comparison for near neighbors")
    parser.add_argument("--skvd",action="store_true",help="calculate skvd metric")

    #information about the datasets being compared
    parser.add_argument('--a_name', type=str, default='Dataset A',
                        help="Name of first dataset in the comparison")
    parser.add_argument('--a_path', type=str, default="output_a",
                        help="Path to first dataset's metrics files")
    parser.add_argument('--a_preds', type=str, default="preds_a",
                        help="Path to first dataset's prediction files")

    parser.add_argument('--b_name', type=str, default='Dataset B',
                        help="Name of second dataset in the comparison")
    parser.add_argument('--b_path', type=str, default="output_b",
                        help="Path to second dataset's metrics files")
    
    
    #miscellaneous parameters
    parser.add_argument("--vis","-v",action="store_true",help="Display the comparisons")
    parser.add_argument("--save","-s",action="store_true",help="Save the comparisons")

    parser.add_argument("--shuffle",action="store_true",help="shuffle lists before selecting n")
    

    parser.add_argument("--output_path","-o",type=str, default=output_path,
                        help="Output Directory for Comparisons")

    parser.add_argument('-n', type=int, default=test_samples,
                        help="number of samples to load for comparison")
    parser.add_argument('--ntest', type=int, default=-1,
                        help="number of samples to test during comparison")


    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    main(args)
