import sys
import argparse
import os
import automl

# Parse arguments
parser = argparse.ArgumentParser(description='Run AutoML.')
parser.add_argument('dataset', nargs='?', default=None, help='Name of the specific dataset to process')
parser.add_argument('--debug', action='store_true', help='Enable verbose debug output')
args = parser.parse_args()

# Use current directory as data_dest
data_dest = os.getcwd() 

if args.debug:
    print(f"Testing with data_dest: {data_dest}")
    if args.dataset:
        print(f"Targeting single dataset: {args.dataset}")

automl.fit(data_dest, dataset_name=args.dataset, debug=args.debug)
automl.eval()
