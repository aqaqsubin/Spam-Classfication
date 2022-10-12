import argparse
import warnings
import pandas as pd

from os.path import join as pjoin
from util import mkdir_p
from preprocess import processing, split_dataset

warnings.filterwarnings(action='ignore')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Build Spam Classification Dataset')
    parser.add_argument('--split',
                        action='store_true',
                        default=False,
                        help='split dataset into train, valid, test')
    
    parser.add_argument('--use_test',
                        action='store_true',
                        default=False,
                        help='build test dataset')

    parser.add_argument('--data_dir',
                        type=str,
                        default='../data')

    parser.add_argument('--save_dir',
                        type=str,
                        default='../data/proc')
    
    
    args = parser.parse_args()
    
    # make save directory
    mkdir_p(args.save_dir)

    data = pd.read_csv(pjoin(args.data_dir, "spam.csv")).dropna(axis=0)
    data.drop_duplicates(subset=['text'], inplace=True)
    
    data = processing(args, data)
    data.to_csv(pjoin(args.data_dir, "proc_data.csv"), index=False)
    
    if args.split:
        split_dataset(args, data)

    
    