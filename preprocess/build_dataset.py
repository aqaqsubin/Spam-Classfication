import argparse
import warnings
import pandas as pd

from os.path import join as pjoin
from util import mkdir_p
from preprocess import processing, split_dataset

warnings.filterwarnings(action='ignore')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Build Hate-Speech Detection Dataset')
    parser.add_argument('--split',
                        action='store_true',
                        default=False,
                        help='split dataset into train, valid, test')

    parser.add_argument('--preprocessing',
                        action='store_true',
                        default=False,
                        help='data preprocessing')

    parser.add_argument('--data_dir',
                        type=str,
                        default='../data')

    parser.add_argument('--result_dir',
                        type=str,
                        default='../result')
    
    
    args = parser.parse_args()
    
    # make result directory
    mkdir_p(args.result_dir)

    data = pd.read_csv(pjoin(args.data_dir, 'data.csv')).dropna(axis=0)
    if args.preprocessing:
        data = processing(args, data)
    
    if args.split:
        split_dataset(args, data)

    
    