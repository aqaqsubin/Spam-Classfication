import argparse
import warnings
import pandas as pd

from os.path import join as pjoin
from util import mkdir_p
from preprocess import processing, split_dataset

warnings.filterwarnings(action='ignore')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Build Spam Classification Dataset')

    parser.add_argument('--data_path',
                        type=str,
                        default='../data/test.csv')
    
    parser.add_argument('--save_dir',
                        type=str,
                        default='../data/proc')
    
    args = parser.parse_args()
    
    # make save directory
    mkdir_p(args.save_dir)

    data = pd.read_csv(args.data_path)
    
    data = processing(args, data, is_test=True)
    data.to_csv(pjoin(args.save_dir, "test.csv"), index=False)
    
    