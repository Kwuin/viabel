import argparse
import json
import random
from datetime import datetime

import numpy
import numpy as np


def set_seed_everywhere(seed):
    np.random.seed(seed)
    random.seed(seed)


def write_info(args, fp):
    data = {
        'timestamp': str(datetime.now()),
        'args': str(args)
    }
    with open(fp, 'w') as f:
        json.dump(data, f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dim', default=2.0, type=float)
    parser.add_argument('--learning_rate', default=0.05, type=float)
    parser.add_argument('--n_iters', default=30000, type=int)
    parser.add_argument('--fixed_lr', default=True, type=bool)


    # Gaussian parameters
    parser.add_argument('--mean', default= np.array([0,0]), type=np.array)
    parser.add_argument('--cov', default= np.array([0, 0]), type=np.array)
    parser.add_argument('--init_var_', default=np.array([0, 0]), type=np.array)




    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--save_interval', default=10, type=int)
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--model_save_dir', default='checkpoints', type=str)
    parser.add_argument('--save_ckpt', default=False, action='store_true')


    args = parser.parse_args()

    return args
