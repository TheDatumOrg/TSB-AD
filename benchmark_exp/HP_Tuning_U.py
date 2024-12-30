# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import numpy as np
import torch
import random, argparse, time, os
import itertools
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Uni_algo_HP_dict

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='HP Tuning')
    parser.add_argument('--dataset_dir', type=str, default='../Datasets/TSB-AD-U/')
    parser.add_argument('--file_lsit', type=str, default='../Datasets/File_List/TSB-AD-U-Tuning.csv')
    parser.add_argument('--save_dir', type=str, default='eval/HP_tuning/uni/')
    parser.add_argument('--AD_Name', type=str, default='IForest')
    args = parser.parse_args()

    file_list = pd.read_csv(args.file_lsit)['file_name'].values

    Det_HP = Uni_algo_HP_dict[args.AD_Name]

    keys, values = zip(*Det_HP.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    write_csv = []
    for filename in file_list:
        print('Processing:{} by {}'.format(filename, args.AD_Name))

        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        # print('data: ', data.shape)
        # print('label: ', label.shape)

        feats = data.shape[1]
        slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
        train_index = filename.split('.')[0].split('_')[-3]
        data_train = data[:int(train_index), :]

        for params in combinations:

            if args.AD_Name in Semisupervise_AD_Pool:
                output = run_Semisupervise_AD(args.AD_Name, data_train, data, **params)
            elif args.AD_Name in Unsupervise_AD_Pool:
                output = run_Unsupervise_AD(args.AD_Name, data, **params)
            else:
                raise Exception(f"{args.AD_Name} is not defined")
                
            try:
                evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                print('evaluation_result: ', evaluation_result)
                list_w = list(evaluation_result.values())
            except:
                list_w = [0]*9
            list_w.insert(0, params)
            list_w.insert(0, filename)
            write_csv.append(list_w)

            ## Temp Save
            col_w = list(evaluation_result.keys())
            col_w.insert(0, 'HP')
            col_w.insert(0, 'file')
            w_csv = pd.DataFrame(write_csv, columns=col_w)

            w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}.csv', index=False)