# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: MIT License

import pandas as pd
import torch
import random
from evaluation.metrics import get_metrics
from utils.slidingWindows import find_length_rank
from model_wrapper import *
from HP_list import Optimal_Uni_algo_HP_dict

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA Available: ", torch.cuda.is_available())
print("cuDNN Version: ", torch.backends.cudnn.version())


AD_Name = 'IForest'
filename = '001_NAB_data_Traffic_4_624_2087.csv'
data_direc = f'../Datasets/TSB-AD-U/{filename}'

df = pd.read_csv(data_direc).dropna()
data = df.iloc[:, 0:-1].values.astype(float)
label = df['Label'].astype(int).to_numpy()

slidingWindow = find_length_rank(data, rank=1)
train_index = filename.split('.')[0].split('_')[-2]
data_train = data[:int(train_index), :]
Optimal_Det_HP = Optimal_Uni_algo_HP_dict[AD_Name]

if AD_Name in Semisupervise_AD_Pool:
    output = run_Semisupervise_AD(AD_Name, data_train, data, **Optimal_Det_HP)
elif AD_Name in Unsupervise_AD_Pool:
    output = run_Unsupervise_AD(AD_Name, data, **Optimal_Det_HP)
else:
    raise Exception(f"{AD_Name} is not defined")

if isinstance(output, np.ndarray):
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
    print('Evaluation Result: ', evaluation_result)
else:
    print(f'At {filename}: '+output)

