import pandas as pd
import numpy as np
import torch
import random, argparse, time, os, re, logging
from HP_list import Optimal_Uni_algo_HP_dict

from TSB_AD.model_wrapper import Semisupervise_AD_Pool, Unsupervise_AD_Pool, run_Semisupervise_AD, run_Unsupervise_AD
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

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

def configure_logger(filename):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Generating Anomaly Scores and Saving Evaluation Results')
    parser.add_argument('--dataset_dir', type=str, help='path to the dataset', default='TSB-AD-U/')
    parser.add_argument('--file_lsit', type=str, help='path to the file list', default='TSB-AD-U-Eval-List.csv')
    parser.add_argument('--score_dir', type=str, help='path to save the anomaly score', default='.../score/uni/')
    parser.add_argument('--save_dir', type=str, help='path to save the evaluation result', default='.../eval/uni/')
    parser.add_argument('--save', type=bool, help='whether to save the evaluation result',  default=True)
    parser.add_argument('--AD_Name', type=str, help='the name of the anomaly detector to use', default='IForest')
    parser.add_argument('--resume', type=bool, help='whether to resume', default=True)
    args = parser.parse_args()

    ## Resume
    if args.resume and os.path.exists(f'{args.save_dir}/{args.AD_Name}.csv'):
        df = pd.read_csv(f'{args.save_dir}/{args.AD_Name}.csv')
        write_csv =  df.values.tolist()
        Resume_file_name_list = df['file'].tolist()
    else:
        write_csv = []
        Resume_file_name_list = []

    target_dir = os.path.join(args.score_dir, args.AD_Name)
    os.makedirs(target_dir, exist_ok = True)
    logger = configure_logger(filename=f'{target_dir}/000_run_{args.AD_Name}.log')

    file_list = pd.read_csv(args.file_lsit)['file_name'].values
    Optimal_Det_HP = Optimal_Uni_algo_HP_dict[args.AD_Name]
    print('Optimal_Det_HP: ', Optimal_Det_HP)

    for filename in file_list:
        if filename in Resume_file_name_list: continue
        print('Processing:{} by {}'.format(filename, args.AD_Name))

        file_path = os.path.join(args.dataset_dir, filename)
        df = pd.read_csv(file_path).dropna().to_numpy()
        data = df[:,0].astype(float).reshape(-1, 1)
        label = df[:,1].astype(int)

        feats = data.shape[1]
        slidingWindow = find_length_rank(data[:,0].reshape(-1, 1), rank=1)
        train_index = filename.split('.')[0].split('_')[-2]
        data_train = data[:int(train_index), :]

        start_time = time.time()

        Run_AD_Name = args.AD_Name.split('.')[0]

        if Run_AD_Name in Semisupervise_AD_Pool:
            output = run_Semisupervise_AD(Run_AD_Name, data_train, data, **Optimal_Det_HP)
        elif Run_AD_Name in Unsupervise_AD_Pool:
            output = run_Unsupervise_AD(Run_AD_Name, data, **Optimal_Det_HP)
        else:
            raise Exception(f"{Run_AD_Name} is not defined")

        end_time = time.time()
        run_time = end_time - start_time

        if isinstance(output, np.ndarray):
            logging.info(f'Success at {filename} using {args.AD_Name} | Time cost: {run_time:.3f}s at length {len(label)}')
            np.save(target_dir+'/'+filename.split('.')[0]+'.npy', output)
        else:
            logger.error(f'At {filename}: '+output)

        ### whether to save the evaluation result
        if args.save:
            try:
                evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow)
                print('evaluation_result: ', evaluation_result)
                list_w = list(evaluation_result.values())
            except:       
                list_w = [0]*9
            list_w.insert(0, run_time)
            list_w.insert(0, filename)
            write_csv.append(list_w)

            ## Temp Save
            col_w = list(evaluation_result.keys())
            col_w.insert(0, 'Time')
            col_w.insert(0, 'file')
            w_csv = pd.DataFrame(write_csv, columns=col_w)
            w_csv.to_csv(f'{args.save_dir}/{args.AD_Name}.csv', index=False)