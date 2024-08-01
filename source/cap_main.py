import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd

warnings.filterwarnings("ignore")
import sys
sys.path.append("..")
from em.online_expectation_maximization import OnlineExpectationMaximization, OnlineExpectationMaximization_ExtendGC
from evaluation.helpers import *
from onlinelearning.online_learning import *
from onlinelearning.ensemble import *
import math
import argparse
from gcimpute.helper_mask import mask_MCAR
import logging
from collections import Counter

def get_cat_range(X, cat_labels):
    res = {}
    for i in cat_labels:
        features = X[:, i][~np.isnan(X[:, i])]
        res[i] = [min(features), max(features)]
    # print(res)
    return res


def get_cont_indices(X):
    max_ord=14
    indices = np.zeros(X.shape[1]).astype(bool)
    for i, col in enumerate(X.T):
        col_nonan = col[~np.isnan(col)]
        col_unique = np.unique(col_nonan)
        if len(col_unique) > max_ord:
            indices[i] = True
    return indices

def logger_config(log_path,logging_name):

    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


data_file_name = {
'credita':'04_credita.csv',
'npha':'05_npha.csv',
}

cat_labels = {
    'credita':[3, 4, 5, 11],
    'npha':[1, 2, 3, 4, 11, ],

}



def set_seed(s):
    np.random.seed(s)
    rd.seed(s)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="extendGC_OL")
    parser.add_argument('--dataset', default='npha', help="npha or credita")
    parser.add_argument('--miss_rate', type=float,help='[0.4, 0.3, 0.2, 0.1, 0.5]', default=0.5)

    args = parser.parse_args()
    if not os.path.exists(f"../log"):
        os.makedirs(f"../log/")

    logger = logger_config(f"../log/{args.dataset}_MCAR{args.miss_rate}.log", f"{args.dataset}")

    dataset = args.dataset
    last_CER = []
    zimp1_MIDDLE_CER, zimp2_MIDDLE_CER, zimp3_MIDDLE_CER = [], [], []

    for ii in range(1,1+1):
        logger.info(f"{dataset}, seed={ii}")
        #getting  hyperparameter
        contribute_error_rate, window_size_denominator, \
        batch_size_denominator, \
        decay_coef_change,decay_choice,shuffle = get_cap_hyperparameter(dataset)

        data_ori = pd.read_csv(f"../dataset/{data_file_name[dataset]}", header=None, dtype=np.float_)
        data_ori.fillna(value=0, inplace=True)
        data_ori = data_ori.values

        X = data_ori[:, :-1]

        Y_label = data_ori[:, -1]
        Y_label = Y_label.flatten()
        Y_label[Y_label != 1] = 0 # label format{-1, 1}

        X_masked = mask_MCAR(X, mask_fraction=args.miss_rate, seed=ii)
        logger.info(f"MCAR miss_rate = {np.sum(np.isnan(X_masked))/(X_masked.shape[0] * X_masked.shape[1])*100}%")


        n = X_masked.shape[0]
        feat = X_masked.shape[1]


        # index for ord, con, cat features
        all_cont_indices = np.ones(len(X[0])).astype(np.bool_)
        all_cat_indices = np.zeros(len(X[0])).astype(np.bool_)
        all_ord_indices = np.zeros(len(X[0])).astype(np.bool_)

        all_cat_indices[cat_labels[dataset]] = True
        all_ord_indices = ~get_cont_indices(X_masked)

        all_ord_indices[cat_labels[dataset]] = False
        all_cont_indices[cat_labels[dataset]] = False
        all_cont_indices[all_ord_indices] = False

        #setting hyperparameter
        # max_iter = batch_size_denominator * 2
        max_iter = batch_size_denominator
        BATCH_SIZE = math.ceil(n / batch_size_denominator)
        WINDOW_SIZE = math.ceil(n / window_size_denominator)
        NUM_ORD_UPDATES = 1
        batch_c = 8

        logger.info(f"n_samples = {n}, n_fea = {feat}, BATch_size={BATCH_SIZE}, windows_size = {WINDOW_SIZE}")



        cat_range_for_this_dataset = get_cat_range(X, cat_labels[dataset])
        oem = OnlineExpectationMaximization_ExtendGC(all_cont_indices, all_cat_indices, all_ord_indices, window_size=WINDOW_SIZE, cat_range=cat_range_for_this_dataset)
        j = 0
        X_imp = np.empty(X_masked.shape)
        Z_imp1, Z_imp2 = np.empty(X_masked.shape), np.empty(X_masked.shape)
        Z_imp = None
        X_masked = np.array(X_masked)
        # while j <= max_iter:
        while j < max_iter:
            # print(f"j = {j} is working!!!")
            start = (j * BATCH_SIZE) % n
            end = ((j + 1) * BATCH_SIZE) % n
            if end < start:
                indices = np.concatenate((np.arange(end), np.arange(start, n, 1)))
            else:
                indices = np.arange(start, end, 1)
            if decay_coef_change == 1:
                this_decay_coef = batch_c / (j + batch_c)
            else:
                this_decay_coef = 0.5

            # Zimp, Z_imp1, Z_imp2, Ximp
            Z_imp_tmp, Z_imp1[indices, :], Z_imp2[indices,:], X_imp[indices, :] = oem.partial_fit_and_predict(X_masked[indices, :], max_workers=1,decay_coef=this_decay_coef)
            if Z_imp is None:
                Z_imp = np.empty((X_masked.shape[0], len(Z_imp_tmp[0])))
            Z_imp[indices,:] = Z_imp_tmp
            j += 1


        temp = np.ones((n, 1))
        X_masked = pd.DataFrame(X_masked)
        X_zeros = X_masked.fillna(value=0)
        X_input_zero = np.hstack((temp, X_zeros))

        #get the error of latent space
        # Z_imp1
        temp_zim = np.ones((n, 1))
        X_input_z_imp1 = np.hstack((temp, Z_imp1))
        Z_imp_CER1 = ensemble(n,X_input_z_imp1,X_input_zero,Y_label,decay_choice,contribute_error_rate)





        # saving results
        zimp1_MIDDLE_CER.append(Z_imp_CER1.tolist())


        last_CER.append([np.around(Z_imp_CER1[-1], 3)])
        plt.plot(np.arange(len(X_masked)) + 1, Z_imp_CER1, label=f"zimp1_{dataset}_seed{ii}")


    # print(f"{dataset}_{args.miss_rate}, CER mean = {np.around(np.mean(last_CER, axis=0), 3)}, std = {np.around(np.std(last_CER, axis=0), 3)}")
    print(f"{dataset}_{args.miss_rate}, seed = {1}, CER = {np.around(np.mean(last_CER, axis=0), 3)}")

    # last_CER.append([f"{np.around(np.mean(last_CER, axis=0), 3)[0]}/{np.around(np.std(last_CER, axis=0), 3)[0]}",
    #                  f"{np.around(np.mean(last_CER, axis=0), 3)[1]}/{np.around(np.std(last_CER, axis=0), 3)[1]}",
    #                  f"{np.around(np.mean(last_CER, axis=0), 3)[2]}/{np.around(np.std(last_CER, axis=0), 3)[2]}"])

    # saving results on the CER format
    if not os.path.exists(f"../res/{dataset}"):
        os.makedirs(f"../res/{dataset}")


    # saving the curves
    if not os.path.exists(f"../curve"):
        os.makedirs(f"../curve")

    plt.legend()
    plt.show()

