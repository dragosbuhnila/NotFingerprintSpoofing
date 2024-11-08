from typing import Dict

import numpy as np
from numpy import ndarray

from modules.calibration import calibration_kfold, evaluate_BayesErrorPlots, evaluate_DCF, evaluate_DCF_fusion, \
    evaluate_BayesErrorPlots_fusion
from modules.evaluation import get_act_DCF
from modules.load_datasets import load_fingerprints, split_db_2to1
from modules.statistics import print_dataset_info

# Print wider lines on console
np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x))
np.set_printoptions(precision=4, suppress=True)


def do_calibration_kfold(scores: ndarray, labels: ndarray, K: int, pi_weight: float, model: str) -> None:
    """ Create a different calibration model for each fold, and use it on that fold.
        Then group the different calibrated folds scores together to have the whole calibrated scores. """
    calibration_kfold(scores, labels, K, name=model, pi_weight=pi_weight)


def evaluate_different_hyperparams_LR(labels: ndarray, pi_eff: float, train=False) -> None:
    """ How to use: comment the calibration part unless you have not added a new K or pi_weight value. """
    llrs_LR = np.load('llrs/llrs_LR.npy')

    Ks = [2, 5, 10, 20, 25, 50, 100]
    # Ks = [100]
    pi_weight_s = [0.05, 0.1, 0.15, 0.2, 0.4, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95]
    # pi_weight_s = []

    if train:
        # Calibration - training and transformation
        for K in Ks:
            for pi_weight in pi_weight_s:
                do_calibration_kfold(llrs_LR, labels, K, pi_weight=pi_weight, model=f"LR_pwt={pi_weight}")

    # Evaluation
    actDCFs = {}
    for K in Ks:
        for pi_weight in pi_weight_s:
            actDCF = get_act_DCF(np.load(f"output_first/clbrtd_scores_LR_pwt={pi_weight}_{K}fold.npy"), labels, pi_eff=pi_eff)
            actDCFs[(K, pi_weight)] = f"{actDCF:.3f}"
    actDCFs = dict(sorted(actDCFs.items(), key=lambda item: item[1]))

    print("Best hyperparameters for LR")
    counter = 10
    for key, value in actDCFs.items():
        print(f"{key}: {value}")
        if counter == 0:
            break
        counter -= 1


def evaluate_different_hyperparams_SVM(labels: ndarray, pi_eff: float, train=False) -> None:
    """ How to use: comment the calibration part unless you have not added a new K or pi_weight value. """
    llrs_SVM = np.load('llrs/llrs_SVM.npy')

    Ks = [2, 5, 10, 20, 25, 50]
    # Ks = []
    pi_weight_s = [0.05, 0.1, 0.15, 0.2, 0.4, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95]
    # pi_weight_s = []

    if train:
        # Calibration - training and transformation
        for K in Ks:
            for pi_weight in pi_weight_s:
                do_calibration_kfold(llrs_SVM, labels, K, pi_weight=pi_weight, model=f"SVM_pwt={pi_weight}")

    actDCFs = {}
    for K in Ks:
        for pi_weight in pi_weight_s:
            actDCF = get_act_DCF(np.load(f"output_first/clbrtd_scores_SVM_pwt={pi_weight}_{K}fold.npy"), labels, pi_eff=pi_eff)
            actDCFs[(K, pi_weight)] = f"{actDCF:.3f}"
    actDCFs = dict(sorted(actDCFs.items(), key=lambda item: item[1]))

    print("Best hyperparameters for SVM")
    counter = 10
    for key, value in actDCFs.items():
        print(f"{key}: {value}")
        if counter == 0:
            break
        counter -= 1


def evaluate_different_hyperparams_GMM(labels: ndarray, pi_eff: float, train=False) -> None:
    """ How to use: comment the calibration part unless you have not added a new K or pi_weight value. """
    llrs_GMM = np.load('llrs/llrs_GMM.npy')

    Ks = [2, 5, 10, 20, 25, 50]
    # Ks = []
    pi_weight_s = [0.05, 0.1, 0.15, 0.2, 0.4, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95]
    # pi_weight_s = []

    if train:
        # Calibration - training and transformation
        for K in Ks:
            for pi_weight in pi_weight_s:
                do_calibration_kfold(llrs_GMM, labels, K, pi_weight=pi_weight, model=f"GMM_pwt={pi_weight}")

    actDCFs = {}
    for K in Ks:
        for pi_weight in pi_weight_s:
            actDCF = get_act_DCF(np.load(f"output_first/clbrtd_scores_GMM_pwt={pi_weight}_{K}fold.npy"), labels,
                                 pi_eff=pi_eff)
            actDCFs[(K, pi_weight)] = f"{actDCF:.3f}"
    actDCFs = dict(sorted(actDCFs.items(), key=lambda item: item[1]))

    print("Best hyperparameters for GMM")
    counter = 10
    for key, value in actDCFs.items():
        print(f"{key}: {value}")
        if counter == 0:
            break
        counter -= 1


def evaluate_LR(labels: ndarray, K:int, pi_weight: float, pi_eff: float) -> None:
    llrs_LR = np.load('llrs/llrs_LR.npy')
    clbrtd_scores = np.load(f"output_first/clbrtd_scores_LR_pwt={pi_weight}_{K}fold.npy")

    evaluate_DCF(llrs_LR, clbrtd_scores, labels, f"LR_pwt={pi_weight}_{K}fold", pi_eff=pi_eff)
    evaluate_BayesErrorPlots(llrs_LR, clbrtd_scores, labels, f"LR_pwt={pi_weight}_{K}fold")


def evaluate_SVM(labels: ndarray, K:int, pi_weight: float, pi_eff: float) -> None:
    llrs_SVM = np.load('llrs/llrs_SVM.npy')
    clbrtd_scores = np.load(f"output_first/clbrtd_scores_SVM_pwt={pi_weight}_{K}fold.npy")

    evaluate_DCF(llrs_SVM, clbrtd_scores, labels, f"SVM_pwt={pi_weight}_{K}fold", pi_eff=pi_eff)
    evaluate_BayesErrorPlots(llrs_SVM, clbrtd_scores, labels, f"SVM_pwt={pi_weight}_{K}fold")


def evaluate_GMM(labels: ndarray, K:int, pi_weight: float, pi_eff: float) -> None:
    llrs_GMM = np.load('llrs/llrs_GMM.npy')
    clbrtd_scores = np.load(f"output_first/clbrtd_scores_GMM_pwt={pi_weight}_{K}fold.npy")

    evaluate_DCF(llrs_GMM, clbrtd_scores, labels, f"GMM_pwt={pi_weight}_{K}fold", pi_eff=pi_eff)
    evaluate_BayesErrorPlots(llrs_GMM, clbrtd_scores, labels, f"GMM_pwt={pi_weight}_{K}fold")


def evaluate_different_hyperparams_fusion(labels: ndarray, pi_eff: float, train=False) -> None:
    """ How to use: comment the calibration part unless you have not added a new K or pi_weight value. """
    llrs_LR = np.load('llrs/llrs_LR.npy')
    llrs_SVM = np.load('llrs/llrs_SVM.npy')
    llrs_GMM = np.load('llrs/llrs_GMM.npy')
    scores = np.vstack((llrs_LR, llrs_SVM, llrs_GMM))

    Ks = [2, 5, 10, 20, 25, 50, 100]
    # Ks = [100]
    pi_weight_s = [0.05, 0.1, 0.15, 0.2, 0.4, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95]
    # pi_weight_s = []

    if train:
        # Calibration - training and transformation
        for K in Ks:
            for pi_weight in pi_weight_s:
                do_calibration_kfold(scores, labels, K, pi_weight=pi_weight, model=f"fusion_pwt={pi_weight}")

    # Evaluation
    actDCFs = {}
    for K in Ks:
        for pi_weight in pi_weight_s:
            actDCF = get_act_DCF(np.load(f"output_first/clbrtd_scores_fusion_pwt={pi_weight}_{K}fold.npy"), labels, pi_eff=pi_eff)
            actDCFs[(K, pi_weight)] = f"{actDCF:.3f}"
    actDCFs = dict(sorted(actDCFs.items(), key=lambda item: item[1]))

    print("Best hyperparameters for LR")
    counter = 10
    for key, value in actDCFs.items():
        print(f"{key}: {value}")
        if counter == 0:
            break
        counter -= 1


def evaluate_fusion(labels: ndarray, K: float, pi_weight: float, pi_eff: float, components_parameters: Dict[str, Dict[str, float]]) -> None:
    clbrtd_scores_LR = np.load(f"output_first/clbrtd_scores_LR_pwt={components_parameters['LR']['pi_w']}_{components_parameters['LR']['K']}fold.npy")
    clbrtd_scores_SVM = np.load(f"output_first/clbrtd_scores_SVM_pwt={components_parameters['SVM']['pi_w']}_{components_parameters['SVM']['K']}fold.npy")
    clbrtd_scores_GMM = np.load(f"output_first/clbrtd_scores_GMM_pwt={components_parameters['GMM']['pi_w']}_{components_parameters['GMM']['K']}fold.npy")
    clbrtd_scores_list = [clbrtd_scores_LR, clbrtd_scores_SVM, clbrtd_scores_GMM]
    clbrtd_scores_names = ["LR (cal.)", "SVM (cal.)", "GMM (cal.)"]

    raw_scores_LR = np.load('llrs/llrs_LR.npy')
    raw_scores_SVM = np.load('llrs/llrs_SVM.npy')
    raw_scores_GMM = np.load('llrs/llrs_GMM.npy')
    raw_scores_list = [raw_scores_LR, raw_scores_SVM, raw_scores_GMM]
    raw_scores_names = ["LR (raw)", "SVM (raw)", "GMM (raw)"]

    fusion_scores = np.load(f"output_first/clbrtd_scores_fusion_pwt={pi_weight}_{K}fold.npy")

    evaluate_DCF_fusion(clbrtd_scores_list + raw_scores_list, fusion_scores, labels,
                        f"fusion_pwt={pi_weight}_{K}fold", pi_eff, clbrtd_scores_names + raw_scores_names)
    evaluate_BayesErrorPlots_fusion(clbrtd_scores_list, fusion_scores, labels, f"fusion_pwt={pi_weight}_{K}fold", clbrtd_scores_names)


def evaluate_fusion_2(labels: ndarray, K: float, pi_weight: float, pi_eff: float, components_parameters: Dict[str, Dict[str, float]]) -> None:
    raw_scores_GMM = np.load('llrs/llrs_GMM.npy')
    raw_scores_list = [raw_scores_GMM]
    raw_scores_names = ["GMM (raw)"]

    fusion_scores = np.load(f"output_first/clbrtd_scores_fusion_pwt={pi_weight}_{K}fold.npy")

    evaluate_BayesErrorPlots_fusion(raw_scores_list, fusion_scores, labels, f"fusion_VS_uncal_GMM_pwt={pi_weight}_{K}fold", raw_scores_names)


if __name__ == '__main__':

    D, L = load_fingerprints()
    (_, _), (D_val, L_val) = split_db_2to1(D, L)
    print_dataset_info(D_val, L_val, "Fingerprints Validation")

    # Our target application is this one
    pi_eff = 0.1

    # """ 1) Find good hyperparameters for each model """
    # evaluate_different_hyperparams_LR(L_val, pi_eff)
    # evaluate_different_hyperparams_SVM(L_val, pi_eff)
    # evaluate_different_hyperparams_GMM(L_val, pi_eff)

    # """ 2) Evaluate the best hyperparameters for each model """
    chosen_parameters = {"LR":  {"K": 50, "pi_w": 0.85},
                         "SVM": {"K": 10, "pi_w": 0.05},
                         "GMM": {"K": 10, "pi_w": 0.1}}
    # evaluate_LR(L_val, K=chosen_parameters["LR"]["K"], pi_weight=chosen_parameters["LR"]["pi_w"], pi_eff=pi_eff)
    # evaluate_SVM(L_val, K=chosen_parameters["SVM"]["K"], pi_weight=chosen_parameters["SVM"]["pi_w"], pi_eff=pi_eff)
    # evaluate_GMM(L_val, K=chosen_parameters["GMM"]["K"], pi_weight=chosen_parameters["GMM"]["pi_w"], pi_eff=pi_eff)

    """ 3) Fusion """
    evaluate_different_hyperparams_fusion(L_val, pi_eff)
    # evaluate_fusion(L_val, K=20, pi_weight=0.05, pi_eff=pi_eff, components_parameters=chosen_parameters)
    evaluate_fusion_2(L_val, K=20, pi_weight=0.05, pi_eff=pi_eff, components_parameters=chosen_parameters)
