from typing import Tuple

import numpy as np
from numpy import ndarray
from prettytable import PrettyTable

from modules.common_matrix_operations import vrow
from modules.evaluation import get_min_and_act_DCFs
from modules.logistic_regression import LogisticRegressionParams, \
    binary_logreg_prior_weighted_train, binary_logreg_prior_weighted_classify
from modules.plottings import plot_bayes_error_plots


def calibration_single_fold(scores: ndarray, labels: ndarray, name: str, pi_weight: float)\
                            -> Tuple[ndarray, LogisticRegressionParams]:
    """ Outputs the calibration pwLR model (w, b, pi_weight, variant) to a pkl file,
          and the calibrated scores to a npy file.
        Basically training and transformation at once """
    if scores.ndim == 1:
        scores = vrow(scores)
    if scores.shape[1] != labels.size:
        raise ValueError("Scores and labels should have the same number of samples")

    SCAL, SVAL = scores[:, ::3], np.hstack([scores[:, 1::3], scores[:, 2::3]])
    LCAL, LVAL = labels[::3], np.hstack([labels[1::3], labels[2::3]])

    w, b, pi_weight = binary_logreg_prior_weighted_train(SCAL, LCAL, l=0, pi_weight=pi_weight)  # l=0 => no regularization
    llrs_calibrated, _, _ = binary_logreg_prior_weighted_classify(w, b, SVAL, pi_weight=pi_weight)

    pwLR_model = LogisticRegressionParams(w, b, pi_weight, "priorweighted")
    pwLR_model.save_to_file(f"output_first/clbrtn_model_{name}.pkl")
    print(f"[[Saved clbrtn_model_{name}.pkl]]")
    np.save(f"output_first/clbrtd_scores_{name}.npy", llrs_calibrated)
    print(f"[[Saved clbrtd_scores_{name}.npy]]")

    return llrs_calibrated, pwLR_model


def calibration_single_fold__only_training(cal_scores: ndarray, cal_labels: ndarray, name: str,
                                           pi_weight: float, store=True) -> LogisticRegressionParams:
    """ Outputs the calibration pwLR model (w, b, pi_weight, variant) to a pkl file.
        cal stands for calibration
        clbrtd stands for calibrated """
    if cal_scores.ndim == 1:
        cal_scores = vrow(cal_scores)
    w, b, pi_weight = binary_logreg_prior_weighted_train(cal_scores, cal_labels, l=0, pi_weight=pi_weight)  # l=0 => no regularization

    pwLR_model = LogisticRegressionParams(w, b, pi_weight, "priorweighted")
    if store:
        pwLR_model.save_to_file(f"output_first/clbrtn_model_{name}.pkl")
        print(f"[[Saved clbrtn_model_{name}.pkl]]")

    return pwLR_model


def calibration_single_fold__only_transform(lr_model: LogisticRegressionParams, val_scores: ndarray, name: str, store=True) -> ndarray:
    """ Outputs the pwLR calibrated scores to a npy file """
    if lr_model.get_variant() != "priorweighted":
        raise ValueError("Model should be priorweighted")

    w, b = lr_model.get_wb()
    pi_weight = lr_model.get_pi()

    if val_scores.ndim == 1:
        val_scores = vrow(val_scores)
    llrs_calibrated, _, _ = binary_logreg_prior_weighted_classify(w, b, val_scores, pi_weight=pi_weight)

    if store:
        np.save(f"output_first/clbrtd_scores_{name}.npy", llrs_calibrated)
        print(f"[[Saved clbrtd_scores_{name}.npy]]")

    return llrs_calibrated


def calibration_kfold(scores: ndarray, labels: ndarray, K: int, name: str, pi_weight: float) -> ndarray:
    """ Outputs the K-fold calibrated scores, along with each folds model params and scores.
        You can use this to fine-tune the best pi_weight value for the pwLR model to use on the whole set."""
    if scores.ndim == 1:
        scores = vrow(scores)
    if K < 2:
        raise ValueError("K should be at least 2")
    if scores.shape[1] != labels.size:
        raise ValueError("Scores and labels should have the same number of samples")

    # Split data into K folds
    scores_folds = []
    labels_folds = []
    clbrtd_scores = np.zeros((1, scores.shape[1]))
    for i in range(K):
        scores_folds.append(scores[:, i::K])
        labels_folds.append(labels[i::K])

    models = []
    # Train K models and apply calibration to each fold
    for i in range(K):
        cal_scores_i = np.hstack([scores_folds[j] for j in range(K) if j != i])
        cal_labels_i = np.hstack([labels_folds[j] for j in range(K) if j != i])
        val_scores_i = scores_folds[i]
        # val_labels_i = labels_folds[i]

        # Train i-th model using 4/5th of the data
        pwLR_model = calibration_single_fold__only_training(cal_scores_i, cal_labels_i, f"{name}_fold_{i}", pi_weight=pi_weight, store=False)
        models.append(pwLR_model)

        # Apply calibration to i-th fold
        clbrtd_scores_i = calibration_single_fold__only_transform(pwLR_model, val_scores_i, f"{name}_fold_{i}", store=False)
        clbrtd_scores[:, i::K] = clbrtd_scores_i

    np.save(f"output_first/clbrtd_scores_{name}_{K}-fold.npy", clbrtd_scores)
    print(f"[[Saved clbrtd_scores_{name}_{K}-fold.npy]]")
    np.save(f"output_first/clbrtn_models_{name}_{K}-fold.npy", models)
    print(f"[[Saved clbrtn_models_{name}_{K}-fold.npy]]")

    return clbrtd_scores


def evaluate_DCF(raw_scores: ndarray, clbrtd_scores: ndarray, val_labels: ndarray, name: str, pi_eff: float) -> None:
    """ Outputs DCFs for raw and calibrated versions of the same original scores """
    minDCF_raw, actDCF_raw = get_min_and_act_DCFs(raw_scores, val_labels, pi_eff=pi_eff)
    minDCF_cal, actDCF_cal = get_min_and_act_DCFs(clbrtd_scores, val_labels, pi_eff=pi_eff)

    # Print Table
    table = PrettyTable()
    table.add_column("", ["Raw", "Calibrated"])
    table.add_column("minDCF", [f"{minDCF_raw:.3f}", f"{minDCF_cal:.3f}"])
    table.add_column("actDCF", [f"{actDCF_raw:.3f}", f"{actDCF_cal:.3f}"])
    table.title = f"System {name}"
    print(table)

    # Save to file
    with open(f'output_first/DCFs_rawNcal_{name}.txt', 'w') as f:
        print(table, file=f)


def evaluate_DCF_fusion(clbrtd_scores_list: list[ndarray], fusion_scores: ndarray, val_labels: ndarray, name: str,
                        pi_eff: float, clbrtd_scores_names: list[str]) -> None:
    """ Outputs DCFs for calibrated scores of the models of the fusion, and for fusion score itself """
    if len(clbrtd_scores_list) != len(clbrtd_scores_names):
        raise ValueError("scores_raw_list and scores_raw_names should have the same number of elements"
                            f" instead have {len(clbrtd_scores_list)} and {len(clbrtd_scores_names)}")
    for (i, clbrtd_scores) in enumerate(clbrtd_scores_list):
        if clbrtd_scores.size != fusion_scores.size:
            raise ValueError(f"Scores '{clbrtd_scores_names[i]}' and fusion scores should have the same number of "
                             f"samples, instead have {clbrtd_scores.size} and {fusion_scores.size}")

    minDCFs = []
    actDCFs = []
    for clbrtd_scores in clbrtd_scores_list:
        clbrtd_minDCF, clbrtd_actDCF = get_min_and_act_DCFs(clbrtd_scores, val_labels, pi_eff=pi_eff)
        minDCFs.append(f"{clbrtd_minDCF:.3f}")
        actDCFs.append(f"{clbrtd_actDCF:.3f}")

    minDCF_fusion, actDCF_fusion = get_min_and_act_DCFs(fusion_scores, val_labels, pi_eff=pi_eff)
    minDCFs.append(f"{minDCF_fusion:.3f}")
    actDCFs.append(f"{actDCF_fusion:.3f}")
    clbrtd_scores_names.append("Fusion")

    # Print Table
    table = PrettyTable()
    table.add_column("", clbrtd_scores_names)
    table.add_column("minDCF", minDCFs)
    table.add_column("actDCF", actDCFs)
    table.title = f"System {name}"
    print(table)

    # Save to file
    with open(f'output_first/DCFs_rawsNfusion_{name}.txt', 'w') as f:
        print(table, file=f)


def evaluate_BayesErrorPlots(raw_scores: ndarray, clbrtd_scores: ndarray, val_labels: ndarray, name: str) -> None:
    """ Outputs Bayes Error Plots for raw and calibrated (and min) versions of the same original scores """
    plot_bayes_error_plots([clbrtd_scores], val_labels, [name], llrs_precal_list=[raw_scores], range=(-3, 3, 21),
                           title=name)


def evaluate_BayesErrorPlots_fusion(clbrtd_scores_list: list[ndarray], fusion_scores: ndarray, val_labels: ndarray,
                                    name: str, clbrtd_scores_names: list[str]) -> None:
    """ Outputs Bayes Error Plots for raw and calibrated (and min) versions of the same original scores """

    all_scores = [fusion_scores] + clbrtd_scores_list
    all_names = ["fusion"] + clbrtd_scores_names
    plot_bayes_error_plots(all_scores, val_labels, all_names, range=(-3, 3, 21), title=name)