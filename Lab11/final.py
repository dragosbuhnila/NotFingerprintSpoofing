from typing import Tuple

import numpy as np
from numpy import ndarray
from prettytable import PrettyTable

from modules.best_classifiers_clean_version import LogisticRegressionParams, SVMParameters, GMMParameters, classify_GMM, \
    binary_logreg_quadratic__classify, run_dual_with_kernel__only_classification
from modules.calibration import calibration_kfold, evaluate_DCF_fusion, evaluate_BayesErrorPlots_fusion
from modules.svm import rbf_kernel
from modules.evaluation import get_min_and_act_DCFs
from modules.load_datasets import load_eval_data
from modules.plottings import plot_bayes_error_plots


def eval(llrs: ndarray, L: ndarray, pi_eff: float, name: str) -> None:
    dcf_min_GMM, dcf_act_GMM = get_min_and_act_DCFs(llrs, L, pi_eff)

    # Print Table
    table = PrettyTable()
    table.add_column("", ["minDCF", "actDCF"])
    table.add_column("", [f"{dcf_min_GMM:.3f}", f"{dcf_act_GMM:.3f}"])
    table.title = f"System {name}"
    print(table)

    # Save to file
    with open(f'output_first/DCFs_min_act_{name}.txt', 'w') as f:
        print(table, file=f)
        print(f"[[Saved DCFs_min_act_{name}.txt]]")

    plot_bayes_error_plots([llrs], L, [name], name)


def do_fusion_calibration_kfold(llrs_LR: ndarray, llrs_SVM: ndarray, llrs_GMM: ndarray, L: ndarray,
                                K: int, pi_weight: float) -> None:
    """ Create a different calibration model for each fold, and use it on that fold.
        Then group the different calibrated folds scores together to have the whole calibrated scores. """

    scores = np.vstack([llrs_LR, llrs_SVM, llrs_GMM])
    labels = L

    calibration_kfold(scores, labels, K, "final_fusion", pi_weight=pi_weight)


def do_fusion_evaluation_kfold(best_nofusion_scores_list: list[ndarray], fusion_scores: ndarray, labels: ndarray,
                               model_names: list[str], K: int, pi_eff: float) -> None:
    # evaluate_DCF_fusion(best_nofusion_scores_list, fusion_scores, labels,
    #                     f"final_fusion_{K}fold", pi_eff, model_names)
    evaluate_BayesErrorPlots_fusion(best_nofusion_scores_list, fusion_scores, labels,
                                    f"fusion_{K}fold", model_names)


if __name__ == '__main__':
    # Load models
    qLRmodel = LogisticRegressionParams.load_from_file("best_models/model_qLR.pkl")
    rbfSVMmodel = SVMParameters.load_from_file("best_models/model_rbfSVM.pkl")
    diag8cGMMmodel_falseClass = GMMParameters.load_from_file("best_models/model_FalseClass_8-cGMM_diag.pkl")
    diag8cGMMmodel_trueClass = GMMParameters.load_from_file("best_models/model_TrueClass_8-cGMM_diag.pkl")

    # Load evaluation data
    D, L = load_eval_data()

    """ Evaluate Best (GMM) """
    # llrs_GMM = classify_GMM(D, diag8cGMMmodel_falseClass, diag8cGMMmodel_trueClass)
    # np.save(f"output_first/scores_diag8cGMM.npy", llrs_GMM)
    # print(f"[[Saved scores_diag8cGMM.npy]]")
    #
    # clbrtd_llrs_GMM = calibration_kfold(llrs_GMM, L, K=10, pi_weight=0.1, name=f"diag8cGMM_pwt-0.1")
    #
    # eval(clbrtd_llrs_GMM, L, 0.1, "clbrtd_diag8cGMM")

    """ Classify three + fusion """
    # Logistic Regression + its calibration
    # w, b = qLRmodel.get_wb()
    # pi_emp = qLRmodel.get_pi()
    # llrs_LR, _, _ = binary_logreg_quadratic__classify(w, b, D, pi_emp)
    # # np.save(f"output_first/scores_qLR.npy", llrs_LR)
    # # print(f"[[Saved scores_qLR.npy]]")
    #
    # clbrtd_llrs_LR = calibration_kfold(llrs_LR, L, K=50, pi_weight=0.85, name=f"qLR_pwt-0.85")
    # np.save(f"output_first/clbrtd_scores_qLR.npy", clbrtd_llrs_LR)
    # print(f"[[Saved clbrtd_scores_qLR.npy]]")
    #
    # # Support Vector Machine + its calibration
    # # llrs_SVM, _ = run_dual_with_kernel__only_classification(rbfSVMmodel, D, kernel=rbf_kernel,
    # #                                                         kernel_params=(np.exp(-2), 1))
    # # np.save(f"output_first/scores_rbfSVM.npy", llrs_SVM)
    # # print(f"[[Saved scores_rbfSVM.npy]]")
    # llrs_SVM = np.load(f"output_first/scores_rbfSVM.npy")
    #
    # clbrtd_llrs_SVM = calibration_kfold(llrs_SVM, L, K=10, pi_weight=0.05, name=f"rbfSVM_pwt-0.05")
    # np.save(f"output_first/clbrtd_scores_rbfSVM.npy", clbrtd_llrs_SVM)
    # print(f"[[Saved clbrtd_scores_rbfSVM.npy]]")

    # GMM (no calibration)
    # llrs_GMM = classify_GMM(D, diag8cGMMmodel_falseClass, diag8cGMMmodel_trueClass)
    # np.save(f"output_first/scores_diag8cGMM.npy", llrs_GMM)
    # print(f"[[Saved scores_diag8cGMM.npy]]")

    # Fusion (will fuse the raw scores, but then compare with calibrated ones for LR and SVM
    # do_fusion_calibration_kfold(llrs_LR, llrs_SVM, llrs_GMM, L, K=20, pi_weight=0.05)

    """ Evaluate three + fusion """
    best_nofusion_scores_list = [np.load(f"output_first/clbrtd_scores_qLR_pwt-0.85_50-fold.npy"),
                                np.load(f"output_first/clbrtd_scores_rbfSVM_pwt-0.05_10-fold.npy"),
                                np.load(f"output_first/scores_diag8cGMM.npy")]
    fusion_scores = np.load("output_first/clbrtd_scores_final_fusion_20-fold.npy")
    model_names = ["qLR (cal.)", "rbfSVM (cal.)", "diag8cGMM (raw)"]

    do_fusion_evaluation_kfold(best_nofusion_scores_list, fusion_scores, L, model_names, K=20, pi_eff=0.1)




