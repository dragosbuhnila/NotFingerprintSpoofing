import numpy as np

from modules.calibration import calibration_single_fold, calibration_single_fold__only_transform, evaluate_DCF, \
    evaluate_BayesErrorPlots, calibration_kfold, calibration_single_fold__only_training, evaluate_DCF_fusion, \
    evaluate_BayesErrorPlots_fusion
from modules.logistic_regression import LogisticRegressionParams
from modules.plottings import plot_bayes_error_plots


def evaluate_no_calibration():
    scores_sys_1 = np.load("Data/scores_1.npy")
    scores_sys_2 = np.load("Data/scores_2.npy")
    labels = np.load("Data/labels.npy")

    # minDCF_sys_1, actDCF_sys_1 = get_min_and_act_DCFs(scores_sys_1, labels, 0.2)
    # minDCF_sys_2, actDCF_sys_2 = get_min_and_act_DCFs(scores_sys_2, labels, 0.2)
    #
    # # Print Table
    # table = PrettyTable()
    # table.add_column("", ["minDCF", "actDCF"])
    # table.add_column("System 1", [minDCF_sys_1, actDCF_sys_1])
    # table.add_column("System 2", [minDCF_sys_2, actDCF_sys_2])
    # table.title = "Raw scores"
    # print(table)

    # Plot Bayes Error Plot
    plot_bayes_error_plots([scores_sys_1], labels, ["System 1"], "sys1_no_cal")
    plot_bayes_error_plots([scores_sys_2], labels, ["System 2"], "sys2_no_cal")





def do_calibration_single_fold(pi_weight: float) -> None:
    """ Create calibration model from calibration subset,
        and use it to transform the scores of: the validation subset, and the new evaluation set"""
    scores_1 = np.load("Data/scores_1.npy")
    scores_2 = np.load("Data/scores_2.npy")
    labels = np.load("Data/labels.npy")
    eval_scores_1 = np.load("Data/eval_scores_1.npy")
    eval_scores_2 = np.load("Data/eval_scores_2.npy")

    calibration_single_fold(scores_1, labels, "1_singlefold", pi_weight=pi_weight)
    calibration_single_fold(scores_2, labels, "2_singlefold", pi_weight=pi_weight)

    calibration_single_fold__only_transform(LogisticRegressionParams.load_from_file("output_first/clbrtd_model_1.pkl"),
                                            eval_scores_1, "e1_singlefold")
    calibration_single_fold__only_transform(LogisticRegressionParams.load_from_file("output_first/clbrtd_model_2.pkl"),
                                            eval_scores_2, "e2_singlefold")


def do_evaluation_single_fold(pi_eff: float) -> None:
    scores_1 = np.load("Data/scores_1.npy")
    scores_2 = np.load("Data/scores_2.npy")
    labels = np.load("Data/labels.npy")
    eval_scores_1 = np.load("Data/eval_scores_1.npy")
    eval_scores_2 = np.load("Data/eval_scores_2.npy")

    scores_1_val = np.hstack([scores_1[1::3], scores_1[2::3]])
    scores_2_val = np.hstack([scores_2[1::3], scores_2[2::3]])
    labels_val = np.hstack([labels[1::3], labels[2::3]])
    eval_labels = np.load("Data/eval_labels.npy")

    evaluate_DCF(scores_1_val, np.load("output_first/clbrtd_scores_1_singlefold.npy"), labels_val, "1", pi_eff=pi_eff)
    evaluate_DCF(scores_2_val, np.load("output_first/clbrtd_scores_2_singlefold.npy"), labels_val, "2", pi_eff=pi_eff)
    evaluate_DCF(eval_scores_1, np.load("output_first/clbrtd_scores_e1_singlefold.npy"), eval_labels, "e1", pi_eff=pi_eff)
    evaluate_DCF(eval_scores_2, np.load("output_first/clbrtd_scores_e2_singlefold.npy"), eval_labels, "e2", pi_eff=pi_eff)

    evaluate_BayesErrorPlots(scores_1_val, np.load("output_first/clbrtd_scores_1_singlefold.npy"), labels_val, "1")
    evaluate_BayesErrorPlots(scores_2_val, np.load("output_first/clbrtd_scores_2_singlefold.npy"), labels_val, "2")
    evaluate_BayesErrorPlots(eval_scores_1, np.load("output_first/clbrtd_scores_e1_singlefold.npy"), eval_labels, "e1")
    evaluate_BayesErrorPlots(eval_scores_2, np.load("output_first/clbrtd_scores_e2_singlefold.npy"), eval_labels, "e2")


def do_calibration_kfold(K: int, pi_weight: float) -> None:
    """ Create a different calibration model for each fold, and use it on that fold.
        Then group the different calibrated folds scores together to have the whole calibrated scores. """
    scores_1 = np.load("Data/scores_1.npy")
    scores_2 = np.load("Data/scores_2.npy")
    labels = np.load("Data/labels.npy")

    calibration_kfold(scores_1, labels, K, "sys1", pi_weight=pi_weight)
    calibration_kfold(scores_2, labels, K, "sys2", pi_weight=pi_weight)


def do_evaluation_kfold(K:int, pi_eff: float) -> None:
    scores_1 = np.load("Data/scores_1.npy")
    scores_2 = np.load("Data/scores_2.npy")
    clbrtd_scores_1 = np.load(f"output_first/clbrtd_scores_sys1_{K}fold.npy")
    clbrtd_scores_2 = np.load(f"output_first/clbrtd_scores_sys2_{K}fold.npy")
    labels = np.load("Data/labels.npy")

    evaluate_DCF(scores_1, clbrtd_scores_1, labels, f"{K}foldSys1", pi_eff=pi_eff)
    evaluate_DCF(scores_2, clbrtd_scores_2, labels, f"{K}foldSys2", pi_eff=pi_eff)

    evaluate_BayesErrorPlots(scores_1, clbrtd_scores_1, labels, f"{K}foldSys1")
    evaluate_BayesErrorPlots(scores_2, clbrtd_scores_2, labels, f"{K}foldSys2")


def do_calibration_training_all(pi_weight: float) -> None:
    """ Create calibration model from the entirety of calibration subset and validation subset combined """
    scores_1 = np.load("Data/scores_1.npy")
    scores_2 = np.load("Data/scores_2.npy")
    labels = np.load("Data/labels.npy")

    calibration_single_fold__only_training(scores_1, labels, f"sys1_useall", pi_weight)
    calibration_single_fold__only_training(scores_2, labels, f"sys2_useall", pi_weight)


def do_calibration_transformation_all() -> None:
    """ Use the calibration model created from the entirety of calibration subset and validation subset combined,
        and use it to transform only the scores of the new evaluation set"""
    eval_scores_1 = np.load("Data/eval_scores_1.npy")
    eval_scores_2 = np.load("Data/eval_scores_2.npy")

    pwLR_model_1 = LogisticRegressionParams.load_from_file("output_first/clbrtd_model_sys1_useall.pkl")
    pwLR_model_2 = LogisticRegressionParams.load_from_file("output_first/clbrtd_model_sys2_useall.pkl")

    calibration_single_fold__only_transform(pwLR_model_1, eval_scores_1, "eval1_useall")
    calibration_single_fold__only_transform(pwLR_model_2, eval_scores_2, "eval2_useall")


def do_evaluation_all(pi_eff: float) -> None:
    eval_scores_1 = np.load("Data/eval_scores_1.npy")
    eval_scores_2 = np.load("Data/eval_scores_2.npy")
    eval_labels = np.load("Data/eval_labels.npy")
    clbrtd_eval_scores_1 = np.load("output_first/clbrtd_scores_eval1_useall.npy")
    clbrtd_eval_scores_2 = np.load("output_first/clbrtd_scores_eval2_useall.npy")

    evaluate_DCF(eval_scores_1, clbrtd_eval_scores_1, eval_labels, "eval1_useall", pi_eff=pi_eff)
    evaluate_DCF(eval_scores_2, clbrtd_eval_scores_2, eval_labels, "eval2_useall", pi_eff=pi_eff)

    evaluate_BayesErrorPlots(eval_scores_1, clbrtd_eval_scores_1, eval_labels, "eval1_useall")
    evaluate_BayesErrorPlots(eval_scores_2, clbrtd_eval_scores_2, eval_labels, "eval2_useall")


def do_fusion_calibration_single_fold(pi_weight: float) -> None:
    scores_1 = np.load("Data/scores_1.npy")
    scores_2 = np.load("Data/scores_2.npy")
    scores = np.vstack([scores_1, scores_2])

    labels = np.load("Data/labels.npy")

    eval_scores_1 = np.load("Data/eval_scores_1.npy")
    eval_scores_2 = np.load("Data/eval_scores_2.npy")
    eval_scores = np.vstack([eval_scores_1, eval_scores_2])

    _, pwLR_model = calibration_single_fold(scores, labels, "fusion_singlefold", pi_weight=pi_weight)

    calibration_single_fold__only_transform(pwLR_model, eval_scores, "eval_fusion_singlefold")


def do_fusion_evaluation_single_fold(pi_eff: float) -> None:
    clbrtd_scores = [np.load("output_first/clbrtd_scores_1_singlefold.npy"),
                  np.load("output_first/clbrtd_scores_2_singlefold.npy")]
    labels = np.load("Data/labels.npy")
    labels_val = np.hstack([labels[1::3], labels[2::3]])

    clbrtd_eval_scores = [np.load("output_first/clbrtd_scores_e1_singlefold.npy"),
                   np.load("output_first/clbrtd_scores_e2_singlefold.npy")]
    eval_labels = np.load("Data/eval_labels.npy")

    evaluate_DCF_fusion(clbrtd_scores, np.load("output_first/clbrtd_scores_fusion_singlefold.npy"), labels_val,
                        "fusion_singlefold", pi_eff, ["sys1 (cal.)", "sys2 (cal.)"])
    evaluate_DCF_fusion(clbrtd_eval_scores, np.load("output_first/clbrtd_scores_eval_fusion_singlefold.npy"), eval_labels,
                        "eval_fusion_singlefold", pi_eff, ["sys1 (cal.)", "sys2 (cal.)"])

    evaluate_BayesErrorPlots_fusion(clbrtd_scores, np.load("output_first/clbrtd_scores_fusion_singlefold.npy"), labels_val,
                                    "fusion_singlefold", ["sys1 (cal.)", "sys2 (cal.)"])
    evaluate_BayesErrorPlots_fusion(clbrtd_eval_scores, np.load("output_first/clbrtd_scores_eval_fusion_singlefold.npy"), eval_labels,
                                    "eval_fusion_singlefold", ["sys1 (cal.)", "sys2 (cal.)"])


def do_fusion_calibration_kfold(K: int, pi_weight: float) -> None:
    """ Create a different calibration model for each fold, and use it on that fold.
        Then group the different calibrated folds scores together to have the whole calibrated scores. """
    scores_1 = np.load("Data/scores_1.npy")
    scores_2 = np.load("Data/scores_2.npy")
    scores = np.vstack([scores_1, scores_2])
    labels = np.load("Data/labels.npy")

    calibration_kfold(scores, labels, K, "fusion", pi_weight=pi_weight)


def do_fusion_evaluation_kfold(K: int, pi_eff: float) -> None:
    clbrtd_scores = [np.load(f"output_first/clbrtd_scores_sys1_{K}fold.npy"),
                  np.load(f"output_first/clbrtd_scores_sys2_{K}fold.npy")]
    labels = np.load("Data/labels.npy")
    fusion_scores = np.load("output_first/clbrtd_scores_fusion_5fold.npy")

    evaluate_DCF_fusion(clbrtd_scores, fusion_scores, labels,
                        f"fusion_{K}fold", pi_eff, ["sys1 (cal.)", "sys2 (cal.)"])
    evaluate_BayesErrorPlots_fusion(clbrtd_scores, fusion_scores, labels,
                                    f"fusion_{K}fold", ["sys1 (cal.)", "sys2 (cal.)"])


def do_fusion_calibration_training_all(pi_weight: float) -> None:
    """ Create calibration model from the entirety of calibration subset and validation subset combined """
    scores_1 = np.load("Data/scores_1.npy")
    scores_2 = np.load("Data/scores_2.npy")
    scores = np.vstack([scores_1, scores_2])
    labels = np.load("Data/labels.npy")

    calibration_single_fold__only_training(scores, labels, f"fusion_useall", pi_weight)


def do_fusion_calibration_transformation_all() -> None:
    """ Use the calibration model created from the entirety of calibration subset and validation subset combined,
        and use it to transform only the scores of the new evaluation set"""
    eval_scores_1 = np.load("Data/eval_scores_1.npy")
    eval_scores_2 = np.load("Data/eval_scores_2.npy")
    eval_scores = np.vstack([eval_scores_1, eval_scores_2])

    pwLR_model = LogisticRegressionParams.load_from_file("output_first/clbrtd_model_fusion_useall.pkl")
    calibration_single_fold__only_transform(pwLR_model, eval_scores, "eval_fusion_useall")


def do_fusion_evaluation_all(pi_eff: float) -> None:
    clbrtd_eval_scores = [np.load("output_first/clbrtd_scores_eval1_useall.npy"),
                          np.load("output_first/clbrtd_scores_eval2_useall.npy")]
    eval_labels = np.load("Data/eval_labels.npy")
    fusion_scores = np.load("output_first/clbrtd_scores_eval_fusion_useall.npy")

    evaluate_DCF_fusion(clbrtd_eval_scores, fusion_scores, eval_labels, "eval_fusion_useall", pi_eff,
                        ["eval1 (cal.)", "eval2 (cal.)"])

    evaluate_BayesErrorPlots_fusion(clbrtd_eval_scores, fusion_scores, eval_labels, "eval_fusion_useall",
                                    ["eval1 (cal.)", "eval2 (cal.)"])


if __name__ == '__main__':
    # """ 1) no calibration """
    # evaluate_no_calibration()
    #
    # """ 2) single-fold calibration """
    # # 1 - Calibrate
    # do_calibration_single_fold(pi_weight=0.2)
    # # 2 - Evaluate
    # do_evaluation_single_fold(pi_eff=0.2)
    #
    """ 3) K-fold calibration """
    # 1 - Calibrate
    do_calibration_kfold(K=5, pi_weight=0.2)
    # 2 - Evaluate
    do_evaluation_kfold(K=5, pi_eff=0.2)

    # """ 4) useall calibration """
    # # 1 - Calibrate
    # do_calibration_training_all(pi_weight=0.2)
    # do_calibration_transformation_all()
    # do_evaluation_all(pi_eff=0.2)

    """ 5) Fusion - 1 - singlefold (both on calibration validation subset AND evaluation set),
                    2 - Kfold (on calibration k-folded), 3 - useall (on evaluation) """
    # 1 - singlefold
    do_fusion_calibration_single_fold(pi_weight=0.2)
    do_fusion_evaluation_single_fold(pi_eff=0.2)

    # 2 - Kfold
    do_fusion_calibration_kfold(K=5, pi_weight=0.2)
    do_fusion_evaluation_kfold(K=5, pi_eff=0.2)

    # 3 - useall
    do_fusion_calibration_training_all(pi_weight=0.2)
    do_fusion_calibration_transformation_all()
    do_fusion_evaluation_all(pi_eff=0.2)
