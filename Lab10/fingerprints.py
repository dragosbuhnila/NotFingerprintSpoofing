import numpy as np
import pandas as pd
from prettytable import PrettyTable
from tqdm import tqdm

from modules.GMM import logpdf_GMM, LBG, GMMParameters
from modules.evaluation import get_min_normalized_dcf_binary, get_dcf__includes_classification, \
    get_normalized_dcf_binary
from modules.load_datasets import split_db_2to1, shrink_dataset, load_fingerprints
from modules.statistics import print_dataset_info, how_many_classes, how_many_samples


def do_training(D_tr, L_tr, total_LBG_iterations, variant):
    if variant not in ["std", "diag", "tied"]:
        raise ValueError("Invalid variant. Choose one of: std, diag, tied.")

    all_params_byclass = {"std": [],
                          "diag": [],
                          "tied": []}

    # Training
    for c in tqdm(range(how_many_classes(L)), desc=f"Training {variant} GMMs..."):
        all_params_byclass[variant].append(LBG(D_tr[:, L_tr == c], total_LBG_iterations, variant=variant))

    if variant == "diag":
        models_class0 = all_params_byclass[variant][0]
        models_class1 = all_params_byclass[variant][1]
        for (i, params_list) in enumerate(models_class0):
            if i == 0:
                continue
            GMM_model = GMMParameters(components=params_list)
            GMM_model.save_to_file(f"output_first/model_FalseClass_{2**i}-cGMM_diag.pkl")
            print(f"[[Saved model_FalseClass_{2**i}-cGMM_diag.pkl]]")
        for (i, params_list) in enumerate(models_class1):
            if i == 0:
                continue
            GMM_model = GMMParameters(components=params_list)
            GMM_model.save_to_file(f"output_first/model_TrueClass_{2**i}-cGMM_diag.pkl")
            print(f"[[Saved model_TrueClass_{2**i}-cGMM_diag.pkl]]")

    return all_params_byclass


def do_evaluation(D_val, L_val, all_params_byclass, total_LBG_iterations, variant, pi):
    if variant not in ["std", "diag", "tied"]:
        raise ValueError("Invalid variant. Choose one of: std, diag, tied.")

    DCFs = [variant]

    for lbg_num in tqdm(range(0, total_LBG_iterations + 1), desc=f"Evaluating {variant} GMMs..."):
        # 1) Compute log-likelihoods class by class
        loglikelihoods_by_class = np.zeros((how_many_classes(L), how_many_samples(D_val)))
        for (params_by_class, class_x) in zip(all_params_byclass[variant], range(how_many_classes(L))):
            loglikelihoods_of_class_x, _ = logpdf_GMM(D_val, params_by_class[lbg_num])
            loglikelihoods_by_class[class_x, :] = loglikelihoods_of_class_x

        # 2) Compute the log joint likelihoods, again class by class
        priors = np.array([1 / how_many_classes(L) for _ in range(how_many_classes(L))])
        log_joints_by_class = loglikelihoods_by_class
        for i in range(len(priors)):
            log_joints_by_class[i, :] = loglikelihoods_by_class[i, :] + np.log(priors[i])

        # 3) Classify
        llrs = log_joints_by_class[1, :] - log_joints_by_class[0, :]

        # if lbg_num == 4:
        #     np.save(f'llrs_GMM_g{2**lbg_num}_{variant}.npy', llrs)
        #     print(f"[[Saved ./llrs_GMM_g{2**lbg_num}_{variant}.npy]]")

        # 4) DCFs
        triplet = (pi, 1, 1)
        dcf, _, _ = get_dcf__includes_classification(llrs, L_val, *triplet)
        dcf_norm = get_normalized_dcf_binary(*triplet, dcf=dcf)
        dcf_min = get_min_normalized_dcf_binary(llrs, L_val, *triplet)

        DCFs.append(str(round(dcf_min, 4)) + " / " + str(round(dcf_norm, 4)))

    return DCFs


if __name__ == '__main__':
    """ Data Preparation """
    # Load the dataset
    D, L = load_fingerprints()

    # # Reduce the dataset
    # D, L = shrink_dataset(D, L, 10)

    (D_tr, L_tr), (D_val, L_val) = split_db_2to1(D, L)
    # print_dataset_info(D, L, "Total")
    # print_dataset_info(D_tr, L_tr, "Training")
    # print_dataset_info(D_val, L_val, "Validation")

    """ Training """
    total_LBG_iterations = 3
    pi = 0.1

    # all_params_byclass = do_training(D_tr, L_tr, total_LBG_iterations, "std")
    # DCFs_std = do_evaluation(D_val, L_val, all_params_byclass, total_LBG_iterations, "std", pi)

    # all_params_byclass = do_training(D_tr, L_tr, total_LBG_iterations, "diag")
    DCFs_diag = do_evaluation(D_val, L_val, all_params_byclass, total_LBG_iterations, "diag", pi)

    # # Prepare Table
    # columns = ["#GMMcomponents"]
    # for i in range(total_LBG_iterations + 1):
    #     columns.append(str(2 ** i))
    #
    # df = pd.DataFrame([DCFs_std, DCFs_diag], columns=columns)
    #
    # table = PrettyTable()
    # table.field_names = df.columns.tolist()
    #
    # for index, row in df.iterrows():
    #     table.add_row(row.tolist())
    #
    # print(table)
