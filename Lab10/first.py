import pandas as pd
import scipy
from prettytable import PrettyTable
from tqdm import tqdm

from Data.GMM_load import load_gmm
from modules.GMM import logpdf_GMM, EMM, LBG
from modules.common_matrix_operations import to_numpy_matrix, vcol, vrow
import numpy as np

from modules.evaluation import get_dcf__includes_classification, get_normalized_dcf_binary, \
    get_min_normalized_dcf_binary
from modules.load_datasets import load_iris, split_db_2to1
from modules.statistics import how_many_samples, how_many_classes, print_dataset_info


#
# # Print wider lines on console
# np.set_printoptions(edgeitems=30, linewidth=100000,
#     formatter=dict(float=lambda x: "%.3g" % x))
# np.set_printoptions(precision=4, suppress=True)


def do_logpdf_GMM_test():
    D = np.load("Data/GMM_data_4D.npy")
    GMM_params_init = load_gmm("Data/GMM_4D_3G_init.json")

    logscore_marginals_expected = np.load("Data/GMM_4D_3G_init_ll.npy")

    print("D shape: " + str(D.shape))
    print("GMM:")
    for i, el in enumerate(GMM_params_init):
        print(f"Component {i}:")
        for thing in el:
            print(thing)
        print()

    print("GMM:")
    for i, el in enumerate(GMM_params_init):
        print(f"Component {i}:")
        for thing in el:
            print("X", end="")
        print()

    print("logscore_marginals_expected shape: " + str(logscore_marginals_expected.shape))
    print("==================================")

    logscore_marginals = logpdf_GMM(D, GMM_params_init)
    print("Difference between my logscore_posterior results and the expected one: ")
    print(np.abs(logscore_marginals - logscore_marginals_expected).max())


def do_EMM_test():
    """ EMM """
    D = np.load("Data/GMM_data_4D.npy")
    GMM_params_init = load_gmm("Data/GMM_4D_3G_init.json")

    GMM_params_expected = load_gmm("Data/GMM_4D_3G_EM.json")
    GMM_params_opt = EMM(D, GMM_params_init, 1e-6)

    # What we call marginal in terms of cluster posterior probability
    # is actually the likelihood in terms of GMM itself
    logscore_marginals_expected, _ = logpdf_GMM(D, GMM_params_expected)
    logscore_marginals_opt, _ = logpdf_GMM(D, GMM_params_opt)

    avg_likelihood_expected = np.mean(logscore_marginals_expected)
    avg_likelihood_opt = np.mean(logscore_marginals_opt)

    print("Expected average likelihood: " + str(avg_likelihood_expected))
    print("Optimized average likelihood: " + str(avg_likelihood_opt))


def do_LBG_test():
    """ LBG 1->2 clusters """
    GMM_params_expected = load_gmm("Data/GMM_4D_4G_EM_LBG.json")
    all_GMM_params_opt = LBG(D, 1e-6, 2, variant="tied")

    logscore_marginals_expected, _ = logpdf_GMM(D, GMM_params_expected)
    logscore_marginals_opt, _ = logpdf_GMM(D, all_GMM_params_opt[1])

    avg_likelihood_expected = np.mean(logscore_marginals_expected)
    avg_likelihood_opt = np.mean(logscore_marginals_opt)

    print("Expected average likelihood: " + str(avg_likelihood_expected))
    print("Optimized average likelihood: " + str(avg_likelihood_opt))


def do_LBG_training_and_classification(D, L):
    """ LBG 1->16 clusters """
    (D_tr, L_tr), (D_val, L_val) = split_db_2to1(D, L)
    # print_dataset_info(D, L, "Total")
    # print_dataset_info(D_tr, L_tr, "Training")
    # print_dataset_info(D_val, L_val, "Validation")

    all_params_byclass = {"std": [],
                          "diag": [],
                          "tied": []}

    total_LBG_iterations = 4
    # Training
    for c in tqdm(range(how_many_classes(L)), desc="Training GMMs..."):
        all_params_byclass["std"].append(LBG(D_tr[:, L_tr == c], total_LBG_iterations, verbose=True))
        all_params_byclass["diag"].append(LBG(D_tr[:, L_tr == c], total_LBG_iterations, variant="diag", verbose=True))
        all_params_byclass["tied"].append(LBG(D_tr[:, L_tr == c], total_LBG_iterations, variant="tied", verbose=True))

    # Classification using std GMM
    error_rates = {"std": ["Full Covariance (standard)"],
                   "diag": ["Diagonal Covariance"],
                   "tied": ["Tied Covariance"]}
    for variant in tqdm(["std", "diag", "tied"], desc="Evaluating GMMs variants..."):
        for lbg_num in range(0, total_LBG_iterations + 1):
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

            # 3) Compute the marginal likelihoods
            log_marginals = vrow(scipy.special.logsumexp(log_joints_by_class, axis=0))

            # 4) Compute the posteriors
            if log_joints_by_class.shape[1] != log_marginals.shape[1]:
                raise ValueError(f"Something went wrong with the shapes of the log-joints and the log-marginals: "
                                 f"log_joints_by_class.shape[1] = {log_joints_by_class.shape[1]}, "
                                 f"log_marginals.shape[1] = {log_marginals.shape[1]}")
            log_posteriors_by_class = log_joints_by_class - vrow(log_marginals)

            # 5) Classify
            predictions = np.argmax(log_posteriors_by_class, axis=0)

            # 6) Error rate
            nof_errors = (predictions != L_val).sum()
            error_rate = nof_errors / len(L_val)
            error_rate = str(round(error_rate, 4) * 100) + "%"
            error_rates[variant].append(error_rate)

    # Print Table
    df = pd.DataFrame(error_rates.values(), columns=["GMM Type", "1", "2", "4", "8", "16"])

    table = PrettyTable()

    # Set the field names
    table.field_names = df.columns.tolist()

    # Add rows to the table
    for index, row in df.iterrows():
        table.add_row(row.tolist())

    # Print the table
    print(table)


def do_LBG_training_and_classification_DCFs():
    """ LBG DCF evaluation, again 1->16 clusters """
    D = np.load("Data/ext_data_binary.npy")
    L = np.load("Data/ext_data_binary_labels.npy")

    (D_tr, L_tr), (D_val, L_val) = split_db_2to1(D, L)
    print_dataset_info(D, L, "Total")
    print_dataset_info(D_tr, L_tr, "Training")
    print_dataset_info(D_val, L_val, "Validation")

    all_params_byclass = {"std": [],
                          "diag": [],
                          "tied": []}

    total_LBG_iterations = 4
    # Training
    for c in tqdm(range(how_many_classes(L)), desc="Training GMMs..."):
        all_params_byclass["std"].append(LBG(D_tr[:, L_tr == c], total_LBG_iterations, verbose=True))
        all_params_byclass["diag"].append(LBG(D_tr[:, L_tr == c], total_LBG_iterations, variant="diag", verbose=True))
        all_params_byclass["tied"].append(LBG(D_tr[:, L_tr == c], total_LBG_iterations, variant="tied", verbose=True))

    # Classification using std GMM
    DCFs = {"std": ["Full Covariance"],
            "diag": ["Diagonal Covariance"],
            "tied": ["Tied Covariance"]}
    for variant in tqdm(["std", "diag", "tied"], desc="Evaluating GMMs variants..."):
        for lbg_num in range(0, total_LBG_iterations + 1):
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
            predictions = np.zeros_like(L_val)[llrs > 0] = 1

            # 4) DCFs
            triplet = (0.5, 1, 1)  # Could very well use an effective prior, but I don't really want to modify the functions
            dcf, _, _ = get_dcf__includes_classification(llrs, L_val, *triplet)
            dcf_norm = get_normalized_dcf_binary(*triplet, dcf=dcf)
            dcf_min = get_min_normalized_dcf_binary(llrs, L_val, *triplet)

            DCFs[variant].append(str(dcf_min) + " / " + str(dcf_norm))

    # Prepare Table
    df = pd.DataFrame(DCFs.values(), columns=["GMM Type", "1", "2", "4", "8", "16"])

    table = PrettyTable()
    table.field_names = df.columns.tolist()

    for index, row in df.iterrows():
        table.add_row(row.tolist())

    # Print the Table
    print(table)


if __name__ == '__main__':
    # # There are other functions

    # """ LBG 1->16 clusters """
    # D, L = load_iris()
    # do_LBG_training_and_classification(D, L)

    do_logpdf_GMM_test()

    """ LBG DCF evaluation, again 1->16 clusters """
    # do_LBG_training_and_classification_DCFs()




