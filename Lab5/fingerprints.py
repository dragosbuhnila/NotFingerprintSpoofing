from modules.load_datasets import load_fingerprints, split_db_2to1
from modules.projections import get_PCA_projection_matrix
from modules.statistics import print_dataset_info, get_unique_classes
from modules.mvg_classification import *
import numpy as np

np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x))
np.set_printoptions(precision=4, suppress=True)


if __name__ == "__main__":
    D, L = load_fingerprints()
    unique_classes = get_unique_classes(L)
    priors = [1/2 for _ in unique_classes]

    # Splitting Dataset
    (D_tr, L_tr), (D_val, L_val) = split_db_2to1(D, L)
    print_dataset_info(D_tr, L_tr, "fingerprints - training")
    print_dataset_info(D_val, L_val, "fingerprints - validation")

    print("=========================================================================================")
    print("====================== Now classifying with MVG, Tied, Naive, LDA =======================")
    print("=========================================================================================")

    print("Classifying with MVG ")
    logscore_MVG, _, _, _ = extract_densities_with_MVG(D_tr, L_tr, D_val)
    classify_two_classes(logscore_MVG, L_val)

    print("-----------------------------------------")
    print("Classifying with Tied ")
    logscore_Tied, _, _, _ = extract_densities_with_tied(D_tr, L_tr, D_val)
    classify_two_classes(logscore_Tied, L_val)

    print("-----------------------------------------")
    print("Classifying with Naive ")
    logscore_Naive, _, _, _ = extract_densities_with_naive(D_tr, L_tr, D_val)
    classify_two_classes(logscore_Naive, L_val)

    print("-----------------------------------------")
    print("Classifying with LDA ")
    classify_over_LDA(D_tr, L_tr, D_val, L_val)

    print("\n")
    print("=========================================================================================")
    print("====================== Now analyzing Covariance and Correlation =========================")
    print("=========================================================================================")

    cov_matrices = get_covariance_matrix_by_class(D_tr, L_tr)

    for class_x in unique_classes:
        print(f"Covariance matrix of class {class_x} is:")
        print(cov_matrices[class_x])
        print("---------------------------------------------")

        corr_matrix = get_correlation_matrix(cov_matrices[class_x])
        print(f"Correlation matrix of class {class_x} is:")
        print(corr_matrix)
        print("---------------------------------------------")

    print("\n")
    print("=========================================================================================")
    print("========================= Now classifying using only 4 features =========================")
    print("=========================================================================================")

    (D_tr, L_tr), (D_val, L_val) = split_db_2to1(D[:4, :], L)
    print_dataset_info(D_tr, L_tr, "fingerprints - training")
    print_dataset_info(D_val, L_val, "fingerprints - validation")

    print("-----------------------------------------")
    print("Classifying with MVG ")
    logscore_MVG, _, _, _ = extract_densities_with_MVG(D_tr, L_tr, D_val)
    classify_two_classes(logscore_MVG, L_val)

    print("-----------------------------------------")
    print("Classifying with Tied ")
    logscore_Tied, _, _, _ = extract_densities_with_tied(D_tr, L_tr, D_val)
    classify_two_classes(logscore_Tied, L_val)

    print("-----------------------------------------")
    print("Classifying with Naive ")
    logscore_Naive, _, _, _ = extract_densities_with_naive(D_tr, L_tr, D_val)
    classify_two_classes(logscore_Naive, L_val)

    print("-----------------------------------------")
    print("Classifying with LDA ")
    classify_over_LDA(D_tr, L_tr, D_val, L_val)

    print("\n")
    print("=========================================================================================")
    print("========================= Now classifying using only features 0-1 =======================")
    print("=========================================================================================")

    (D_tr, L_tr), (D_val, L_val) = split_db_2to1(D[:2, :], L)
    print_dataset_info(D_tr, L_tr, "fingerprints - training")
    print_dataset_info(D_val, L_val, "fingerprints - validation")

    print("-----------------------------------------")
    print("Classifying with MVG ")
    logscore_MVG, _, _, _ = extract_densities_with_MVG(D_tr, L_tr, D_val)
    classify_two_classes(logscore_MVG, L_val)

    print("-----------------------------------------")
    print("Classifying with Tied ")
    logscore_Tied, _, _, _ = extract_densities_with_tied(D_tr, L_tr, D_val)
    classify_two_classes(logscore_Tied, L_val)

    print("\n")
    print("=========================================================================================")
    print("==== Classifying features 0-1 but on training set to see if overfits or is just bad =====")
    print("=========================================================================================")

    (D_tr, L_tr), (D_val, L_val) = split_db_2to1(D[:2, :], L)
    print_dataset_info(D_tr, L_tr, "fingerprints - training")
    print_dataset_info(D_val, L_val, "fingerprints - validation")

    print("-----------------------------------------")
    print("Classifying with MVG ")
    logscore_MVG, _, _, _ = extract_densities_with_MVG(D_tr, L_tr, D_tr)
    classify_two_classes(logscore_MVG, L_tr)

    print("-----------------------------------------")
    print("Classifying with Tied ")
    logscore_Tied, _, _, _ = extract_densities_with_tied(D_tr, L_tr, D_tr)
    classify_two_classes(logscore_Tied, L_tr)

    print("\n")
    print("=========================================================================================")
    print("========================= Now classifying using only features 2-3 =======================")
    print("=========================================================================================")

    (D_tr, L_tr), (D_val, L_val) = split_db_2to1(D[2:4, :], L)
    print_dataset_info(D_tr, L_tr, "fingerprints - training")
    print_dataset_info(D_val, L_val, "fingerprints - validation")

    print("-----------------------------------------")
    print("Classifying with MVG ")
    logscore_MVG, _, _, _ = extract_densities_with_MVG(D_tr, L_tr, D_val)
    classify_two_classes(logscore_MVG, L_val)

    print("-----------------------------------------")
    print("Classifying with Tied ")
    logscore_Tied, _, _, _ = extract_densities_with_tied(D_tr, L_tr, D_val)
    classify_two_classes(logscore_Tied, L_val)

    print("\n")
    print("=========================================================================================")
    print("==================================== PCA pre-processing =================================")
    print("=========================================================================================")

    (D_tr, L_tr), (D_val, L_val) = split_db_2to1(D, L)
    print_dataset_info(D_tr, L_tr, "fingerprints - training")
    print_dataset_info(D_val, L_val, "fingerprints - validation")

    for m_PCA in range(1, how_many_features(D_tr) + 1):
        print("==========================================")
        print(f"PCA reducing to {m_PCA} dimensions:")

        P = get_PCA_projection_matrix(D_tr, m_PCA)
        D_tr_P = P.T @ D_tr
        D_val_P = P.T @ D_val

        print("-----------------------------------------")
        print("Classifying with MVG")
        logscore_MVG, _, _, _ = extract_densities_with_MVG(D_tr_P, L_tr, D_val_P)
        classify_two_classes(logscore_MVG, L_val)

        print("-----------------------------------------")
        print("Classifying with Tied")
        logscore_Tied, _, _, _ = extract_densities_with_tied(D_tr_P, L_tr, D_val_P)
        classify_two_classes(logscore_Tied, L_val)

        print("-----------------------------------------")
        print("Classifying with Naive")
        logscore_Naive, _, _, _ = extract_densities_with_naive(D_tr_P, L_tr, D_val_P)
        classify_two_classes(logscore_Naive, L_val)

