import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from modules.common_matrix_operations import vcol
from modules.load_datasets import load_fingerprints, split_db_2to1, shrink_dataset
from modules.statistics import print_dataset_info, how_many_features, how_many_samples
from modules.svm import run_dual, expand_data_with_bias_regterm, plot_dcfs, run_dual_with_kernel, poly_kernel, \
    rbf_kernel, get_kern_H, get_kern_H_with_loops, run_dual_with_kernel__only_train


# Suppress displaying plots
def nop():
    pass
plt.show = nop

# Print wider lines on console
np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.3g" % x))
np.set_printoptions(precision=4, suppress=True)


def linear_SVM(D_tr, L_tr, D_val, L_val):
    # 1) Linear SVM
    # Params
    K = 1.0
    Cs = np.logspace(-5, 0, 11)
    pi = 0.1

    # Preparing plot dimensions
    min_dcfs = []
    act_dcfs = []

    # Expanding data
    D_tr_expanded = expand_data_with_bias_regterm(D_tr, K)
    D_val_expanded = expand_data_with_bias_regterm(D_val, K)

    # Actual training and validation
    for C in Cs:
        _, _, dcf_min, dcf_act = run_dual(D_tr_expanded, L_tr, D_val_expanded, L_val, C, pi=pi)

        min_dcfs.append(dcf_min)
        act_dcfs.append(dcf_act)

    plot_dcfs(Cs, act_dcfs, min_dcfs, "Fingerprints - Linear SVM")
    print("Linear SVM")
    print(Cs)
    print(act_dcfs)
    print(min_dcfs)
    print()


def linear_SVM_with_precentering(D_tr, L_tr, D_val, L_val):
    # 2) Linear SVM with pre-centering
    # Center the data
    dataset_mean = D_tr.mean(1)
    D_tr_c = D_tr - vcol(dataset_mean)
    D_val_c = D_val - vcol(dataset_mean)

    # Expanding data
    D_tr_c_expanded = expand_data_with_bias_regterm(D_tr_c, K)
    D_val_c_expanded = expand_data_with_bias_regterm(D_val_c, K)

    # Params
    K = 1.0
    Cs = np.logspace(-5, 0, 11)
    pi = 0.1

    # Repreparing plot dimensions
    min_dcfs = []
    act_dcfs = []

    for C in Cs:
        _, _, dcf_min, dcf_act = run_dual(D_tr_c_expanded, L_tr, D_val_c_expanded, L_val, C, pi=pi)

        min_dcfs.append(dcf_min)
        act_dcfs.append(dcf_act)

    plot_dcfs(Cs, act_dcfs, min_dcfs, "Fingerprints - Linear SVM with precentering")
    print("Linear SVM with precentering")
    print(Cs)
    print(act_dcfs)
    print(min_dcfs)
    print()


def poly_kernel_SVM(D_tr, L_tr, D_val, L_val, verbose=True):
    # 3) Poly-kernel     (c=1, d=2, ξ=0) SVM
    if verbose:
        print("Poly-kernel SVM entered")
    # Params
    poly_kernel_params = (1.0, 2.0, 0.0)
    Cs = np.logspace(-5, 0, 11)
    pi = 0.1

    # Repreparing plot dimensions
    min_dcfs = []
    act_dcfs = []

    # Precalculate the H matrix with kernel
    if verbose:
        print("Computing H matrix for RBF-kernel SVM with params: ", poly_kernel_params)
    z = 2 * L_tr - 1
    H = get_kern_H_with_loops(D_tr, z, poly_kernel, poly_kernel_params)

    # Precalculate some values of the kernel. Let's call this score_kernel_matrix
    score_kernel_matrix = np.zeros((how_many_samples(D_tr), how_many_samples(D_val)))
    for j in tqdm(range(0, how_many_samples(D_val)), desc="Loading..."):
        for i in range(0, how_many_samples(D_tr)):
            score_kernel_matrix[i, j] = poly_kernel(D_tr[:, i], D_val[:, j], *poly_kernel_params)

    i = 1
    for C in Cs:
        _, _, dcf_min, dcf_act = run_dual_with_kernel(D_tr, L_tr, D_val, L_val, C, H, score_kernel_matrix, pi=pi)
        min_dcfs.append(dcf_min)
        act_dcfs.append(dcf_act)
        if verbose:
            print(f"Computed DCFs for Poly-kernel SVM - C#{i} = {C}")
            i += 1

    plot_dcfs(Cs, act_dcfs, min_dcfs, "Fingerprints - Poly(c=1, d=2, ξ=0) SVM")
    print("Poly(c=1, d=2, ξ=0) SVM")
    print(Cs)
    print(act_dcfs)
    print(min_dcfs)
    print()


def rbf_kernel_SVM(D_tr, L_tr, D_val, L_val, verbose=True):
    # 4) RBF-kernel     (γ=..., ξ=1) SVM
    # Params
    Cs = np.logspace(-3, 2, 11)
    gammas = [np.exp(-4), np.exp(-3), np.exp(-2), np.exp(-1)]  # γ
    K = 1  # ξ = K per 0 e 1
    pi = 0.1

    # Prepare the figure
    plt.figure()
    plt.title("Fingerprints - RBF(γ in [e-4, e-3, e-2, e-1], ξ=1) SVM")
    plt.xlabel('C')
    plt.ylabel('DCFs')
    plt.xscale('log', base=10)

    # Actual training and validation
    gamma_index = 4
    print("RBF")
    print(Cs)
    for gamma in gammas:
        # Other Params
        rbf_kernel_params = (gamma, K)

        # Precalculate the H matrix with kernel
        if verbose:
            print("Computing H matrix for RBF-kernel SVM with params: ", rbf_kernel_params)
        z = 2 * L_tr - 1
        H = get_kern_H_with_loops(D_tr, z, rbf_kernel, rbf_kernel_params)

        # Precalculate some values of the kernel. Let's call this score_kernel_matrix
        score_kernel_matrix = np.zeros((how_many_samples(D_tr), how_many_samples(D_val)))
        for j in tqdm(range(0, how_many_samples(D_val)), desc="Loading..."):
            for i in range(0, how_many_samples(D_tr)):
                score_kernel_matrix[i, j] = rbf_kernel(D_tr[:, i], D_val[:, j], *rbf_kernel_params)


        # Repreparing plot dimensions
        min_dcfs = []
        act_dcfs = []
        iii = 1
        for C in Cs:
            _, _, dcf_min, dcf_act, scores = run_dual_with_kernel(D_tr, L_tr, D_val, L_val, C, H, score_kernel_matrix, pi=pi)

            min_dcfs.append(dcf_min)
            act_dcfs.append(dcf_act)
            if verbose:
                print(f"Computed DCFs for Poly-kernel SVM - C#{iii} = {C}")
                iii += 1

            np.save(f'llrs_SVM_shrink10_C{C}_g{gamma}.npy', scores)
            print(f"[[Saved ./llrs_SVM_shrink10_C{C}_g{gamma}.npy]]")

        # Plot DCFs for current gamma (2 out of 8)
        plt.plot(Cs, act_dcfs, label=f'actDCF γ=e-{gamma_index}')
        plt.plot(Cs, min_dcfs, label=f'minDCF γ=e-{gamma_index}')
        gamma_index -= 1

        print("γ=e-" + str(gamma_index + 1))
        print(act_dcfs)
        print(min_dcfs)
        print()

    plt.legend()
    plt.savefig("Fingerprints - RBF(γ in [e-4, e-3, e-2, e-1], ξ=1) SVM")
    print(f"[[Saved Fingerprints - RBF(γ in [e-4, e-3, e-2, e-1], ξ=1) SVM]]")
    plt.show()


if __name__ == '__main__':
    # Load the dataset
    D, L = load_fingerprints()

    # # Reduce the dataset
    # D, L = shrink_dataset(D, L, 10)

    (D_tr, L_tr), (D_val, L_val) = split_db_2to1(D, L)
    # print_dataset_info(D, L, "Total")
    # print_dataset_info(D_tr, L_tr, "Training")
    # print_dataset_info(D_val, L_val, "Validation")


    # poly_kernel_SVM(D_tr, L_tr, D_val, L_val)

    # rbf_kernel_SVM(D_tr, L_tr, D_val, L_val)

    z = 2 * L_tr - 1
    #                                                gamma    , K or Xi
    H = get_kern_H_with_loops(D_tr, z, rbf_kernel, (np.exp(-2), 1))
    run_dual_with_kernel__only_train(D_tr, L_tr, D_val, L_val, C=31, H=H)
    # run_dual_with_kernel__only_classify()
    # np.save(f"output_first/scores_rbfSVM.npy", scores)
    # print(f"[[Saved scores_rbfSVM.npy]]")
