import numpy as np
import pandas as pd
from prettytable import PrettyTable
from tqdm import tqdm

from modules.load_datasets import load_iris_binary, split_db_2to1
from modules.statistics import how_many_features, how_many_samples
from modules.svm import run_primal, run_dual, run_dual_with_kernel, poly_kernel, rbf_kernel, \
    expand_data_with_bias_regterm, get_kern_H_with_loops


def SVM_no_kernel(D_simple, L):
    (D_tr_simple, L_tr), (D_val_simple, L_val) = split_db_2to1(D_simple, L)

    results = []
    for K in [1, 10]:
        D_tr = expand_data_with_bias_regterm(D_tr_simple, K)
        D_val = expand_data_with_bias_regterm(D_val_simple, K)

        for C in [0.1, 1.0, 10.0]:
            dual_loss, error_rate, dcf_min, dcf_norm = run_dual(D_tr, L_tr, D_val, L_val, C)
            primal_loss, _, _, _ = run_primal(D_tr, L_tr, D_val, L_val, C)
            duality_gap = primal_loss - dual_loss

            results.append([K, C, primal_loss, dual_loss, duality_gap, error_rate, dcf_min, dcf_norm])

    df = pd.DataFrame(results,
                      columns=['K', 'C', 'Primal loss', 'Dual loss', 'Duality gap', 'Error rate', 'minDCF (π_T = 0.5)',
                               'actDCF (π_T = 0.5)'])

    # Format the losses and the duality gap in scientific notation
    df['Primal loss'] = df['Primal loss'].apply(lambda x: f'{x:.6e}')
    df['Dual loss'] = df['Dual loss'].apply(lambda x: f'{x:.6e}')
    df['Duality gap'] = df['Duality gap'].apply(lambda x: f'{x:.6e}')
    # Format the Error rate, minDCF and actDCF as required
    df['Error rate'] = (df['Error rate'] * 100).apply(lambda x: f'{x:.1f}%')
    df['minDCF (π_T = 0.5)'] = df['minDCF (π_T = 0.5)'].apply(lambda x: f'{x:.4f}')
    df['actDCF (π_T = 0.5)'] = df['actDCF (π_T = 0.5)'].apply(lambda x: f'{x:.4f}')

    table = PrettyTable()
    table.field_names = df.columns.tolist()

    for index, row in df.iterrows():
        table.add_row(row.tolist())

    # Print the table
    print(table)


def SVM_kernel(D, L, verbose=True):
    (D_tr, L_tr), (D_val, L_val) = split_db_2to1(D, L)

    results = []

    # Other params
    C = 1.0

    # Poly kernel parameters: (c, d, K)
    poly_kernel_params_list = [(0.0, 2.0, 0.0),
                               (0.0, 2.0, 1.0),
                               (1.0, 2.0, 0.0),
                               (1.0, 2.0, 1.0)]
    for poly_kernel_params in poly_kernel_params_list:
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

        loss, error_rate, dcf_min, dcf_norm = run_dual_with_kernel(D_tr, L_tr, D_val, L_val, C, H, score_kernel_matrix)
        results.append([poly_kernel_params[2], 1.0, f"Poly (d={poly_kernel_params[1]}, c={poly_kernel_params[0]})",
                        loss, error_rate, dcf_min, dcf_norm])

    # RBF kernel parameters: (gamma, K)
    rbf_kernel_params = [(1.0, 0.0),
                         (1.0, 1.0),
                         (10.0, 0.0),
                         (10.0, 1.0)]
    for rbf_kernel_params in rbf_kernel_params:
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

        loss, error_rate, dcf_min, dcf_norm = run_dual_with_kernel(D_tr, L_tr, D_val, L_val, C, H, score_kernel_matrix)
        results.append([rbf_kernel_params[1], 1.0, f"RBF (γ={rbf_kernel_params[0]})",
                        loss, error_rate, dcf_min, dcf_norm])

    # Convert results to a DataFrame
    df = pd.DataFrame(results, columns=['K', 'C', 'Kernel', 'Dual loss', 'Error rate', 'minDCF (π_T = 0.5)',
                                        'actDCF (π_T = 0.5)'])

    # Format the losses and error rate columns
    df['Dual loss'] = df['Dual loss'].apply(lambda x: f'{x:.6e}')
    df['Error rate'] = df['Error rate'].apply(lambda x: f'{x:.1%}')
    df['minDCF (π_T = 0.5)'] = df['minDCF (π_T = 0.5)'].apply(lambda x: f'{x:.4f}')
    df['actDCF (π_T = 0.5)'] = df['actDCF (π_T = 0.5)'].apply(lambda x: f'{x:.4f}')

    # Create a PrettyTable
    table = PrettyTable()
    table.field_names = df.columns.tolist()

    for index, row in df.iterrows():
        table.add_row(row.tolist())

    # Print the table
    print(table)

if __name__ == '__main__':
    D, L = load_iris_binary()
    (D_tr, L_tr), (D_val, L_val) = split_db_2to1(D, L)

    # # First part of the lab: training and evaluating SVM with both Dual and Primal formulations
    # SVM_no_kernel(D, L)

    # # Second part of the lab: now using (on Dual) kernels - Polynomial (d=2) and RBF
    SVM_kernel(D, L)




