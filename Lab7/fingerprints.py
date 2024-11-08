import numpy as np

from modules.evaluation import get_dcf__includes_classification, get_normalized_dcf_binary, get_effective_prior, \
    get_min_normalized_dcf_binary
from modules.load_datasets import load_fingerprints, split_db_2to1
from modules.mvg_classification import extract_densities_with_MVG, extract_densities_with_naive, \
    extract_densities_with_tied
from modules.plottings import plot_bayes_error_plots
from modules.probability_first import get_llrs
from modules.projections import get_PCA_projection_matrix


def eval_noPCA_noEffectivePI(D_tr, L_tr, D_val, L_val):
    #            pi   Cfn  Cfp
    triplets = [(0.5, 1.0, 1.0), (0.9, 1.0, 1.0), (0.1, 1.0, 1.0), (0.5, 1.0, 9.0), (0.5, 9.0, 1.0)]

    print(
        "  π   Cfn  Cfp  fπ  |  MVG    MVGmin    margin  |  naive    naiveMin    margin  |  tied    tiedMin    margin")
    for triplet in triplets:
        logscore, _, _, _ = extract_densities_with_MVG(D_tr, L_tr, D_val)
        llrs_MVG = get_llrs(logscore)
        logscore, _, _, _ = extract_densities_with_naive(D_tr, L_tr, D_val)
        llrs_naive = get_llrs(logscore)
        logscore, _, _, _ = extract_densities_with_tied(D_tr, L_tr, D_val)
        llrs_tied = get_llrs(logscore)

        dcf_MVG, _, _ = get_dcf__includes_classification(llrs_MVG, L_val, *triplet)
        dcf_MVG_norm = get_normalized_dcf_binary(*triplet, dcf=dcf_MVG)
        dcf_MVG_min = get_min_normalized_dcf_binary(llrs_MVG, L_val, *triplet)

        dcf_naive, _, _ = get_dcf__includes_classification(llrs_naive, L_val, *triplet)
        dcf_naive_norm = get_normalized_dcf_binary(*triplet, dcf=dcf_naive)
        dcf_naive_min = get_min_normalized_dcf_binary(llrs_naive, L_val, *triplet)

        dcf_tied, _, _ = get_dcf__includes_classification(llrs_tied, L_val, *triplet)
        dcf_tied_norm = get_normalized_dcf_binary(*triplet, dcf=dcf_tied)
        dcf_tied_min = get_min_normalized_dcf_binary(llrs_tied, L_val, *triplet)

        print(f"{triplet}->{get_effective_prior(*triplet)}   "
              f"{dcf_MVG_norm:.3f}  {dcf_MVG_min:.3f}     {(dcf_MVG_norm / dcf_MVG_min - 1) * 100:.1f}%        "
              f"{dcf_naive_norm:.3f}     {dcf_naive_min:.3f}     {(dcf_naive_norm / dcf_naive_min - 1) * 100:.1f}%       "
              f"{dcf_tied_norm:.3f}   {dcf_tied_min:.3f}     {(dcf_tied_norm / dcf_tied_min - 1) * 100:.1f}%")


def eval_PCA_effectivePI(D_tr, L_tr, D_val, L_val):
    eff_priors = [0.1, 0.5, 0.9]

    for m in range(1, 7):
        print("===========================================================================================")
        print(f"PCA_m = {m}")
        print("π    MVG   MVGmin  margin | naive   naiveMin  margin | tied   tiedMin  margin")

        P = get_PCA_projection_matrix(D_tr, m)
        D_tr_P = P.T @ D_tr
        D_val_P = P.T @ D_val

        logscore, _, _, _ = extract_densities_with_MVG(D_tr_P, L_tr, D_val_P)
        llrs_MVG = get_llrs(logscore)
        logscore, _, _, _ = extract_densities_with_naive(D_tr_P, L_tr, D_val_P)
        llrs_naive = get_llrs(logscore)
        logscore, _, _, _ = extract_densities_with_tied(D_tr_P, L_tr, D_val_P)
        llrs_tied = get_llrs(logscore)

        for pi in eff_priors:
            dcf_MVG, _, _ = get_dcf__includes_classification(llrs_MVG, L_val, pi, 1, 1)
            dcf_MVG_norm = get_normalized_dcf_binary(pi, 1, 1, dcf=dcf_MVG)
            dcf_MVG_min = get_min_normalized_dcf_binary(llrs_MVG, L_val, pi, 1, 1)

            dcf_naive, _, _ = get_dcf__includes_classification(llrs_naive, L_val, pi, 1, 1)
            dcf_naive_norm = get_normalized_dcf_binary(pi, 1, 1, dcf=dcf_naive)
            dcf_naive_min = get_min_normalized_dcf_binary(llrs_naive, L_val, pi, 1, 1)

            dcf_tied, _, _ = get_dcf__includes_classification(llrs_tied, L_val, pi, 1, 1)
            dcf_tied_norm = get_normalized_dcf_binary(pi, 1, 1, dcf=dcf_tied)
            dcf_tied_min = get_min_normalized_dcf_binary(llrs_tied, L_val, pi, 1, 1)

            print(f"{pi}  "
                  f"{dcf_MVG_norm:.3f} {dcf_MVG_min:.3f}   {(dcf_MVG_norm / dcf_MVG_min - 1) * 100:.1f}%     "
                  f"{dcf_naive_norm:.3f}    {dcf_naive_min:.3f}    {(dcf_naive_norm / dcf_naive_min - 1) * 100:.1f}%     "
                  f"{dcf_tied_norm:.3f}  {dcf_tied_min:.3f}    {(dcf_tied_norm / dcf_tied_min - 1) * 100:.1f}%")


def plot_bayes_error(D_tr, L_tr, D_val, L_val):
    logscore, _, _, _ = extract_densities_with_MVG(D_tr, L_tr, D_val)
    llrs_MVG = get_llrs(logscore)
    logscore, _, _, _ = extract_densities_with_naive(D_tr, L_tr, D_val)
    llrs_naive = get_llrs(logscore)
    logscore, _, _, _ = extract_densities_with_tied(D_tr, L_tr, D_val)
    llrs_tied = get_llrs(logscore)

    plot_bayes_error_plots([llrs_MVG, llrs_naive, llrs_tied], L_val, ["MVG", "Naive", "Tied"], "MVG_variants")



if __name__ == '__main__':
    D, L = load_fingerprints()
    (D_tr, L_tr), (D_val, L_val) = split_db_2to1(D, L)

    # eval_noPCA_noEffectivePI(D_tr, L_tr, D_val, L_val)
    # eval_PCA_effectivePI(D_tr, L_tr, D_val, L_val)

    # # # Best result is no PCA (same as m=5 and m=6 pretty much, but better for Naive Bayes)

    plot_bayes_error(D_tr, L_tr, D_val, L_val)
