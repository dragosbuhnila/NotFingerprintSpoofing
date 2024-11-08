import scipy
from modules.classification import classify_with_threshold, get_classification_err, get_dumb_threshold
from modules.projections import get_LDA_projection_matrix
from modules.statistics import *
from modules.probability_first import logpdf_GAU_ND
from modules.probability_first import get_llr


def extract_densities_with_MVG(D_tr, L_tr, D_val, priors=None, verbose=False):
    if priors is None:
        priors = [1 / how_many_classes(L_tr) for _ in range(how_many_classes(L_tr))]

    logscore = compute_loglikelihoods(D_tr, L_tr, D_val, "MVG")

    # logSJointCheck = np.load('solution/logSJoint_MVG.npy')
    # logMarginalCheck = np.load('solution/logMarginal_MVG.npy')
    # logPosteriorCheck = np.load('solution/logPosterior_MVG.npy')
    # extract_densities_of_MVG_or_variants(logscore, logSJointCheck, logMarginalCheck, logPosteriorCheck, verbose=True)

    return extract_densities_of_MVG_or_variants(logscore, priors, verbose=verbose)


def extract_densities_with_naive(D_tr, L_tr, D_val, priors=None, verbose=False):
    if priors is None:
        priors = [1 / how_many_classes(L_tr) for _ in range(how_many_classes(L_tr))]

    logscore = compute_loglikelihoods(D_tr, L_tr, D_val, "NAIVE")

    # logSJointCheck = np.load('solution/logSJoint_NaiveBayes.npy')
    # logMarginalCheck = np.load('solution/logMarginal_NaiveBayes.npy')
    # logPosteriorCheck = np.load('solution/logPosterior_NaiveBayes.npy')
    # extract_densities_of_MVG_or_variants(logscore, logSJointCheck, logMarginalCheck, logPosteriorCheck, verbose=True)

    return extract_densities_of_MVG_or_variants(logscore, priors, verbose=verbose)


def extract_densities_with_tied(D_tr, L_tr, D_val, priors=None, verbose=False):
    if priors is None:
        priors = [1 / how_many_classes(L_tr) for _ in range(how_many_classes(L_tr))]

    logscore = compute_loglikelihoods(D_tr, L_tr, D_val, "TIED")

    # logSJointCheck = np.load('solution/logSJoint_TiedMVG.npy')
    # logMarginalCheck = np.load('solution/logMarginal_TiedMVG.npy')
    # logPosteriorCheck = np.load('solution/logPosterior_TiedMVG.npy')
    # extract_densities_of_MVG_or_variants(logscore, logSJointCheck, logMarginalCheck, logPosteriorCheck, verbose=True)

    return extract_densities_of_MVG_or_variants(logscore, priors, verbose=verbose)


def compute_loglikelihoods(D, L, D_val, which_cov, verbose=False):
    which_cov = which_cov.upper()
    if which_cov not in ("MVG", "TIED", "NAIVE"):
        raise ValueError("which_cov must be one of 'MVG', 'tied', or 'naive'")

    unique_classes = get_unique_classes(L)
    score = []

    for class_x in sorted(unique_classes):
        D_of_class_x = D[:, L == class_x]
        mu_of_class_x = get_mean(D_of_class_x)
        cov_of_class_x = get_covariance_matrix(D_of_class_x)

        # Qui non avrebbe alcun senso calcolare la logpdf usando D_of_class_x, siccome il nostro obiettivo è semplicemente
        #   ottenere vaire pdf (media e covarianza) classe per classe, ma poi applicarle ai dati a prescindere dalla classe.
        # Quindi dobbiamo si prendere media e cov da D_of_class ma poi valutiamo le likelihood dell'intero dataset su tutte le possibili pdf (in questo caso 3 perchè 3 classi abbiamo)
        #   e quindi ci piazziamo l'intero dataset D oppure solo la parte di validation D_val come in questo caso
        if which_cov == "MVG":
            loglikelihoods_of_class_x = logpdf_GAU_ND(D_val, mu_of_class_x, cov_of_class_x)
        elif which_cov == "NAIVE":
            loglikelihoods_of_class_x = logpdf_GAU_ND(D_val, mu_of_class_x, cov_of_class_x * np.eye(cov_of_class_x.shape[0]))
        elif which_cov == "TIED":
            within_class_cov = get_within_class_covariance_matrix(D, L)
            loglikelihoods_of_class_x = logpdf_GAU_ND(D_val, mu_of_class_x, within_class_cov)

        score.append(loglikelihoods_of_class_x)
        if verbose:
            print(f"Size of likelihoods of class {class_x} is {loglikelihoods_of_class_x.shape}")

    return np.vstack(score)


def compute_logjoint(score, priors):
    score_joint = [0 for _ in priors]
    for i in range(len(priors)):
        score_joint[i] = score[i] + np.log(priors[i])

    return np.vstack(score_joint)


def extract_densities_of_MVG_or_variants(logscore, priors, logSJointCheck=None, logMarginalCheck=None, logPosteriorCheck=None, verbose=False):
    # ==================================================
    # Compute and check Joint Score
    logscore_joint = compute_logjoint(logscore, priors)
    if not logSJointCheck is None and verbose:
        score_joint_expected = logSJointCheck
        print(np.abs(logscore_joint - score_joint_expected).max())

    # ==================================================
    # Compute marginal
    logscore_marginal = onedim_arr_to_rowvector(scipy.special.logsumexp(logscore_joint, axis=0))
    if not logMarginalCheck is None and verbose:
        logscore_marginal_expected = logMarginalCheck
        print(np.abs(logscore_marginal - logscore_marginal_expected).max())

    # ==================================================
    # Compute posteriors
    logscore_posterior = logscore_joint - logscore_marginal
    if not logPosteriorCheck is None and verbose:
        logscore_posterior_expected = logPosteriorCheck
        print(np.abs(logscore_posterior - logscore_posterior_expected).max())


    return logscore, logscore_joint, logscore_marginal, logscore_posterior


""" This function has a flaw, which is having class labels that do not start from zero (tbf this was fixed I think)
    or that have gaps in the values (e.g. unique_classes = [1, 4, 5, 6]).
    If you wish to fix it you're going to need to modify the result of
        > predictions       or      > L_val_shifted (like I do for this quick fix)"""
def classify_nclasses(logscore_posterior, L_val, verbose=False):
    score_posterior = np.exp(logscore_posterior)

    # Compute best classes sample by sample
    predictions = np.argmax(score_posterior, axis=0)
    L_val_shifted = L_val - min(get_unique_classes(L_val))  # This is needed bc argmax does not know the name of the
                                                            #   class, so if we remove class 0 all classes will be shifted.

    nof_errors = (predictions != L_val_shifted).sum()
    accuracy = (len(predictions) - nof_errors) / len(predictions)
    error_rate = nof_errors / len(predictions)

    if verbose:
        print("----------------------------------")
        print(f"Now classifying over {how_many_classes(L_val)} classes")
        print(f"There are {nof_errors} errors over {len(predictions)} test samples. "
              f"Accuracy is: {accuracy * 100}%. Error Rate is: {error_rate * 100}%")
        print(predictions)
        print(L_val_shifted)

    return nof_errors, accuracy, error_rate


""" Priors are set to 50%. Same problems as for classify_nclasses() may hold. """
def classify_two_classes(logscore, L_val, verbose=False, verbose_only_err=True, prior_one=0.5, prior_two=0.5):
    llrs = get_llr(logscore)
    threshold = -1 * np.log(prior_one/prior_two)
    predictions = classify_with_threshold(llrs, threshold)

    L_val_shifted = L_val - min(get_unique_classes(L_val))

    (nof_errors, error_rate, accuracy) = get_classification_err(L_val_shifted, predictions)

    if verbose:
        print(f"Now classifying over 2 classes (using llr and threshold)")
        print(f"There are {nof_errors} errors over {len(predictions)} test samples. "
              f"Accuracy is: {accuracy * 100:.2f}%. Error Rate is: {error_rate * 100:.2f}%")
        print(f"Predictions:    {predictions}")
        print(f"Labels:         {L_val_shifted}")
    elif verbose_only_err:
        print(f"err_rate = {error_rate * 100:.2f}%")

    return nof_errors, accuracy, error_rate

def classify_over_LDA(D_tr, L_tr, D_val, L_val, verbose=False, verbose_only_err=True):
    W = get_LDA_projection_matrix(D_tr, L_tr, get_unique_classes(L_tr))
    D_tr_W = W.T @ D_tr
    D_val_W = W.T @ D_val

    L_val_shifted = L_val - min(get_unique_classes(L_val))

    (threshold, D_val_W) = get_dumb_threshold(D_tr_W, D_val_W, L_tr, get_unique_classes(L_tr))
    predictions = classify_with_threshold(D_val_W, threshold)
    (nof_errors, error_rate, accuracy) = get_classification_err(L_val_shifted, predictions)

    if verbose:
        print(f"Now classifying over {how_many_classes(L_val)} classes (using LDA)")
        print(f"There are {nof_errors} errors over {len(predictions)} test samples. "
              f"Accuracy is: {accuracy * 100:.2f}%. Error Rate is: {error_rate * 100:.2f}%")
        print(f"Predictions:    {predictions}")
        print(f"Labels:         {L_val_shifted}")
    elif verbose_only_err:
        print(f"err_rate = {error_rate * 100:.2f}%")
