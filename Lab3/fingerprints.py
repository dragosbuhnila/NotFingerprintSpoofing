import numpy
import os

from common_matrix_operations import onedim_arr_to_colvector, onedim_arr_to_rowvector
from plottings import *
from statistics import *
from projections import *
from classification import *

def nop():
    pass
plt.show = nop


def load_fingerprints():
    fname = "fingerprints.txt"

    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = onedim_arr_to_colvector(numpy.array([float(i.strip()) for i in attrs]))
                label = line.split(',')[-1].strip()

                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


def split_db_2to1(D, L, seed=0):
    numpy.random.seed(seed)

    idx = numpy.random.permutation(D.shape[1])
    nTrain = int(D.shape[1] * 2.0 / 3.0)

    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)


if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    # Loading
    D, L = load_fingerprints()
    unique_classes = get_unique_classes(L)
    classes_dimensionality = len(unique_classes)
    class_names = {0: 'False', 1: 'True'}


    ''' Analyzing PCAed features '''
    print(">> Analyzing PCAed features")

    P = get_PCA_projection_matrix(D, D.shape[0])
    D_P = P.T @ D
    plot_hist(D_P, L, "Fingerprints_PCA", features_dimensionality=D_P.shape[0], classes_dimensionality=classes_dimensionality,
              index2class_name=class_names)
    for dirx in range(how_many_features(D_P)):
        plot_scatter_1d(D_P, L, "Fingerprints_PCA_Scatter", classes_dimensionality=classes_dimensionality,
                        index2class_name=class_names, direction_of_interest=dirx)
    print("====================================")



    ''' Analyzing LDAed features '''
    print(">> Analyzing LDAed features")

    W = get_LDA_projection_matrix(D, L, unique_classes)
    D_W = W.T @ D
    plot_hist(D_W, L, "Fingerprints_LDA", features_dimensionality=D_W.shape[0], classes_dimensionality=classes_dimensionality,
              index2class_name=class_names)
    plot_scatter_1d(D_W, L, "Fingerprints_LDA_Scatter", classes_dimensionality=classes_dimensionality,
                    index2class_name=class_names, direction_of_interest=0)
    print("====================================")


    ''' Classifying over LDA (W matrix) '''
    print(">> Classifying over LDA (W matrix)")

    # Splitting in training and validation sets
    (D_training, L_training), (D_validation, L_validation) = split_db_2to1(D, L)
    unique_classes = get_unique_classes(L_training)
    print_dataset_info(D_training, L_training, "Training")
    print_dataset_info(D_validation, L_validation, "Validation")

    # Calculating projection matrix for LDA (W), then projecting all the data
    W = get_LDA_projection_matrix(D_training, L_training, unique_classes)

    D_training_W = W.T @ D_training
    D_validation_W = W.T @ D_validation
    plot_hist(D_training_W, L_training, "Fingerprints_Training_LDA",
              features_dimensionality=D_training_W.shape[0], classes_dimensionality=len(unique_classes),
              index2class_name=class_names)
    plot_hist(D_validation_W, L_validation, "Fingerprints_Validation_LDA",
              features_dimensionality=D_validation_W.shape[0], classes_dimensionality=len(unique_classes),
              index2class_name=class_names)

    # Fetching some kind of classifier from the training data and then using it on the validation data
    (threshold, D_validation_W) = get_dumb_threshold(D_training_W, D_validation_W, L_training, unique_classes, class_names)
    prediction = classify(D_validation_W, threshold)
    (misses, err_rate_LDAnaive) = get_classification_err(L_validation, prediction, how_many_samples(D_validation))

    print(f"Threshold={threshold}: {misses} misses (over {how_many_samples(D_validation)} samples) "
          f"detected using LDA, which is a {err_rate_LDAnaive}% error rate.")
    print("====================================")


    ''' Variating Threshold '''
    print(">> Variating Threshold")

    (min_threshold, misses, err_rate_LDAoptimized) = try_variating_threshold(D_validation_W, L_validation, misses, threshold,
                                                                folder_name="LDA_only_Thresholds_And_Misses")
    print(f"Threshold={min_threshold}: {misses} misses (over {how_many_samples(D_validation)} samples) "
          f"detected using LDA, which is a {err_rate_LDAoptimized}% error rate.")
    print("====================================")


    ''' Classifying by preprocessing LDA with PCA while variating m_PCA *** NEED TO RECOMPUTE P AND W *** '''
    print(">> Classifying by preprocessing LDA with PCA while variating m_PCA")

    errs_naive = [x for x in range(2, 7)]
    errs_optimized = [x for x in range(2, 7)]
    m_PCAs = [x for x in range(2, 7)]
    # First: PCA. The m for PCA has to be greater than the m for LDA (m_LDA is 1 in this case, since only 2 classes)
    for m_PCA in range(2, 7):
        P = get_PCA_projection_matrix(D_training, m_PCA)
        D_training_P = P.T @ D_training
        D_validation_P = P.T @ D_validation

        # Second: LDA, over the already PCAed data.
        W = get_LDA_projection_matrix(D_training_P, L_training, unique_classes)
        D_training_PW = W.T @ D_training_P
        D_validation_PW = W.T @ D_validation_P

        # Finally getting classifier from the training part of the data and using it on validation data
        (threshold, D_validation_PW) = get_dumb_threshold(D_training_PW, D_validation_PW, L_training, unique_classes, class_names)

        prediction = classify(D_validation_PW, threshold)
        (misses, err_rate) = get_classification_err(L_validation, prediction, how_many_samples(D_validation))
        print(f"m={m_PCA} naive threshold: {misses} misses (over {how_many_samples(D_validation)} samples) "
              f"detected using LDA with PCA preprocessing, which is a {err_rate}% error rate.")

        (_, misses_min, err_rate_min) = try_variating_threshold(D_validation_PW, L_validation, misses, threshold,
                                                                folder_name="LDA_and_PCA_Thresholds_And_Misses",
                                                                plot_name=f"m={m_PCA}")
        print(f"m={m_PCA} best threshold: {misses_min} misses (over {how_many_samples(D_validation)} samples) "
              f"detected using LDA with PCA preprocessing, which is a {err_rate_min}% error rate.")

        print("-------------------------------------")

        errs_naive[m_PCA - 2] = err_rate
        errs_optimized[m_PCA - 2] = err_rate_min

    # Plot the results
    plot_name = "m_vs_Error_Rate"
    plt.figure(num=plot_name)
    plt.xlabel("m")
    plt.ylabel("Error Rate")
    plt.title(f"Error rate variation")

    plt.plot(m_PCAs, errs_naive, label="Naive Thresholds")
    plt.plot(m_PCAs, errs_optimized, label="Optimized Thresholds")
    plt.plot(m_PCAs, [err_rate_LDAnaive for _ in range(len(m_PCAs))])
    plt.plot(m_PCAs, [err_rate_LDAoptimized for _ in range(len(m_PCAs))])

    plt.legend()
    plt.tight_layout()

    save_folder = f"./{plot_name}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    full_name_with_path = os.path.join(save_folder, f"./{plot_name}.pdf")

    plt.savefig(full_name_with_path)
    print(f"Saved {plot_name}.pdf")
    plt.show()
