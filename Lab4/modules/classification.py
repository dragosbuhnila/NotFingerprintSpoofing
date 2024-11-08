import numpy
import matplotlib.pyplot as plt
import os

from modules.statistics import *

def classify(D, threshold):
    prediction = numpy.zeros(shape=(D.shape[1]), dtype=numpy.int32)
    prediction[D[0] >= threshold] = 1
    prediction[D[0] < threshold] = 0

    return prediction


def get_classification_err(L, prediction, total_samples, verbose=False):
    misses = 0
    for i in range(len(L)):
        if verbose:
            print(f"actual: {L[i]}, predicted: {prediction[i]}")
        if L[i] != prediction[i]:
            misses += 1

    rate = misses / total_samples * 100
    return misses, rate


def get_dumb_threshold(D_training, D_validation, L, unique_classes, index2class_names):
    means = get_mean_by_class(D_training, L, unique_classes, index2class_names)  # Size of means will be equal to features dimensionality
    if means[index2class_names[1]] < means[index2class_names[0]]:
        D_validation *= -1

    threshold = np.array([0.0])
    for mean in means.values():
        threshold += mean
    threshold /= len(means)

    if threshold.size == 1:
        return threshold[0], D_validation
    else:
        return threshold, D_validation


def try_variating_threshold(D, L, misses, threshold, verbose=False, extremely_verbose=False,
                            plot_name="Thresholds_vs_Misses", folder_name=".", step=0.001, tries=400):
    thresholds_vec = []
    misses_vec = []

    min_misses = misses
    min_threshold = threshold
    min_i = 0

    original_threshold = threshold
    threshold = threshold - step*(tries / 2)

    for i in range(tries):
        threshold += 0.001
        prediction = classify(D, threshold)
        (misses, err_rate) = get_classification_err(L, prediction, how_many_samples(D))

        thresholds_vec.append(threshold)
        misses_vec.append(misses)

        if misses < min_misses:
            min_misses = misses
            min_threshold = threshold
            min_i = i
            if extremely_verbose:
                print(f"{i}) Threshold={threshold}: {misses} misses (over {how_many_samples(D)} samples) "
                      f"detected using LDA, which is a {err_rate}% error rate.")

    plt.figure(num=plot_name)
    plt.xlabel("Threshold")
    plt.ylabel("Misses")
    plt.title(f"{plot_name}")

    plt.plot(thresholds_vec, misses_vec)

    plt.legend(f"Total Samples = {how_many_samples(D)}")
    plt.tight_layout()

    save_folder = f"./{folder_name}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    full_name_with_path = os.path.join(save_folder, f"./{plot_name}.pdf")

    plt.savefig(full_name_with_path)
    print(f"Saved {plot_name} variating threshold.pdf")
    plt.show()

    if verbose:
        print(f"info:::The minimum was found at try {min_i}, so at threshold: {original_threshold} + {min_i*step} = {min_threshold}")

    return min_threshold, min_misses, min_misses / how_many_samples(D) * 100
