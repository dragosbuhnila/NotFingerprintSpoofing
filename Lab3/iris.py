import numpy

from common_matrix_operations import onedim_arr_to_colvector, onedim_arr_to_rowvector
from plottings import *
from statistics import *
from projections import *

# def nop():
#     pass
# plt.show = nop


def load_iris():

    # The dataset is already available in the sklearn library (pay attention that the library represents samples as row vectors, not column vectors - we need to transpose the data matrix)
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']


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

    """ >>> Reducing to 2 classes only (removing class 0)***"""
    D, L = load_iris()
    D = D[:, L != 0]
    L = L[L != 0]
    class_names = {0: "Setosa", 1: 'Versicolor', 2: 'Virginica'}

    unique_classes = get_unique_classes(L)
    classes_dimensionality = how_many_classes(L)
    print(f"Unique classes: {unique_classes}. How many: {classes_dimensionality}")

    ''' Classifying over LDA (W matrix) '''
    # Splitting in training and validation sets
    (D_training, L_training), (D_validation, L_validation) = split_db_2to1(D, L)

    # Calculating projection matrix for LDA (W), then projecting all the data
    W = get_LDA_projection_matrix(D_training, L_training, unique_classes)

    D_training_W = W.T @ D_training
    D_validation_W = W.T @ D_validation

    plot_hist(D_training_W, L_training, "Iris_Training_Set_LDAed",
              features_dimensionality=how_many_features(D_training_W), classes_dimensionality=3,
              index2class_name=class_names, bins=5)
    plot_hist(D_validation_W, L_validation, "Iris_Validation_Set_LDAed",
              features_dimensionality=how_many_features(D_validation_W), classes_dimensionality=3,
              index2class_name=class_names, bins=5)