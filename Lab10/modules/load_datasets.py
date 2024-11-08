from modules.common_matrix_operations import *
import numpy
import sklearn.datasets


def load_fingerprints():
    """ Returns 2 valeus: Data, Labels """
    fname = "fingerprints.txt"

    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = vcol(numpy.array([float(i.strip()) for i in attrs]))
                label = line.split(',')[-1].strip()

                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


def load_iris():
    """ Returns 2 values: Data, Labels (Labels is an (N,) np.array) """
    # The dataset is already available in the sklearn library (pay attention that the library represents samples
    # as row vectors, not column vectors - we need to transpose the data matrix)
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)
    return D, L


def split_db_2to1(D, L, seed=0):
    """ Returns two tuples: (TrainingData, TrainingLabels), (ValidationData, ValidationLabels)"""
    numpy.random.seed(seed)

    idx = numpy.random.permutation(D.shape[1])
    nTrain = int(D.shape[1] * 2.0 / 3.0)

    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    D_tr = D[:, idxTrain]
    D_val = D[:, idxTest]
    L_tr = L[idxTrain]
    L_val = L[idxTest]

    return (D_tr, L_tr), (D_val, L_val)


def shrink_dataset(D, L, k):
    """ Reduce the dataset by a factor k (e.g. k=50 and samples=1000 will return 20 samples) """
    if k < 1:
        raise ValueError("k must be an integer greater than 0")
    if k == 1:
        return D, L
    return D[:, ::k], L[::k]
