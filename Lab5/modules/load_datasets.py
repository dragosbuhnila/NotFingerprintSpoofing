from modules.classification import *
from modules.plottings import *


""" Returns 2 valeus: Data, Labels """
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


""" Returns 2 values: Data, Labels """
def load_iris():
    # The dataset is already available in the sklearn library (pay attention that the library represents samples as row vectors, not column vectors - we need to transpose the data matrix)
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']


""" Returns two tuples: (TrainingData, TrainingLabels), (ValidationData, ValidationLabels)"""
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
