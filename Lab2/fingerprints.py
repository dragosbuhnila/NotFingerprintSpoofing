import numpy
import matplotlib
import matplotlib.pyplot as plt
from plottings import plot_hist, plot_scatter


def nop():
    pass
plt.show = nop


'''
Should only use on row_vectors or (n,) numpy arrays
'''
def to_col_vector(v):
    return v.reshape((v.size, 1))


def load_fingerprints():
    fname = "fingerprints.txt"

    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = to_col_vector(numpy.array([float(i.strip()) for i in attrs]))
                label = line.split(',')[-1].strip()

                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load_fingerprints()
    # plot_hist(D, L, "Test", classes_dimensionality=2, features_dimensionality=D.shape[0],
    #           index2class_name={0: "False", 1: "True"})
    plot_scatter(D, L, "Test", classes_dimensionality=2, features_dimensionality=D.shape[0],
                 index2class_name={0: "False", 1: "True"}, couples_of_interest=[(0,1), (2,3), (4,5)])

    mu = D.mean(1).reshape((D.shape[0], 1))
    print('Mean:')
    print(mu)
    print()

    DC = D - mu

    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    print('Covariance:')
    print(C)
    print()

    var = D.var(1)
    std = D.std(1)
    print('Variance:', var)
    print('Std. dev.:', std)
    print()

    
    for cls in [0, 1]:
        print('Class', cls)
        DCls = D[:, L == cls]
        mu = DCls.mean(1).reshape(DCls.shape[0], 1)
        print('Mean:')
        print(mu)
        C = ((DCls - mu) @ (DCls - mu).T) / float(DCls.shape[1])
        print('Covariance:')
        print(C)
        var = DCls.var(1)
        std = DCls.std(1)
        print('Variance:', var)
        print('Std. dev.:', std)
        print()
    

