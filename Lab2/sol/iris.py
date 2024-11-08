import numpy
import matplotlib
import matplotlib.pyplot as plt
from plottings import plot_hist, plot_scatter

def mcol(v):
    return v.reshape((v.size, 1))

def load(fname):
    DList = []
    labelsList = []
    hLabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
        }

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


def load2():

    # The dataset is already available in the sklearn library (pay attention that the library represents samples as row vectors, not column vectors - we need to transpose the data matrix)
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

    
if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load('iris.csv')

    class_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    feature_names = {0: 'Sepal-length', 1: 'Sepal-width', 2: 'Petal-length', 3: 'Petal-width'}

    plot_hist(D, L, "test", features_dimensionality=D.shape[0], classes_dimensionality=3,
              index2class_name=class_names, index2feature_name=feature_names, bins=10)
    plot_scatter(D, L, "test", features_dimensionality=D.shape[0], classes_dimensionality=3,
                 index2class_name=class_names, index2feature_name=feature_names)


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

    for cls in [0,1]:
        print('Class', cls)
        DCls = D[:, L==cls]
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
        
    
