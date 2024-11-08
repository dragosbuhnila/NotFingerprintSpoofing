import numpy
import scipy


""" m is features_dimensionality, i.e. how many dimensions for the features """
def get_PCA_projection_matrix(D, features_dimensinality, verbose=False):
    mu = D.mean(1).reshape((D.shape[0], 1))
    if verbose:
        print('Mean:')
        print(mu)
        print()

    DC = D - mu

    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    if verbose:
        print('Covariance:')
        print(C)
        print()

    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:features_dimensinality]
    P *= -1

    if verbose:
        print(f"First {features_dimensinality} eigenvectors:")
        print(P)
        print()

        print(f"Expected ones:")
        print(numpy.load("IRIS_PCA_matrix_m4.npy"))

    return P


""" class_indices isn't really necessary because D for the class that has been removed is empty, thus not changing
    the results of the computations (probably) """
def get_LDA_projection_matrix(D, L, class_indices, verbose=False):
    nof_features = D.shape[0]
    nof_samples = D.shape[1]

    # First we calculate the Within Class Variance
    Cov_W = numpy.zeros((nof_features, nof_features))

    for cls in class_indices:
        D_cls = D[:, L == cls]
        mu_cls = D_cls.mean(1).reshape((D_cls.shape[0]), 1)

        Cov_cls = (D_cls - mu_cls) @ (
                    D_cls - mu_cls).T  # Missing the /nc term, bc it would elide later, so we save a computation

        Cov_W += Cov_cls

    Cov_W /= D.shape[1]
    if verbose:
        print("Sw is:")
        print(Cov_W)
        print()

    # Now the Between Class Variance
    Cov_B = numpy.zeros((nof_features, nof_features))
    mu = D.mean(1).reshape((D.shape[0]), 1)

    for cls in class_indices:
        D_cls = D[:, L == cls]
        mu_cls = D_cls.mean(1).reshape((D_cls.shape[0]), 1)

        nof_samples_in_cur_class = D_cls.shape[1]
        Cov_B += nof_samples_in_cur_class * ((mu_cls - mu) @ (mu_cls - mu).T)

    Cov_B /= nof_samples
    if verbose:
        print("Sb is:")
        print(Cov_B)
        print()

    # Complete Cov is actually the sum of these two components
    # Cov = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    # if verbose:
    #     print('Covariance:')
    #     print(Cov)
    #     print()
    #     print("Sum of Sb and Sw:")
    #     print(Cov_B + Cov_W)
    #     print()

    # Solve maximization problem through generalized eigenproblem
    _, U = scipy.linalg.eigh(Cov_B, Cov_W)  # first returned is eigenvalues second is eigenvectors
    W = U[:, ::-1][:, 0:(len(class_indices) - 1)]

    # Eventual orthonormalization of W
    # W_ort_full, _, _ = numpy.linalg.svd(W)
    # W_ort = W_ort_full[:, 0:(nof_classes-1)]

    if verbose:
        res = numpy.load("IRIS_LDA_matrix_m2.npy")

        print("Basis for LDA is:")
        print(W)
        print()

        print("Expected result was:")
        print(res)
        print()

    return W