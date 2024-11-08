from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from modules.statistics import *
from modules.probability_first import *
from modules.plottings import *
from modules.load_datasets import load_iris


if __name__ == "__main__":
    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1, 1)) * 1.0
    C = np.ones((1, 1)) * 2.0
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(XPlot, m, C)))
    plt.show()

    D, L = load_iris()
    D0 = D[:, L == 0]

    # Correctness check #1 - Single Feature
    pdfSol = np.load('Solution/llGAU.npy')
    pdfGau = logpdf_GAU_ND(XPlot, m, C)
    print(np.abs(pdfSol - pdfGau).max())

    # Correctness check #2 - Multiple Features (actually they're just 2)
    XND = np.load('Solution/XND.npy')
    mu = np.load('Solution/muND.npy')
    C = np.load('Solution/CND.npy')
    pdfSol = np.load('Solution/llND.npy')
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    print(np.abs(pdfSol - pdfGau).max())
    print(np.exp(pdfGau))

    # # Trying 3d plotting the histogram
    # plot_hist3d(XND, "3D_Hist_MVG_Test")
    #
    # # Trying 3d plotting the Bivariate Gaussian
    # XND_mu = XND.mean(1)
    # XND_Sigma = get_covariance_matrix(XND)
    # plot_mvg(XND.mean(1), get_covariance_matrix(XND), "3D_BivariateGaussian_Test")
    #
    # # Trying to combine the 3d plots into a single one (Histogram and Bivariate Gaussian)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111, projection='3d')
    # plot_hist3d_no_figure_show(XND, "3D_Hist_MVG_Test", ax1, bins=20)
    # plot_mvg_no_figure_show(XND.mean(1), get_covariance_matrix(XND), "3D_BivariateGaussian_Test", ax1)
    # plt.show()
    #
    # # Calculating and checking log likelihood
    # ll = loglikelihood(XND, XND.mean(1), get_covariance_matrix(XND))
    # print(ll)
    #
    # # Trying with another dataset, this time 1D, so we can easily plot histogram and distribution
    # X1D = np.load('Solution/X1D.npy')
    # plt.figure()
    # plt.hist(X1D.ravel(), bins=50)
    # XPlot = np.linspace(-8, 12, 1000)
    # plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(XPlot, X1D.mean(1), get_covariance_matrix(X1D))))
    # plt.show()
    #
    # ll = loglikelihood(X1D, X1D.mean(1), get_covariance_matrix(X1D))
    # print(ll)