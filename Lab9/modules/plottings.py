import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

from others import get_dcf
from modules.evaluation import get_normalized_dcf_binary, get_min_normalized_dcf_binary
from modules.statistics import *

# # Example of possible class/feature name translators for better looking graphs
# index2class_name = {
#     0: 'Setosa',
#     1: 'Versicolor',
#     2: 'Virginica',
# }
# index2feature_name = {
#     0: 'Sepal length',
#     1: 'Sepal width',
#     2: 'Petal length',
#     3: 'Petal width'
# }


""" I should update this (probably scatter too) in order to accept the list of classes instead of their dimensionality,
    because if I, for example, have a dataset with class n removed, the way the function currently is it will break """
def plot_hist(Data, Labels, plot_name: str, features_dimensionality: int, classes_dimensionality: int,
              index2feature_name=None, index2class_name=None, bins=10, alpha=0.4):
    if Data.shape[1] != Labels.size:
        raise ValueError("Number of columns in Data does not match the size of Labels array")

    # Prepare label dictionaries if missing:
    if index2feature_name is None:
        index2feature_name = {}
        for feature_n in range(features_dimensionality):
            index2feature_name[feature_n] = "feature-" + str(feature_n)
    if index2class_name is None:
        index2class_name = {}
        for class_n in range(classes_dimensionality):
            index2class_name[class_n] = "class-" + str(class_n)

    # Plot a figure for each feature
    for feature_n in range(features_dimensionality):
        plt.figure(num=f"Histogram_{plot_name}_{index2feature_name[feature_n]}")
        plt.xlabel(index2feature_name[feature_n])

        # On that figure, plot each class separately (i.e. different colors or shapes)
        for class_n in range(classes_dimensionality):
            Data_of_class_x = Data[:, Labels == class_n]
            plt.hist(Data_of_class_x[feature_n, :], bins=bins, density=True, alpha=alpha,
                     label=index2class_name[class_n])

        plt.legend()
        plt.tight_layout()

        # Making filename and dir organization readable
        save_folder = f"./{plot_name}"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        full_name_with_path = os.path.join(save_folder, f"./Histogram_{plot_name}_"
                                                        f"{index2feature_name[feature_n]}.pdf")

        plt.savefig(full_name_with_path)
        print(f"[[Saved {full_name_with_path}]]")

    plt.show()


def plot_hist3d(Data, plot_name: str, Labels=None, ax=None,
                index2feature_name=None, index2class_name=None, bins=20):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    plot_hist3d_no_figure_show(Data, plot_name, ax, Labels, index2feature_name, index2class_name, bins)
    plt.show()


""" This function plots a single scatter: Data must have ONLY two features. If it has more, extract a subset 
    using Data[[feature0, feature1], :]"""
def plot_hist3d_no_figure_show(Data, plot_name: str, ax, Labels=None,
                index2feature_name=None, index2class_name=None, bins=20):

    print(how_many_features(Data))

    # Check if we have too many features/dimensions
    if how_many_features(Data) != 2:
        raise ValueError("In 'plot_hist3d(...)' the parameter Data should have only two features, "
                         "i.e. np.array.shape = (2, n)")

    # Check if we have class labels: if not use a single class, i.e. consider the data as a whole
    if Labels is None:
        unique_classes = [0]
        Labels = np.zeros(how_many_samples(Data))
    else:
        unique_classes = np.unique(Labels)

    # Prepare label dictionaries if missing:
    features_dimensionality = 2
    if index2feature_name is None:
        index2feature_name = {}
        for feature_n in range(features_dimensionality):
            index2feature_name[feature_n] = "feature-" + str(feature_n)
    if index2class_name is None:
        index2class_name = {}
        for class_n in range(len(unique_classes)):
            index2class_name[class_n] = "class-" + str(class_n)

    # Check if name_translation-dictionaries are the correct size
    if len(index2class_name) != len(unique_classes):
        raise ValueError(f"In 'plot_hist3d(...)' the parameter 'index2class_name' should have "
                         f"{len(unique_classes)} items, but instead has {len(index2class_name)}")
    if len(index2feature_name) != 2:
        raise ValueError(f"In 'plot_hist3d(...)' the parameter 'index2feature_name' should have "
                         f"2 items, but instead has {len(index2class_name)}")

    for class_label in unique_classes:
        Data_of_class_x = Data[:, Labels == class_label]

        hist, xedges, yedges = np.histogram2d(Data_of_class_x[0, :], Data_of_class_x[1, :], bins=bins, density=True)
        hist /= 5.714

        # Construct arrays for the anchor positions of the bars.
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        # Construct arrays with the dimensions for the bars.
        dx = dy = 0.5 * np.ones_like(zpos)
        dz = hist.ravel()

        # Plot the 3D histogram
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', label=f'{index2class_name[class_label]}')

    # Errors may arise here if dictionary has keys that are not 0 and 1
    ax.set_xlabel(f'{index2feature_name[0]}')
    ax.set_ylabel(f'{index2feature_name[1]}')
    ax.set_zlabel('Frequency')

    ax.legend()

    plt.legend()
    plt.tight_layout()
    plt.title(plot_name)

    # Making filename and dir organization readable
    save_folder = f"./{plot_name}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    full_name_with_path = os.path.join(save_folder, f"3DHistogram_{plot_name}_"
                                                    f"x-{index2feature_name[1]}_y-{index2feature_name[0]}.pdf")

    plt.savefig(full_name_with_path)
    print(f"[[Saved {full_name_with_path}]]")

""" This function plots many scatters: either all combinations of features or the couples specified
    in the 'couples_of_interest: tuple' parameter """
def plot_scatter(Data, Labels, plot_name, features_dimensionality, classes_dimensionality,
                 index2feature_name=None, index2class_name=None, alpha=0.5, couples_of_interest=None):
    if Data.shape[1] != Labels.size():
        raise ValueError("Number of columns in Data does not match the size of Labels array")

    # Prepare label dictionaries if missing:
    if index2feature_name is None:
        index2feature_name = {}
        for feature_n in range(features_dimensionality):
            index2feature_name[feature_n] = "feature-" + str(feature_n)
    if index2class_name is None:
        index2class_name = {}
        for class_n in range(classes_dimensionality):
            index2class_name[class_n] = "class-" + str(class_n)

    # Plot a figure for each feature
    for feature_i in range(features_dimensionality):
        for feature_j in range(features_dimensionality):
            if feature_i == feature_j:
                continue
            # Trim the combinations only to the ones of interest
            if not (couples_of_interest is None) and not ((feature_i, feature_j) in couples_of_interest):
                continue
            plt.figure(num=f"Scatter_{plot_name}_x-{index2feature_name[feature_i]}_y-{index2feature_name[feature_j]}")
            plt.xlabel(index2feature_name[feature_i])
            plt.ylabel(index2feature_name[feature_j])

            # On that figure, plot each class separately (i.e. different colors or shapes)
            for class_n in range(classes_dimensionality):
                Data_of_class_x = Data[:, Labels == class_n]
                plt.scatter(Data_of_class_x[feature_i, :], Data_of_class_x[feature_j, :],
                            alpha=alpha, label=index2class_name[class_n])

            plt.legend()
            plt.tight_layout()

            # Making filename and dir organization readable
            save_folder = f"./{plot_name}"
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            full_name_with_path = os.path.join(save_folder, f"Scatter_{plot_name}_"
                                                            f"x-{index2feature_name[feature_i]}_y-{index2feature_name[feature_j]}.pdf")

            plt.savefig(full_name_with_path)
            print(f"[[Saved {full_name_with_path}]]")

        plt.show()


""" direction_of_interest is the same as the feature of interest """
def plot_scatter_1d(Data, Labels, plot_name, classes_dimensionality, index2class_name=None,
                    direction_of_interest=0, feature_name=None, alpha=0.5, spread_out=True):
    # Prepare label dictionaries if missing:
    if index2class_name is None:
        index2class_name = {}
        for class_n in range(classes_dimensionality):
            index2class_name[class_n] = "class-" + str(class_n)
    if feature_name is None:
        feature_name = f"direction-{direction_of_interest}"

    plt.figure()
    plt.xlabel(f"samples")
    plt.ylabel(f"{feature_name}")

    # Plot each class separately (i.e. different colors or shapes)
    for class_n in range(classes_dimensionality):
        Data_of_class_x = Data[:, Labels == class_n]
        # Show samples spread out or on a line
        if spread_out:
            x = np.arange(Data_of_class_x.shape[1])
            x = x.reshape((1, len(x)))
        else:
            x = np.ones(Data_of_class_x.shape[1])
            x = x.reshape((1, len(x)))

        plt.scatter(x[0, :], Data_of_class_x[direction_of_interest, :],
                    alpha=alpha, label=index2class_name[class_n])

    plt.legend()
    plt.tight_layout()

    # Making filename and dir organization readable
    save_folder = f"./{plot_name}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    full_name_with_path = os.path.join(save_folder, f"Scatter_{plot_name}_{feature_name}.pdf")

    plt.savefig(full_name_with_path)
    print(f"[[Saved {full_name_with_path}]]")

    plt.show()


def plot_scatter_3d(D, L, projection_type, dir0=0, dir1=1, dir2=2):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    D2 = D[:, L == 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(D0[dir0, :], D0[dir1, :], D0[dir2, :], label='Setosa')
    ax.scatter(D1[dir0, :], D1[dir1, :], D1[dir2, :], label='Versicolor')
    ax.scatter(D2[dir0, :], D2[dir1, :], D2[dir2, :], label='Virginica')

    # Set labels
    ax.set_xlabel(f"{projection_type}_direction_{dir0}")
    ax.set_ylabel(f"{projection_type}_direction_{dir1}")
    ax.set_zlabel(f"{projection_type}_direction_{dir2}")

    ax.set_title('3D Scatter Plot')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
    plt.savefig(f'{projection_type}_iris_scatter_1D-Directions_{dir0}_{dir1}_{dir2}.pdf')

    plt.show()


def plot_mvg(mu, Sigma, plot_name, ax=None, n_std=3, resolution=100, alpha=0.9):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    plot_mvg_no_figure_show(mu, Sigma, plot_name, ax, n_std=3, resolution=100, alpha=alpha)
    plt.show()


def plot_mvg_no_figure_show(mu, Sigma, plot_name, ax, n_std=3, resolution=100, alpha=0.4):
    if mu.size != 2:
        raise ValueError(f'In "plot_mvg(mu, Sigma, ...)" the parameter "mu" should be size 2, instead is {mu.size}.')
    if Sigma.shape != (2, 2):
        raise ValueError(f'In "plot_mvg(mu, Sigma, ...)" the parameter "Sigma" should have shape (2, 2), '
                         f'which instead is {Sigma.shape}.')

    # Create a grid of points
    x = np.linspace(mu[0] - n_std * np.sqrt(Sigma[0, 0]), mu[0] + n_std * np.sqrt(Sigma[0, 0]), resolution)
    y = np.linspace(mu[1] - n_std * np.sqrt(Sigma[1, 1]), mu[1] + n_std * np.sqrt(Sigma[1, 1]), resolution)
    X, Y = np.meshgrid(x, y)

    # Create the MVG distribution
    from scipy.stats import multivariate_normal
    rv = multivariate_normal(mu, Sigma)

    # Calculate the probability density at each point in the grid
    Z = rv.pdf(np.dstack((X, Y)))

    # Plot the MVG distribution in 3D
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=alpha)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Probability Density')
    ax.set_title(f"{plot_name}")


""" If you want to modify the range just do it manually for now """
def plot_bayes_error_plots(llrs_list, labels, names, range=None):
    if range is None:
        range = (-4, 4, 21)
    elif len(range) != 3:
        raise ValueError("range is a triplet that should contain (start, end, nof_steps)")
    effPriorLogOdds = np.linspace(*range)

    plt.figure(num="Bayes Error Plot")

    for (llrs, name) in zip(llrs_list, names):
        normalized_dcfs = []
        min_dcfs = []

        for p in effPriorLogOdds:
            print(f"[[[[now in prior {p}]]]]")
            pi = 1 / (1 + np.exp(-p))
            dcf = get_dcf(pi, 1, 1, llrs, labels)
            normalized_dcf = get_normalized_dcf_binary(pi, 1, 1, dcf)
            min_dcf = get_min_normalized_dcf_binary(llrs, labels, pi, 1, 1)

            normalized_dcfs.append(normalized_dcf)
            min_dcfs.append(min_dcf)

        plt.plot(effPriorLogOdds, normalized_dcfs, label=f"{name}")
        plt.plot(effPriorLogOdds, min_dcfs, label=f"min-{name}")
        print("[[Plotted 1st classifier]]")

    plt.ylim([0, 1.1])
    plt.xlim([range[0], range[1]])
    plt.xlabel('prior log-odds')
    plt.ylabel('DCF value')
    plt.title("Bayes Error Plot")
    plt.legend()
    plt.grid(True)
    plt.show()
