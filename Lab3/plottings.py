import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

# ho un certo numero di classi
#     > devo nominare evenatualmente ogni classe
#         - dizionario tipo:
#         index2class_name = {
#             0: 'Setosa',
#             1: 'Versicolor',
#             2: 'Virginica',
#         }
# ho un certo numero di attributi
#     > devo generare un istogramma per ognuno
#     > devo nominare ogni grafico eventualmente con il nome dell'attributo
#         - dizionario tipo:
#         index2feature_name = {
#             0: 'Sepal length',
#             1: 'Sepal width',
#             2: 'Petal length',
#             3: 'Petal width'
#         }


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
            plt.hist(Data_of_class_x[feature_n, :],bins=bins, density=True, alpha=alpha,
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