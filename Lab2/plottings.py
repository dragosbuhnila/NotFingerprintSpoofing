import matplotlib.pyplot as plt
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


def plot_hist(Data, Labels, plot_name, features_dimensionality, classes_dimensionality,
              index2feature_name=None, index2class_name=None, bins=10, alpha=0.4):

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
        plt.figure()
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

    plt.show()


def plot_scatter(Data, Labels, plot_name, features_dimensionality, classes_dimensionality,
                 index2feature_name=None, index2class_name=None, alpha=0.5, couples_of_interest=None):
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
            plt.figure()
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

        plt.show()

