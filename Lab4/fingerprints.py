from modules.classification import *
from modules.probability_first import *
from modules.plottings import *


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


def plot_histograms_and_pdfs_for_each_feature(D, class_name="AllDataset"):

    features_dimensionality = how_many_features(D)

    # Generating domain for calculating pdf
    XPlot = np.linspace(-5, 5, 1000)

    for feature_x in range(features_dimensionality):
        plot_name = f"Histogram_vs_Gaussian-class_{class_name}-feature{feature_x}"

        D_feature_x = D[feature_x, :].reshape(1, how_many_samples(D))
        mu_feature_x = D_feature_x.mean(1)
        Sigma_feature_x = get_covariance_matrix(D_feature_x)

        pdf_Gaussian_feature_x = np.exp(logpdf_GAU_ND(XPlot, mu_feature_x, Sigma_feature_x))

        plt.figure()

        plt.hist(D_feature_x.ravel(), bins=70, density=True)
        plt.plot(XPlot.ravel(), pdf_Gaussian_feature_x)

        plt.xlabel(f"class_{class_name}-feature-{feature_x}")

        # Making filename and dir organization readable
        save_folder = f"./Histogram_vs_Gaussian"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        full_name_with_path = os.path.join(save_folder, f"{plot_name}.pdf")

        plt.savefig(full_name_with_path)
        print(f"[[Saved {full_name_with_path}]]")

        plt.title(plot_name)
        plt.show()


if __name__ == "__main__":
    # Loading Data
    D, L = load_fingerprints()
    print_dataset_info(D, L, "fingerprints")

    unique_classes = get_unique_classes(L)
    print(unique_classes)
    index2class_nameFingerprints = {0: 'False', 1: 'True',}

    # Plotting, for each feature, the histogram distribution
    # and the pdf of the Gaussian with Mean and Variance equal to the empirical ones
    # Plotting also class by class

    # plot_histograms_and_pdfs_for_each_feature(D)
    # for class_x in unique_classes:
    #     D_of_class_x = D[:, L == class_x]
    #     plot_histograms_and_pdfs_for_each_feature(D_of_class_x, index2class_nameFingerprints[class_x])
