# # Suppress displaying plots
# def nop():
#     pass
# plt.show = nop

# # Print wider lines on console
# np.set_printoptions(edgeitems=30, linewidth=100000,
#     formatter=dict(float=lambda x: "%.3g" % x))
# np.set_printoptions(precision=4, suppress=True)


# Example of possible class/feature name translators for better looking graphs
index2class_nameIris = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica',}
index2feature_nameIris = {0: 'Sepal length', 1: 'Sepal width', 2: 'Petal length', 3: 'Petal width'}

index2class_nameFingerprints = {0: 'False', 1: 'True',}
