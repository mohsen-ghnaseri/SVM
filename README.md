# SVM
## implement Support vector mechine (SVM) on Iris dataset with different kernel
Comparison of different linear SVM classifiers on a 2D projection of the iris dataset. We only consider the first 2 features of this dataset:

- petal length
- petal width

This example shows how to plot the decision surface for four SVM classifiers with different kernels
The linear models **LinearSVC()** and **SVC(kernel='linear')** yield slightly different decision boundaries. This can be a consequence of the following differences:

LinearSVC minimizes the squared hinge loss while SVC minimizes the regular hinge loss.

LinearSVC uses the _One-vs-All_ (also known as _One-vs-Rest_) multiclass reduction while SVC uses the _One-vs-One_ multiclass reduction.

Both linear models have linear decision boundaries (intersecting hyperplanes) while the non-linear kernel models (**polynomial** or **Gaussian RBF**) have more flexible non-linear decision boundaries with shapes that depend on the kind of kernel and its parameters.
