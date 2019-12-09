import sys
import numpy as np
from scipy.stats import describe
import os
import time
import matplotlib
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator
import warnings
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import itertools
import multiprocessing as mp

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import svd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

h = .02  # step size in the mesh
delimiter = ";"

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# plt.interactive(True)
# http://ksrowell.com/blog-visualizing-data/2012/02/02/
# optimal-colors-for-graphs/
MY_BLUE = (57, 106, 177)
MY_ORANGE = (218, 124, 48)
MY_GREEN = (62, 150, 81)
MY_RED = (204, 37, 41)
MY_BLACK = (83, 81, 84)
MY_GOLD = (148, 139, 61)
MY_VIOLET = (107, 76, 154)
MY_BROWN = (146, 36, 40)
MY_OWN = (25, 150, 10)


def get_color(COLOR_TUPLE_255):
    return [x / 255 for x in COLOR_TUPLE_255]


def plot(nr_samples, error_rates,
         title='error rate vs. # of train samples'):
    plt.plot(nr_samples, error_rates, color=get_color(MY_RED))
    plt.title(title)
    plt.xlabel('# of train samples')
    plt.ylabel('Error rate')
    file_name = title.replace(': ', '_').replace(' ', '_')
    file_name = file_name.replace(', ', '_').replace('.', '-')
    # plt.savefig("./iris_" + file_name + "2.pdf")
    plt.savefig("./muscle.pdf")
    plt.clf()
    plt.close()


class LeastSquareClassifier(BaseEstimator, ClassifierMixin):

    def add_ones_short(self, X):
        ones = np.ones(X.shape[0])
        X = np.concatenate((ones[:, np.newaxis], X), axis=1)
        return X

    def fit(self, X, y=None):
        # make the classifier affine
        X = self.add_ones_short(X)
        self.w = find_w(X, y)

    def predict(self, X, y=None):
        X = self.add_ones_short(X)
        return [x for x in np.sign(np.matmul(X, self.w))]

    def score(self, X, y):
        X = self.add_ones_short(X)
        n, _ = X.shape
        y_hat = np.sign(np.matmul(X, self.w))
        score = np.sum(y_hat == y)
        return score / n

    def predict_proba(self, X):
        X = self.add_ones_short(X)
        ys = np.matmul(X, self.w)
        probs = []
        for y in ys:
            if y < 0:
                if y < -1:
                    probs.append([1, 0.0])
                else:
                    y = 0.5 * np.abs(y) + 0.5
                    probs.append([y, 1 - y])
            else:
                if y > 1:
                    probs.append([0.0, 1])
                else:
                    y = 0.5 * np.abs(y) + 0.5
                    probs.append([1 - y, y])
        probs = np.array(probs)
        return probs


classifiers = {
    "Least Squares": LeastSquareClassifier(),
    "Nearest Neighbors": KNeighborsClassifier(3),
    "SVM": SVC(kernel="linear", C=0.025, probability=True),
    "RBF SVM": SVC(gamma=2, C=1, probability=True),
    "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
    "Decision Tree": DecisionTreeClassifier(max_depth=None),
    "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10,
                                            max_features=1),
    "Neural Net": MLPClassifier(alpha=0.01, max_iter=1000),
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis()
}


def find_w_X_more_rows_than_cols(X, y):
    H, W = X.shape
    assert H >= W
    X_t = X.transpose()
    X_t_X = np.matmul(X_t, X)
    X_t_X_inv = np.linalg.inv(X_t_X)
    X_t_X_inv_X_t = np.matmul(X_t_X_inv, X_t)
    w_hat = np.matmul(X_t_X_inv_X_t, y)
    return w_hat


def find_w_X_more_cols_than_rows(X, y):
    H, W = X.shape
    assert H < W
    X_t = X.transpose()
    X_X_t = np.matmul(X, X_t)
    X_X_t_inv = np.linalg.inv(X_X_t)
    X_t_X_X_t_inv = np.matmul(X_t, X_X_t_inv)
    w_hat = np.matmul(X_t_X_X_t_inv, y)
    return w_hat


def find_w_svd(X, y):
    H, W = X.shape
    assert W >= H
    u, s, vh = svd(a=X, full_matrices=False)
    s = 1 / s
    u_v = np.matmul(u * s[..., None, :], vh)
    w = np.matmul(u_v.T, y)
    return w


def find_w(X, y):
    H, W = X.shape
    if H >= W:
        return find_w_X_more_rows_than_cols(X, y)
    else:
        # return find_w_X_more_cols_than_rows(X, y)
        return find_w_svd(X, y)


def take_n_samples_each_clas(X, Y, nr_class, nr_samples_each_class):
    n, _ = X.shape
    n_class = n // nr_class
    x = []
    y = []
    start_index = 0
    end_index = n_class
    # We need to extract samples for each class separately
    # ensure that there are the same number of samples for
    # each class in the train and the validation sets.
    for i in range(nr_class):
        x.append(X[start_index:end_index, ...])
        y.append(Y[start_index:end_index])
        start_index += n_class
        end_index += n_class
        # Randomize the samples within this class.
        # We could also do it after the extraction
        # of the validation set.
        randomized_indices = np.random.choice(
            n_class, nr_samples_each_class, replace=False)
        x[i] = x[i][randomized_indices]
        y[i] = y[i][randomized_indices]

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    return x, y


def cross_validate(X, Y, classifier, cv_count=6, nr_class=2, repeat=3,
                   col_names=None, train_limit=None):
    """
    Cross-validate the model.

    :param X: the input matrix of features
    We expect that the samples for each class are of
    the same number and arranged in the continuous way in
    the input dataset.
    :param Y: the input vector of correct predictions
    :param cv_count: cross validation count
    how many subsets of the data we want, where
    one of the subsets is the validation set
    and the remaining subsets create constitute
    the train set. We have cv_count iterations,
    where each of the cv_count subsets is
    validation set in one of the iterations.
    :param nr_class: number of classes in the dataset
    :param repeat: how many times to repeat the process
    :param is_affine: add the column with all 1's (ones)
    :return: the average accuracy across all repetitions
    and cross-validations within the repetitions.
    """
    n, _ = X.shape
    n_class = n // nr_class
    # number of samples per class
    assert n_class % cv_count == 0
    # length of the validated set from a single class
    cv_len = n_class // cv_count
    all_accuracies = []
    all_aucs = []

    for _ in range(repeat):
        x = []
        y = []
        start_index = 0
        end_index = n_class
        # We need to extract samples for each class separately
        # ensure that there are the same number of samples for
        # each class in the train and the validation sets.
        for i in range(nr_class):
            x.append(X[start_index:end_index, ...])
            y.append(Y[start_index:end_index])
            start_index += n_class
            end_index += n_class
            # Randomize the samples within this class.
            # We could also do it after the extraction
            # of the validation set.
            randomized_indices = np.random.choice(
                n_class, n_class, replace=False)
            x[i] = x[i][randomized_indices, ...]
            y[i] = y[i][randomized_indices]

        # Cross-validate the model cv_count times.
        for i in range(cv_count):
            bottom_index = i * cv_len
            top_index = (i + 1) * cv_len
            bottom_x = []
            top_x = []
            bottom_y = []
            top_y = []

            for j in range(nr_class):
                bottom_x.append(x[j][:bottom_index, :])
                top_x.append(x[j][top_index:, :])
                bottom_y.append(y[j][:bottom_index])
                top_y.append(y[j][top_index:])

            bottom_x = np.concatenate(bottom_x, axis=0)
            top_x = np.concatenate(top_x, axis=0)
            bottom_y = np.concatenate(bottom_y, axis=0)
            top_y = np.concatenate(top_y, axis=0)

            if i == 0:
                x_train = top_x
                y_train = top_y
            elif i == cv_count - 1:
                x_train = bottom_x
                y_train = bottom_y
            else:
                x_train = np.concatenate((bottom_x, top_x), axis=0)
                y_train = np.concatenate((bottom_y, top_y), axis=0)

            if train_limit:
                x_train = x_train[:train_limit, :]
                y_train = y_train[:train_limit]

            x_train, means, stds = normalize_with_nans(x_train)

            x_test = []
            y_test = []
            for j in range(nr_class):
                x_test.append(x[j][bottom_index:top_index, :])
                y_test.append(y[j][bottom_index:top_index])

            x_test = np.concatenate(x_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)

            x_test, _, _ = normalize_with_nans(x_test, means=means, stds=stds,
                                               col_names=col_names)

            clf = classifier
            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)
            # y_score = clf.predict(x_test)
            y_probs = clf.predict_proba(x_test)
            auc = sklearn.metrics.roc_auc_score(y_true=y_test,
                                                y_score=y_probs[:, 1])
            all_aucs.append(auc)
            all_accuracies.append(score)

    return np.average(all_accuracies), np.average(all_aucs)


def err_percent(error_rate):
    return str(100 * error_rate) + " %"


def accuracy_percent(accuracy):
    return str(100 * accuracy) + " %"


def missing_values_col(data, nans, col_names, missing_rate=0.5):
    remove_cols = []
    missing_values_col = []
    for col_nr in range(data.shape[1]):
        col = data[:, col_nr].copy()
        col_clean = col[col != nans]
        nr_missing_values = len(col) - len(col_clean)
        col_name = col_names[col_nr]
        if nr_missing_values >= (missing_rate * len(col)):
            print(f'More than {missing_rate} of the patients have missing '
                  f'value for column number {col_nr} labeled {col_name}')
            remove_cols.append(col_nr)
        missing_values_col.append(nr_missing_values)
    avg_missing_values_per_column = np.average(missing_values_col)
    print('average number of missing values per column: ',
          avg_missing_values_per_column)
    return remove_cols


def missing_values_row(data, nans, missing_rate=0.5):
    missing_values_row = []
    remove_patients = []
    for row_nr in range(data.shape[0]):
        row = data[row_nr, :].copy()
        row_clean = row[row != nans]
        nr_missing_values = len(row) - len(row_clean)
        missing_values_row.append(nr_missing_values)
        if nr_missing_values >= (missing_rate * len(row)):
            print(
                f'{nr_missing_values} (more than {missing_rate * 100}%) of the '
                f'measurements are missing for patient number: {row_nr}')
            remove_patients.append(row_nr)
    avg_missing_values_per_row = np.average(missing_values_row)
    print('average number of missing values per row: ',
          avg_missing_values_per_row)
    return remove_patients


def normalize_with_nans(data, nans=999, means=None, stds=None,
                        col_names=None):
    """
    Normalize the data after setting nans to mean values.
    :param data: the input data
    :param nans: values for non-applied data items
    :param means: the mean values for each feature column
    :param stds: the standard deviations for each feature column
    :return: normalized data

    """
    if means is None and stds is not None:
        raise Exception('Provide also means.')
    if means is not None and stds is None:
        raise Exception('Provide also stds.')

    is_test = True
    if means is None and stds is None:
        is_test = False
        means = []
        stds = []

    for col_nr in range(data.shape[1]):
        col = data[:, col_nr].copy()
        col_clean = col[col != nans]

        if np.count_nonzero(col_clean) == 0:
            message = f'All data elements in column nr: {col_nr} are zero.'
            if col_names is not None:
                message += f' The column name is: {col_names[col_nr]}'
            # print('normalization message: ', message)
            # raise Exception(message)

        if is_test:
            mean = means[col_nr]
            std = stds[col_nr]
        else:
            mean = np.mean(col_clean)
            std = np.std(col_clean)
            means.append(mean)
            stds.append(std)
        # normalize the column
        col[col == nans] = mean
        col -= mean
        if std != 0:
            col /= std
        data[:, col_nr] = col

    return data, means, stds


priority_classifiers = {
    "Least Squares": LeastSquareClassifier(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5)
}


def column_priority(X, y, X_cv, y_cv, labels, classifiers=priority_classifiers):
    w = find_w_svd(X, y)
    w_abs = np.abs(w)
    index_w = [[i, w] for i, w in enumerate(w_abs)]
    # sort in descending order
    sort_index_w = sorted(index_w, key=lambda index_w: [-index_w[1]])
    w_sorted_indexes = [index for (index, _) in sort_index_w]
    for index, w in sort_index_w:
        print(index, ';', labels[index], ';', w)
    # print('sort_index_w: ', sort_index_w)

    print('# of columns', end="")
    classifier_names = classifiers.keys()
    for classifier_name in classifier_names:
        print(delimiter, classifier_name, "accuracy train,", classifier_name,
              ",accuracy cross-validation", end="")
    print()

    for i in range(1, len(w_sorted_indexes) + 1):
        print(i, end="")
        # Extract most important columns from the dataset X.
        column_subset = w_sorted_indexes[:i]
        X_short = X[:, column_subset]
        X_cv_short = X_cv[:, column_subset]
        for clf in classifiers.values():
            clf.fit(X_short, y)
            train_score = clf.score(X_short, y)
            print(delimiter, train_score, end="")
            try:
                cv_score = np.average(
                    cross_val_score(clf, X_cv_short, y_cv, cv=6))
                print(delimiter, cv_score, end="")
            except np.linalg.LinAlgError as err:
                print(delimiter, "N/A", end="")

        print()
    return w_sorted_indexes


def show_decision_tree(estimator, col_names, means, stds):
    # source: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    # The decision estimator has an attribute called tree_  which stores the entire
    # tree structure and allows access to low level attributes. The binary tree
    # tree_ is represented as a number of parallel arrays. The i-th element of each
    # array holds information about the node `i`. Node 0 is the tree's root. NOTE:
    # Some of the arrays only apply to either leaves or split nodes, resp. In this
    # case the values of nodes of the other type are arbitrary!
    #
    # Among those arrays, we have:
    #   - left_child, id of the left child of the node
    #   - right_child, id of the right child of the node
    #   - feature, feature used for splitting the node
    #   - threshold, threshold value at the node
    #

    # Using those arrays, we can parse the tree structure:

    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            feature_nr = feature[i]
            print(
                # "%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                # "node %s."
                "%snode=%s test node: go to node %s if '%s' <= %s else to "
                "node %s."
                % (node_depth[i] * "\t",
                   i,
                   children_left[i],
                   # feature[i],
                   col_names[feature_nr],
                   # threshold[i],
                   threshold[i] * stds[feature_nr] + means[feature_nr],
                   children_right[i],
                   ))


# Utility function to report best scores:
# source: https://scikit-learn.org/stable/auto_examples/model_selection/
# plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot
# randomized-search-py
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def run_param_search(X, y, clf=SVC(probability=True)):
    # specify parameters and distributions to sample from
    param_dist = {'C': scipy.stats.expon(scale=100),
                  'gamma': scipy.stats.expon(scale=.1),
                  'kernel': ['rbf', 'linear'],
                  'class_weight': ['balanced', None]}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5, iid=False)

    start = time.time()
    random_search.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter_search))
    report(random_search.cv_results_)


def show_param_performance(X_cv, y_cv, nr_class, col_names):
    print(
        'Accuracy on self-crafted cross-validation with normalization: ')
    for C in np.linspace(start=0.0001, stop=200, num=100):
        for name, clf in [('SVM', SVC(kernel="linear", C=C, probability=True))]:
            accuracy, auc = cross_validate(X_cv, y_cv, classifier=clf,
                                           nr_class=nr_class,
                                           col_names=col_names)
            print(name, "C=", C, delimiter, accuracy_percent(accuracy), auc)
    print()


def svd_spectrum(X):
    u, s, vh = svd(a=X, full_matrices=False)
    print("Rank of X: ", len(s))
    s = [x ** 2 for x in s]
    sum_s = np.sum(s)
    s = [x / sum_s for x in s]
    print("Importance of singular values: ", s)
    print("length of singular values: ", len(s))
    ax1 = plt.subplot(111)
    ax1.plot(
        range(len(s)), s, label="$\frac{\sigma_i^2}{\sum_j^N \sigma_j^2}$",
        marker="o", linestyle="", color=get_color(MY_BLUE))
    ax1.set_title("Spectrum of X")
    ax1.set_xlabel("index i of $\sigma_i$")
    # ax1.set_ylabel("$\sigma_i^2$/\n$\sum_j^N \sigma_j^2$", rotation=0)
    ax1.set_ylabel("$\sigma_i^2$", rotation=0)
    # ax1.legend(["true rating", "predicted rating"], loc="upper left")
    # ax1.axis([0, num_train, -15, 10])
    # plt.show()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.join(dir_path, "svd_spectrum.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def pca(X, index):
    """
    Compute PCA for X.

    :param X: the whole input data
    :param index: how many singular values retain
    (the dimension of the subspace)
    :return: the projected rows of X on a lower dimensional space
    """
    XT = X.T  # samples in columns
    u, s, vh = svd(a=XT, full_matrices=False)
    # We want to project the samples on a lower dimensional space.
    u = u[:, :index]
    # Columns of z are the new lower dimensional coordinates for the intial samples.
    z = np.matmul(X, u)
    return z


def pca_scikit_whole_train(X, index):
    pca = PCA(n_components=index)  # adjust yourself
    pca.fit(X)
    z = pca.transform(X)
    return z


def pca_scikit_train_test(X, y, clf):
    print("PCA train test from scikit learn:")
    print("index, score")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)
    for index in range(1, X_test.shape[0]):
        pca = PCA(n_components=index)  # adjust yourself
        pca.fit(X_train)
        X_t_train = pca.transform(X_train)
        X_t_test = pca.transform(X_test)
        clf.fit(X_t_train, y_train)
        print(index, ',', clf.score(X_t_test, y_test))


def pca_scikit_train(X, y, clf):
    print("PCA train from scikit learn:")
    print("index, score")
    for index in range(1, X.shape[0]):
        pca = PCA(n_components=index)  # adjust yourself
        pca.fit(X)
        xpca = pca.transform(X)
        clf.fit(xpca, y)
        print(index, ',', clf.score(xpca, y))


def accuracy_on_pca_data(X, y, classifiers, nr_class, col_names,
                         pca_function=pca):
    H, W = X.shape
    print('PCA:')
    print('index' + delimiter + 'accuracy' + delimiter + 'auc')
    for index in range(1, H):
        xpca = pca_function(X, index=index)
        for name, clf in classifiers.items():
            accuracy, auc = cross_validate(
                X=xpca, Y=y, classifier=clf, nr_class=nr_class,
                col_names=col_names)
            result = [index, accuracy, auc]
            str_result = delimiter.join([str(x) for x in result])
            print(str_result)


def findsubsets(s, max=None):
    if max is None:
        n = len(s) + 1
    else:
        n = max + 1
    subsets = []
    for i in range(1, n):
        subset = list(itertools.combinations(s, i))
        for x in subset:
            subsets.append(x)
    return subsets


def col_scores_single_thread(X, y, clf, nr_class, col_names, max_col_nr):
    H, W = X.shape
    col_scores = {}
    for col in range(W):
        col_scores[col] = 0

    subsets = findsubsets(np.arange(W), max_col_nr)
    print('subsets count: ', len(subsets))
    for set in subsets:
        Xset = X[:, set]
        accuracy, auc = cross_validate(Xset, y, classifier=clf,
                                       nr_class=nr_class, col_names=col_names)
        for col in set:
            col_scores[col] += accuracy

    for w in sorted(col_scores, key=col_scores.get, reverse=True):
        result = [w, col_names[w], col_scores[w]]
        result_str = delimiter.join([str(x) for x in result])
        print(result_str)
    return col_scores


col_scores = {}


def get_accuracy(X, set, y, clf, nr_class, col_names):
    Xset = X[:, set]
    accuracy, _ = cross_validate(Xset, y, classifier=clf,
                                 nr_class=nr_class, col_names=col_names)
    return set, accuracy


def collect_accuracies(result):
    set, accuracy = result
    for col in set:
        col_scores[col] += accuracy


def col_scores_parallel(X, y, clf, nr_class, col_names, max_col_nr):
    H, W = X.shape
    global col_scores
    for col in range(W):
        col_scores[col] = 0

    subsets = findsubsets(np.arange(W), max_col_nr)
    print('subsets count: ', len(subsets))

    # parallel python
    # source:
    # https://www.machinelearningplus.com/python/parallel-processing-python/
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())

    # Step 2: pool.apply to a function
    for set in subsets:
        pool.apply_async(
            get_accuracy,
            args=(X, set, y, clf, nr_class, col_names),
            callback=collect_accuracies
        )

    # Step 3: close the pool
    pool.close()

    # Step 4: postpones the execution of next line of code until all
    # processes in the queue are done.
    pool.join()

    for w in sorted(col_scores, key=col_scores.get, reverse=True):
        result = [w, col_names[w], col_scores[w]]
        result_str = delimiter.join([str(x) for x in result])
        print(result_str)
    sys.stdout.flush()
    return col_scores_single_thread


def compute():
    warnings.filterwarnings("ignore")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # data_path = os.path.join(dir_path, "remy_data_all.csv")
    # data_path = os.path.join(dir_path, "remy_data_cleaned_with_header.csv")
    # data_path = os.path.join(dir_path, "remy_data_final.csv")
    # data_path = os.path.join(dir_path, "remy_data_final_sign_class.csv")
    # data_path = os.path.join(dir_path, "clean-2019-11-24-3.csv")

    # data_path = os.path.join(dir_path, "remy_2019_10_29.csv")
    # data_path = os.path.join(dir_path, "arnold_2019_12_07.csv")
    data_path = os.path.join(dir_path, "garrett_2019_11_24.csv")

    print('data_path: ', data_path)
    data_all = pd.read_csv(data_path, header=0)
    labels = np.asarray(data_all.iloc[:, 0], dtype=np.int)
    nr_class = len(np.unique(labels))
    X = np.asarray(data_all.iloc[:, 1:], dtype=np.float)
    y = labels
    row_nr, col_nr = X.shape
    assert len(y) == row_nr
    col_names = np.array(list(data_all.columns.values))
    col_names = col_names[1:]  # skip the ASD column name
    assert len(col_names) == col_nr
    # print('X: ', X)
    # print('y: ', y)
    # print('size of X: ', X.shape)
    # print('size of y: ', y.shape)

    print('row number: ', row_nr)
    print('column number: ', col_nr)

    # remove the dependent columns
    # Q, R = qr(a=X, mode='reduced')

    # print('descriptive statistics for X: ', describe(X))

    # print('X affine: ', X)

    # remove column with all zeros
    # print('columns with all zeros: ', np.where(~X_norm.any(axis=0))[0])

    nans = 999

    """
    Special case:
    Column: “Asymmetry  Total CSA > 12% at C3” – it has only zero values or 
    ‘999’s only (it is the 3rd column from the end).
    """
    # X = np.delete(X, -3, axis=1)
    # col_names = np.delete(col_names, -3)

    remove_cols = missing_values_col(data=X, nans=nans, col_names=col_names)
    remove_rows = missing_values_row(data=X, nans=nans)

    print('Delete columns: ', remove_cols)
    X = np.delete(X, remove_cols, axis=1)
    col_names = np.delete(col_names, remove_cols)

    print('Delete rows: ', remove_rows)
    X = np.delete(X, remove_rows, axis=0)
    y = np.delete(y, remove_rows)

    X_norm, means, stds = normalize_with_nans(data=X.copy(), nans=nans)
    # print('means: ', means)
    # print('stds: ', stds)

    svd_spectrum(X)

    w_hat = find_w(X_norm, y)
    y_hat = np.sign(np.matmul(X_norm, w_hat))
    # print("check y_hat: ", y_hat)
    diff = np.sum(y_hat == y)
    accuracy = diff / len(y)
    print('On the whole data: ')
    print('Full Least Squares accuracy: ',
          accuracy_percent(accuracy))

    # for cross validation we take the same number of samples for each class
    X_cv = np.concatenate((X[:30, :], X[31:61, :]))
    y_cv = np.concatenate((y[:30], y[31:61]))

    SVM = classifiers["SVM"]

    max_col_nr = 3

    start = time.time()
    col_scores_parallel(
        X=X_cv, y=y_cv, col_names=col_names, clf=SVM, nr_class=nr_class,
        max_col_nr=max_col_nr)
    print('col scores parallel timing: ', time.time() - start)
    sys.stdout.flush()

    # start = time.time()
    # col_scores_single_thread(
    #     X=X_cv, y=y_cv, col_names=col_names, clf=SVM, nr_class=nr_class,
    #     max_col_nr=max_col_nr)
    # print('col scores single timing: ', time.time() - start)

    # pca_scikit_train_test(X=X_cv, y=y_cv, clf=SVM)
    pca_scikit_train(X=X_cv, y=y_cv, clf=SVM)

    # pca_function=pca
    pca_function = pca_scikit_whole_train
    accuracy_on_pca_data(X=X_cv, y=y_cv, classifiers={"SVM": SVM},
                         nr_class=nr_class, col_names=col_names,
                         pca_function=pca_function)

    # run_param_search(X=X_cv, y=y_cv, clf=SVC(probability=True))

    # show_param_performance(X_cv=X_cv, y_cv=y_cv, nr_class=nr_class,
    #                        col_names=col_names)

    print("Column priority: ")
    w_sorted_indexes = column_priority(X_norm, y, X_cv, y_cv, labels=col_names)

    print('labels len: ', len(col_names))
    print('w_hat len: ', len(w_hat))

    ones = np.ones(X_norm.shape[0])
    X_ones = np.concatenate((ones[:, np.newaxis], X_norm), axis=1)
    w_hat = find_w_svd(X_ones, y)
    y_hat = np.sign(np.matmul(X_ones, w_hat))
    # print("check y_hat: ", y_hat)
    diff = np.sum(y_hat == y)
    accuracy = diff / len(y)
    print('Least Squares accuracy: ', accuracy_percent(accuracy))

    clf = LeastSquareClassifier()
    clf.fit(X_norm, y)
    score = clf.score(X_norm, y)
    print('Least Squares accuracy: ', accuracy_percent(score))

    clf = classifiers['Neural Net']
    clf.fit(X_norm, y)
    score = clf.score(X_norm, y)
    print('Neural net accuracy: ', accuracy_percent(score))

    clf = classifiers['Decision Tree']
    clf.fit(X_norm, y)
    score = clf.score(X_norm, y)
    print('Decision Tree accuracy: ', accuracy_percent(score))
    show_decision_tree(estimator=clf, col_names=col_names, means=means,
                       stds=stds)

    print('Accuracy on normalized X_norm: ')
    for name, clf in classifiers.items():
        clf.fit(X_norm, y)
        score = clf.score(X_norm, y)
        print(name, accuracy_percent(score))

    col_subset = 32
    print(
        f'Accuracy and AUC on self-crafted cross-validation with normalization '
        f'and subset of {col_subset} columns: ')
    X_subset = X_cv[:, w_sorted_indexes[:col_subset]]
    for name, clf in classifiers.items():
        accuracy, auc = cross_validate(X_subset, y_cv, classifier=clf,
                                       nr_class=nr_class, col_names=col_names)
        print(name, delimiter, accuracy_percent(accuracy), delimiter, auc)
    print()

    print('Accuracy from cross-validation (non-normalized data): ')
    for name, clf in classifiers.items():
        accuracy = np.average(cross_val_score(clf, X_cv, y_cv, cv=6))
        print(name, delimiter, accuracy_percent(accuracy))
    print()

    X_norm2 = np.concatenate((X_norm[:30, :], X_norm[31:61, :]))
    print('Accuracy from cross-validation (normalized the whole): ')
    for name, clf in classifiers.items():
        accuracy = np.average(cross_val_score(clf, X_norm2, y_cv, cv=6))
        print(name, delimiter, accuracy_percent(accuracy))
    print()

    print(
        'Accuracy and AUC on self-crafted cross-validation with normalization: ')
    print("model name, accuracy (%), AUC")
    for name, clf in classifiers.items():
        accuracy, auc = cross_validate(X_cv, y_cv, classifier=clf,
                                       nr_class=nr_class, col_names=col_names)
        print(name, delimiter, accuracy_percent(accuracy), delimiter, auc)
    print()


if __name__ == "__main__":
    compute()
