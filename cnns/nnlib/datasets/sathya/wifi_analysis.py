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


class LeastSquareClassifierWithOnes(BaseEstimator, ClassifierMixin):

    def add_ones_short(self, X):
        ones = np.ones(X.shape[0])
        X = np.concatenate((ones[:, np.newaxis], X), axis=1)
        return X

    def fit(self, X, y=None):
        # make the classifier affine (add columns with ones)
        X = self.add_ones_short(X)
        self.w = find_w(X, y)

    def predict(self, X, y=None):
        X = self.add_ones_short(X)
        return [x for x in np.rint(np.matmul(X, self.w))]

    def score(self, X, y):
        X = self.add_ones_short(X)
        n, _ = X.shape
        y_hat = np.rint(np.matmul(X, self.w))
        score = np.sum(y_hat == y)
        return score / n


classifiers = {
    "AdaBoost": AdaBoostClassifier(),
    "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10,
                                            max_features=1),
    "Decision Tree": DecisionTreeClassifier(max_depth=None),
    "FC Neural Net": MLPClassifier(alpha=0.01, max_iter=1000),
    "SVM": SVC(kernel="linear", C=0.025, probability=True),
    "RBF SVM": SVC(gamma=2, C=1, probability=True),
    "Least Squares (with ones)": LeastSquareClassifierWithOnes(),
    "Nearest Neighbors": KNeighborsClassifier(3),
    "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
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


def take_n_samples_each_class(X, Y, nr_class, nr_samples_each_class):
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
    return str(100 * error_rate) + "%"


def accuracy_percent(accuracy):
    return str(100 * accuracy) + "%"


def missing_values_col(data, nans, col_names, missing_rate=0.5):
    remove_cols = []
    missing_values_col = []
    for col_nr in range(data.shape[1]):
        col = data[:, col_nr].copy()
        col_clean = col[col != nans]
        nr_missing_values = len(col) - len(col_clean)
        col_name = col_names[col_nr]
        if nr_missing_values >= (missing_rate * len(col)):
            print(f'More than {missing_rate} of the row have missing '
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


def normalize_with_nans(data, nans='N/A', means=None, stds=None,
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


def limit_data_rows(X, y, nr_class, class_labels, limit_row_nr):
    """
    Limit number of rows representing a given label.

    :param X: the whole dataset with values
    :param y: labels
    :param nr_class: number of classes
    :param class_labels: instances of class labels, e.g., ['apple', 'carrot']
    :param limit_row_nr: how many rows we expect in the output dataset
    :return: the limited dataset
    """
    limit = limit_row_nr // nr_class
    X_new, y_new = None, None
    for class_label in class_labels:
        index = (y == class_label)
        # extract rows for this class label
        X_add = X[index]
        y_add = y[index]
        # limit the number of rows
        X_add = X_add[:limit]
        y_add = y_add[:limit]
        if X_new is None:
            X_new, y_new = X_add, y_add
        else:
            # concatenate the data for this class label with X_new, y_new
            X_new = np.concatenate((X_new, X_add))
            y_new = np.concatenate((y_new, y_add))
    return X_new, y_new


def get_data(data_path, limit_row_nr=1000):
    print('data_path: ', data_path)
    data_all = pd.read_csv(data_path, header=0)
    # extract first column
    labels = np.asarray(data_all.iloc[:, 0], dtype=np.int)
    class_labels = np.unique(labels)
    nr_class = len(class_labels)
    print('number of classes: ', nr_class)
    # skip first column
    X = np.asarray(data_all.iloc[:, 1:], dtype=np.float)
    y = labels
    row_nr, col_nr = X.shape
    assert len(y) == row_nr
    # print('size of X: ', X.shape)
    # print('size of y: ', y.shape)

    print('row number: ', row_nr)
    print('column number: ', col_nr)

    if limit_row_nr is not None and limit_row_nr > 0:
        X, y = limit_data_rows(X=X, y=y, nr_class=nr_class,
                               class_labels=class_labels,
                               limit_row_nr=limit_row_nr)
    return X, y


def basic_least_squares(X_train, y_train, X_test, y_test):
    w_hat = find_w(X_train, y_train)

    y_hat_train = np.rint(np.matmul(X_train, w_hat))
    diff_train = np.sum(y_hat_train == y_train)
    train_accuracy = diff_train / len(y_train)

    y_hat_test = np.rint(np.matmul(X_test, w_hat))
    diff_test = np.sum(y_hat_test == y_test)
    test_accuracy = diff_test / len(y_test)

    print('On the whole data: ')
    print('Least Squares (without column of ones)')
    print('train accuracy, test accuracy')
    print(train_accuracy, ',', test_accuracy)


def compute():
    warnings.filterwarnings("ignore")
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # wifi_path = "data_journal/NLOS-6/2_classes_WIFI/2_classes_WIFI"
    wifi_path = "data_journal/NLOS-6/3_classes_WIFI/3_classes_WIFI"

    data_path = os.path.join(dir_path, wifi_path)
    limit_row_nr = None
    X_train, y_train = get_data(data_path=data_path + '_TRAIN',
                                limit_row_nr=limit_row_nr)
    X_test, y_test = get_data(data_path=data_path + '_TEST',
                              limit_row_nr=limit_row_nr)

    # basic_least_squares(X_train=X_train, y_train=y_train,
    #                     X_test=X_test, y_test=y_test)

    print('Classifier, train accuracy, test accuracy')
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        score_train = clf.score(X_train, y_train)
        score_test = clf.score(X_test, y_test)
        print(name, ',',
              score_train, ',',
              score_test)
        sys.stdout.flush()

if __name__ == "__main__":
    compute()
