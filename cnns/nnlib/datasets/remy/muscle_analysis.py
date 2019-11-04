import numpy as np
from scipy.stats import describe
import os
import matplotlib
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import qr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

h = .02  # step size in the mesh

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


classifiers = {
    "Nearest Neighbors": KNeighborsClassifier(3),
    "SVM": SVC(kernel="linear", C=0.025),
    "RBF SVM": SVC(gamma=2, C=1),
    "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10,
                                            max_features=1),
    "Neural Net": MLPClassifier(alpha=0.01, max_iter=1000),
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis()}


def find_w(X, y):
    X_t = X.transpose()
    X_t_X = np.matmul(X_t, X)
    X_t_X_inv = np.linalg.inv(X_t_X)
    X_t_X_inv_X_t = np.matmul(X_t_X_inv, X_t)
    w_hat = np.matmul(X_t_X_inv_X_t, y)
    return w_hat


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
                   train_limit=None, is_affine=True):
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

            x_test, _, _ = normalize_with_nans(x_test, means=means, stds=stds)

            clf = classifier
            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)
            all_accuracies.append(score)

    return np.average(all_accuracies)


class LeastSquareClassifier(BaseEstimator, ClassifierMixin):

    def add_ones_short(self, X):
        ones = np.ones(X.shape[0])
        X_short = np.concatenate((ones[:, np.newaxis], X), axis=1)
        X_short = X_short[:, :X_short.shape[0]]
        return X_short

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


def err_percent(error_rate):
    return str(100 * error_rate) + " %"


def accuracy_percent(accuracy):
    return str(100 * accuracy) + " %"


def normalize_with_nans(data, nans=999, means=None, stds=None):
    """
    Normalize the data after setting nans to mean values.
    :param data: the input data
    :param nans: values for non-applied data items
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
        col = col[col != nans]
        if is_test:
            mean = means[col_nr]
            std = stds[col_nr]
        else:
            mean = np.mean(col)
            std = np.std(col)
            means.append(mean)
            stds.append(std)
        # normalize the column
        col = data[:, col_nr]
        col[col == nans] = mean
        data[:, col_nr] = (col - mean) / std
    return data, means, stds


def compute():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # data_path = os.path.join(dir_path, "remy_data_all.csv")
    # data_path = os.path.join(dir_path, "remy_data_cleaned_with_header.csv")
    data_path = os.path.join(dir_path, "remy_data_final.csv")
    data_all = pd.read_csv(data_path, header=0)
    labels = np.asarray(data_all.iloc[:, 0], dtype=np.int)
    nr_class = len(np.unique(labels))
    X = np.asarray(data_all.iloc[:, 1:], dtype=np.float)
    y = labels
    # print('X: ', X)
    # print('y: ', y)
    print('size of X: ', X.shape)
    print('size of y: ', y.shape)

    # remove the dependent columns
    # Q, R = qr(a=X, mode='reduced')

    # print('descriptive statistics for X: ', describe(X))

    # print('X affine: ', X)

    # remove column with all zeros
    # print('columns with all zeros: ', np.where(~X_norm.any(axis=0))[0])
    X = np.delete(X, -3, 1)

    X_norm, means, stds = normalize_with_nans(data=X.copy(), nans=999)
    # print('means: ', means)
    # print('stds: ', stds)

    ones = np.ones(X_norm.shape[0])
    X_short = np.concatenate((ones[:, np.newaxis], X_norm), axis=1)
    X_short = X_short[:, :X_short.shape[0]]
    w_hat = find_w(X_short, y)
    y_hat = np.rint(np.matmul(X_short, w_hat))
    # print("check y_hat: ", y_hat)
    diff = np.sum(y_hat == y)
    accuracy = diff / len(y)
    print('On the whole data: ')
    print('Least Squares accuracy: ',
          accuracy_percent(accuracy))

    clf = LeastSquareClassifier()
    clf.fit(X_norm, y)
    score = clf.score(X_norm, y)
    print('Least Squares accuracy: ', accuracy_percent(score))

    clf = classifiers['Neural Net']
    clf.fit(X_norm, y)
    score = clf.score(X_norm, y)
    print('Neural net accuracy: ', accuracy_percent(score))

    for name, clf in classifiers.items():
        clf.fit(X_norm, y)
        score = clf.score(X_norm, y)
        print(name, accuracy_percent(score))

    # for cross validation we take the same number of samples for each class
    X = np.concatenate((X[:30, :], X[31:61, :]))
    y = np.concatenate((y[:30], y[31:61]))

    print('Accuracy from cross-validation (non-normalized data): ')
    for name, clf in classifiers.items():
        accuracy = np.average(cross_val_score(clf, X, y, cv=6))
        print(name, accuracy_percent(accuracy))
    print()

    X_norm2 = np.concatenate((X_norm[:30, :], X_norm[31:61, :]))
    print('Accuracy from cross-validation (normalized the whole): ')
    for name, clf in classifiers.items():
        accuracy = np.average(cross_val_score(clf, X_norm2, y, cv=6))
        print(name, accuracy_percent(accuracy))
    print()

    print('Accuracy on self-crafted cross-validation with normalization: ')
    for name, clf in classifiers.items():
        accuracy = cross_validate(X, y, classifier=clf,
                                  nr_class=nr_class, is_affine=False)
        print(name, accuracy_percent(accuracy))
    print()


if __name__ == "__main__":
    compute()
