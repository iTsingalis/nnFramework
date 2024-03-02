"""
Sample code automatically generated on 2024-01-09 10:59:17

by geno from www.geno-project.org

from input

parameters
  matrix X
  matrix T
variables
  matrix W
min
  sum((-T).*(W'*X))+sum(log(exp(X'*W)*vector(1)))
st
  W >= 0
  W'*vector(1) == vector(1)


The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np
from math import inf
from timeit import default_timer as timer

import os
import pickle
import argparse
import seaborn as sns
from mnist import MNIST
import matplotlib.pyplot as plt

from Utils.timer import Timer
from Utils.utils import check_folder
from Utils.mnist_reader import load_mnist

from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


try:
    from genosolver import minimize, check_version

    USE_GENO_SOLVER = True
except ImportError:
    from scipy.optimize import minimize

    USE_GENO_SOLVER = False
    WRN = 'WARNING: GENO solver not installed. Using SciPy solver instead.\n' + \
          'Run:     pip install genosolver'
    print('*' * 63)
    print(WRN)
    print('*' * 63)


class GenoNLP:
    def __init__(self, X, T, np):
        self.np = np
        self.X = X
        self.T = T
        assert isinstance(X, self.np.ndarray)
        dim = X.shape
        assert len(dim) == 2
        self.X_rows = dim[0]
        self.X_cols = dim[1]
        assert isinstance(T, self.np.ndarray)
        dim = T.shape
        assert len(dim) == 2
        self.T_rows = dim[0]
        self.T_cols = dim[1]
        self.W_rows = self.X_rows
        self.W_cols = self.T_rows
        self.W_size = self.W_rows * self.W_cols
        # the following dim assertions need to hold for this problem
        assert self.T_rows == self.W_cols
        assert self.X_cols == self.T_cols
        assert self.W_rows == self.X_rows

    def getLowerBounds(self):
        bounds = []
        bounds += [0] * self.W_size
        return self.np.array(bounds)

    def getUpperBounds(self):
        bounds = []
        bounds += [inf] * self.W_size
        return self.np.array(bounds)

    def getStartingPoint(self):
        self.WInit = self.np.zeros((self.W_rows, self.W_cols))
        return self.WInit.reshape(-1)

    def variables(self, _x):
        W = _x
        W = W.reshape(self.W_rows, self.W_cols)
        return W

    def fAndG(self, _x):
        W = self.variables(_x)
        T_0 = self.np.exp((self.X.T).dot(W))
        t_1 = (T_0).dot(self.np.ones(self.W_cols))
        f_ = (self.np.sum(self.np.log(t_1)) - self.np.sum(
            ((self.T * (W.T).dot(self.X))).dot(self.np.ones(self.X_cols))))
        g_0 = ((self.X).dot(((self.np.ones(self.X_cols) / t_1)[:, self.np.newaxis] * T_0)) - (self.X).dot(self.T.T))
        g_ = g_0.reshape(-1)
        return f_, g_

    def functionValueEqConstraint000(self, _x):
        W = self.variables(_x)
        f = ((W.T).dot(self.np.ones(self.W_rows)) - self.np.ones(self.W_cols))
        return f

    def jacProdEqConstraint000(self, _x, _v):
        W = self.variables(_x)
        gv_ = (self.np.outer(self.np.ones(self.W_rows), _v)).reshape(-1)
        return gv_


def toArray(v):
    return np.ascontiguousarray(v, dtype=np.float64).reshape(-1)


def solve(X, T, np):
    start = timer()
    NLP = GenoNLP(X, T, np)
    x0 = NLP.getStartingPoint()
    lb = NLP.getLowerBounds()
    ub = NLP.getUpperBounds()
    # These are the standard solver options, they can be omitted.
    options = {'eps_pg': 1E-4,
               'constraint_tol': 1E-4,
               'max_iter': 2,
               'm': 10,
               'ls': 0,
               'verbose': 5  # Set it to 0 to fully mute it.
               }

    if USE_GENO_SOLVER:
        # Check if installed GENO solver version is sufficient.
        check_version('0.1.0')
        constraints = ({'type': 'eq',
                        'fun': NLP.functionValueEqConstraint000,
                        'jacprod': NLP.jacProdEqConstraint000})
        result = minimize(NLP.fAndG, x0, lb=lb, ub=ub, options=options,
                          constraints=constraints, np=np)
    else:
        constraints = ({'type': 'eq',
                        'fun': NLP.functionValueEqConstraint000})
        result = minimize(NLP.fAndG, x0, jac=True, method='SLSQP',
                          bounds=list(zip(lb, ub)),
                          constraints=constraints)

    # assemble solution and map back to original problem
    elapsed = timer() - start

    # assemble solution and map back to original problem
    x = result.x
    eqConstraint000 = NLP.functionValueEqConstraint000(x)
    W = NLP.variables(x)

    solution = {}
    solution['success'] = result.success
    solution['message'] = result.message
    solution['fun'] = result.fun
    solution['grad'] = result.jac
    if USE_GENO_SOLVER:
        solution['slack'] = result.slack
    solution['W'] = W
    solution['eqConstraint000'] = toArray(eqConstraint000)
    solution['elapsed'] = elapsed
    return solution


def predict(X, W):
    _activations = X @ W

    _activations_max = np.max(_activations, axis=1)
    _activations -= _activations_max[:, None]

    sigma = np.exp(_activations) / np.sum(np.exp(_activations), axis=1, keepdims=True)

    y_pred = np.argmax(sigma, axis=1)

    return y_pred


def plot_gallery(title, images, n_col, n_row, image_shape, cmap=plt.cm.gray, save_path=None):
    fig = plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    fig.suptitle(title)

    # fig.canvas.set_window_title(title)

    for i, comp in enumerate(images[:n_row * n_col]):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=cmap,
                   interpolation='nearest',
                   # vmin=-vmax, vmax=vmax
                   )
        plt.xticks(())
        plt.yticks(())

    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'coef_.eps'), format='eps',
                    # bbox_extra_artists=(lgd,),
                    dpi=100, bbox_inches='tight')


def main():
    acc_list = []
    exp_folder_num = check_folder('exp', FLAGS.task_name)

    print('solving..')
    timer = Timer()
    timer.start()
    solution = solve(X_tr.T, labels_tr.T, np=np)
    timer.stop()

    print('*' * 5, 'solution', '*' * 5)
    print(solution['message'])
    if solution['success']:
        print('optimal function value   = ', solution['fun'])
        print('norm of the gradient     = ',
              np.linalg.norm(solution['grad'], np.inf))
        # print('optimal variable w = ', solution['w'])
        print('solving took %.3f sec' % solution['elapsed'])

    print('Elapsed time (sec) {}'.format(solution['elapsed']))

    W = solution["W"]
    pred_labels_tst = predict(X_tst, W)

    acc_ = np.mean(pred_labels_tst == labels_tst)
    acc_list.append(acc_)
    print('Acc. {}'.format(acc_))

    cm = confusion_matrix(labels_tst, pred_labels_tst)

    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True,
                linewidths=.5, square=True, cmap='Blues_r', fmt='0.4g')

    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    plt.savefig(os.path.join(exp_folder_num, 'cm_{0:.5f}.eps'.format(acc_)), format='eps',
                dpi=100, bbox_inches='tight')

    plot_gallery(title="coef_", images=W.T, n_col=10, n_row=1,
                 image_shape=image_shape, save_path=exp_folder_num)

    with open(os.path.join(exp_folder_num, 'components.pkl'), 'wb') as f:
        pickle.dump(W, f)

    # plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ### Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train")

    FLAGS = parser.parse_args()

    if 'mnist' in FLAGS.task_name:

        DATASET = 'mnist'
        image_shape = (28, 28)
        mndata = MNIST('../../Datasets/mnist')

        images_tr, labels_tr = mndata.load_training()
        images_tst, labels_tst = mndata.load_testing()

        X_tr = np.asarray(images_tr).astype(np.float64) / 255.
        labels_tr = np.asarray(labels_tr).astype(np.int32)
        X_tr, labels_tr = shuffle(X_tr, labels_tr, random_state=0)

        # X_tr = X_tr[:1000]
        # labels_tr = labels_tr[:1000]

        X_tst = np.asarray(images_tst).astype(np.float64) / 255.
        labels_tst = np.asarray(labels_tst).astype(np.int32)
        X_tst, labels_tst = shuffle(X_tst, labels_tst, random_state=0)

        tr_n_samples, _ = X_tr.shape
        tst_n_samples, _ = X_tst.shape
        h, w = image_shape
        n_classes = len(np.unique(labels_tr))
        n_features = h * w

        print("Total dataset: n_tr_samples {}, n_tst_samples {}, n_features (h {} x w {}) {}, n_classes {}"
              .format(tr_n_samples, tst_n_samples, h, w, h * w, n_classes))

        # # scaler = MinMaxScaler(feature_range=(0, 1))
        # scaler = StandardScaler(with_std=False)
        # X_tr = scaler.fit_transform(X_tr)
        # X_tst = scaler.transform(X_tst)

    elif 'fashion' in FLAGS.task_name:

        from sklearn.utils import shuffle

        DATASET = 'fashion'
        image_shape = (28, 28)

        images_tr, labels_tr = load_mnist('../../Datasets/fashion', perc_samples=1., kind='train')
        images_tst, labels_tst = load_mnist('../../Datasets/fashion', perc_samples=1., kind='t10k')

        X_tr = np.asarray(images_tr).astype(np.float64) / 255.
        # max_X_tr = np.max(X_tr)
        # X_tr /= max_X_tr
        labels_tr = np.asarray(labels_tr).astype(np.int32)
        X_tr, labels_tr = shuffle(X_tr, labels_tr, random_state=0)

        X_tst = np.asarray(images_tst).astype(np.float64) / 255.
        # X_tst /= max_X_tr
        labels_tst = np.asarray(labels_tst).astype(np.int32)
        X_tst, labels_tst = shuffle(X_tst, labels_tst, random_state=0)

        n_classes = len(np.unique(labels_tr))

        tr_n_samples, _ = X_tr.shape
        tst_n_samples, _ = X_tst.shape
        h, w = image_shape
        n_classes = len(np.unique(labels_tr))
        n_features = h * w

        print("Total dataset: n_tr_samples {}, n_tst_samples {}, n_features (h {} x w {}) {}, n_classes {}"
              .format(tr_n_samples, tst_n_samples, h, w, h * w, n_classes))

    else:
        raise ValueError('task_name is not specified.')

    # # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = StandardScaler(with_std=False)
    X_tr = scaler.fit_transform(X_tr)
    X_tst = scaler.transform(X_tst)

    lbin = preprocessing.LabelBinarizer()
    labels_tr = lbin.fit_transform(labels_tr)
    # labels_tst = lbin.fit_transform(labels_tst)

    main()
