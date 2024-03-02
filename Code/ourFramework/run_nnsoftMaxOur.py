import os
import json
import pickle
import argparse
import numpy as np
import seaborn as sns
from mnist import MNIST

from Utils.CLR import CLR
from Utils.timer import Timer
from Utils.utils import check_folder
from Utils.mnist_reader import load_mnist

import matplotlib.pyplot as plt
from nnSoftMax import nnMultiLogReg

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

linestyles = ['-', '-.', '--', ':', '-', '-.', '--', ':', '-']
markers = ['*', '', '', '', 's', 'v', '|', 'p', 'X']
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "k", "c", 'y', 'm', 'b']
markersizes = [1, 1, 1, 1, 1.5, 2.5]

rng = np.random.RandomState(42)


def plot_gallery(exp_folder_num, title, images, n_col, n_row, image_shape, cmap=plt.cm.gray, save=False):
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

    if save:
        plt.savefig(os.path.join(exp_folder_num, 'components.eps'), format='eps',
                    # bbox_extra_artists=(lgd,),
                    dpi=100, bbox_inches='tight')


def plt_clr(clr_args, num_iterations=2000):
    lr_fun = CLR(**clr_args).get_lr_fun()

    lr_trend = list()
    for iteration in range(num_iterations):
        lr = lr_fun(iteration=iteration)
        lr_trend.append(lr)

    plt.plot(lr_trend, '.-')
    plt.grid()

    # plt.savefig('logs/clr.eps'.format(FLAGS.task_name), format='eps',
    #             # bbox_extra_artists=(lgd,),
    #             dpi=100, bbox_inches='tight')

    plt.show()


def main():
    exp_folder_num = check_folder('exp', FLAGS.task_name)

    constr = 'l1'
    if constr == 'l2':
        clr_args = {'clr_type': 'exp_drop', 'step_size': 1000, 'decay': 0.96, 'base_lr': 1e-3}
    else:
        clr_args = {'clr_type': 'fixed_step', 'base_lr': 1e-4}

    tb_args = {'board': True, 'image_shape': image_shape, 'logdir': exp_folder_num}

    par_dict = {'beta': 0.0,
                'atol': 1e-10,
                'reltol': 1e-6,
                'maxit': 3000,
                'constr': 'l1',
                'tb_args': tb_args,
                'fit_intercept': False,
                'clr_args': clr_args,
                'verbose': 20,
                'seed': 2}

    print(par_dict)
    estimators = [('nnLR', nnMultiLogReg(**par_dict))]

    timer = Timer()
    n_row, n_col = 1, 10
    for idx, alg_ in enumerate(estimators):
        method_name, estimator = alg_
        # if method_name != 'nnLR':

        print("Extracting the top %d %s..." % (n_classes, method_name))

        timer.start()
        estimator.fit(X_tr, labels_tr)
        elapsed_time = timer.stop()

        pred_labels_tst = estimator.predict(X_tst)

        acc = accuracy_score(labels_tst, pred_labels_tst)

        # acc = np.mean(pred_labels_tst == labels_tst)
        print('Method {} Acc. {}'.format(method_name, acc))

        coef_ = estimator.coef_
        intercept_ = estimator.intercept_
        par_dict = estimator.get_params()

        plot_gallery(exp_folder_num=exp_folder_num, title='Original Images', images=X_tst[labels_tst == 6],
                     n_col=n_col, n_row=n_row,
                     image_shape=image_shape)

        plot_gallery(exp_folder_num=exp_folder_num, title="coef_", images=coef_, n_col=n_col, n_row=n_row,
                     image_shape=image_shape, save=True)

        cm = confusion_matrix(labels_tst, pred_labels_tst)

        plt.figure(figsize=(12, 12))
        sns.heatmap(cm, annot=True,
                    linewidths=.5, square=True, cmap='Blues_r', fmt='0.4g');

        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

        plt.savefig(os.path.join(exp_folder_num, 'cm_{0:.5f}.eps'.format(acc)), format='eps',
                    dpi=100, bbox_inches='tight')

        with open(os.path.join(exp_folder_num, 'coef_.pkl'), 'wb') as f:
            pickle.dump(coef_, f)

        with open(os.path.join(exp_folder_num, 'intercept_.pkl'), 'wb') as f:
            pickle.dump(intercept_, f)

        with open(os.path.join(exp_folder_num, 'par_dict.json'), 'w') as fp:
            json.dump(par_dict, fp, sort_keys=True, indent=4, default=lambda o: '<not serializable>')

        with open(os.path.join(exp_folder_num, 'elapsed_time.json'), 'w') as fp:
            json.dump({"Elapsed time": elapsed_time}, fp, sort_keys=True, indent=4,
                      default=lambda o: '<not serializable>')

    print(par_dict)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # # Required parameters
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

    elif 'fashion' in FLAGS.task_name:

        from sklearn.utils import shuffle

        ###########3
        DATASET = 'fashion'
        image_shape = (28, 28)

        images_tr, labels_tr = load_mnist('../../Datasets/fashion', perc_samples=1., kind='train')
        images_tst, labels_tst = load_mnist('../../Datasets/fashion', perc_samples=1., kind='t10k')

        X_tr = np.asarray(images_tr).astype(np.float64) / 255.
        labels_tr = np.asarray(labels_tr).astype(np.int32)
        X_tr, labels_tr = shuffle(X_tr, labels_tr, random_state=0)

        # X_tr, labels_tr = X_tr[:1000], labels_tr[:1000]

        X_tst = np.asarray(images_tst).astype(np.float64) / 255.
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

    scaler = StandardScaler(with_std=False)     # Set with_std=False if you want z-score normalization
    X_tr = scaler.fit_transform(X_tr)
    X_tst = scaler.transform(X_tst)

    main()
