import logging
import numpy as np
from Utils.CLR import CLR
from Utils.timer import Timer
from tabulate import tabulate
from sklearn import preprocessing
from scipy.stats import halfnorm
from tensorboardX import SummaryWriter
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin


class FoundViolation(Exception): pass


class ExecStop(Exception): pass


class nnMultiLogReg(BaseEstimator, TransformerMixin):
    l2 = lambda self, t, axis=0: np.linalg.norm(t, ord=2, axis=axis)
    l1 = lambda self, t, axis=0: np.linalg.norm(t, ord=1, axis=axis)

    def __init__(self, atol=1e-4, reltol=1e-8,
                 constr='l1', beta=0.0, maxit=1000, seed=0,
                 fit_intercept=True, tb_args=None,
                 clr_args=None, verbose=True, max_verb=50):

        self.atol = atol
        self.beta = beta
        self.seed = seed
        self.constr = constr
        self.maxit = maxit
        self.reltol = reltol
        self.timer = Timer()
        self.tb_args = tb_args
        self.verbose = verbose
        self.clr_args = clr_args
        self.max_verb = max_verb
        self.fit_intercept = fit_intercept

        if self.constr not in ['l2', 'l1']:
            raise FoundViolation("--Constraint should be l2 or l1.")

        if tb_args:
            self.tf_writer = SummaryWriter(tb_args['logdir'])
            self.image_shape = tb_args['image_shape']

        self.lbin = preprocessing.LabelBinarizer()

        self.coef_ = None
        self.intercept_ = 0

        if not clr_args:
            self.lr_fun = CLR(clr_type='fixed_step', base_lr=1e-3).get_lr_fun()
            logging.warning('No learning rate strategy is picked. A fixed learning is chosen with value %f' % 1e-3)
        else:
            self.lr_fun = CLR(**self.clr_args).get_lr_fun()

    @staticmethod
    def __timer(start, end):

        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    def __halfnorm(self, dim, seed=0):

        random_state = np.random.RandomState(seed=seed)
        fan_in, fan_out = dim
        # variance = 2. / (fan_in + fan_out)
        variance = 1. / fan_in
        std = 1 * np.sqrt(variance)
        w_init = halfnorm.rvs(loc=0, scale=std, size=dim, random_state=random_state)
        if self.constr == 'l2':
            w_init = w_init / self.l2(w_init, axis=1)[:, np.newaxis]
        elif self.constr == 'l1':
            w_init = w_init / self.l1(w_init, axis=1)[:, np.newaxis]

        return w_init

    @staticmethod
    def __is_poss(weights):
        check_poss = weights < 0.0  # self.atol
        if any(check_poss):
            raise FoundViolation("--Not Positive Violation (break)--")

    def __is_converged(self, w_j, w_j_prev, n_iter, n_comp, stop_flags, check_poss=True):
        error = np.linalg.norm(w_j_prev - w_j, ord=2) / np.linalg.norm(w_j_prev, ord=2)
        if error <= self.reltol:
            # if check_poss: self.__is_poss(w_j)
            # raise ExecStop('--Loop Stop (break): Iterations: {}'.format(n_iter))
            stop_flags[n_comp] = True

        if n_iter >= self.maxit:
            if check_poss: self.__is_poss(w_j)
            raise ExecStop('--Loop Stop (break): Iterations: {}'.format(n_iter))
            # stop_flags[n_comp] = True

        if all(stop_flags):
            if check_poss: self.__is_poss(w_j)
            raise ExecStop('--Loop Stop (break): Iterations: {}'.format(n_iter))

        if self.tb_args:
            self.tf_writer.add_scalars(f'error', {'w_{}'.format(n_comp): error}, n_iter)

        return error

    def softmax(self, a_j, a, normalize=False):

        if normalize:
            a_max = np.max(a)
            a_j -= a_max
            a -= a_max

        y_j = np.exp(a_j) / np.sum(np.exp(a))

        return y_j

    def partial_fit(self, x_n, t_n, w, w_prev, active_idx, lr, n_comp):
        w_j = w[n_comp][:, None].copy()
        w_j_prev = w_prev[n_comp][:, None].copy()
        active_idx_j = active_idx[n_comp][:, None].copy()

        # c = 0.
        # Constraint projection
        if self.constr == 'l2':
            norm_w_j = self.l2(w_j)
            proj_w = np.eye(self.n_features) - np.dot(w_j, w_j.T) / (norm_w_j ** 2)
            # conv = c * w_j * (self.l2(w_j) - 1)
        elif self.constr == 'l1':
            q = np.zeros((self.n_features, 1))
            q[active_idx_j == False] = 1
            proj_w = np.eye(self.n_features) - np.dot(q, q.T) / np.sum(active_idx_j == False)  # float(self.n_features)
            # proj_w = np.eye(self.n_features) - np.ones(self.n_features) / float(self.n_features)
            # conv = c * np.ones((self.n_features, 1)) * (self.metric(w_j) - 1)
        else:
            proj_w = np.eye(self.n_features)

        a = w @ x_n
        a_j = np.vdot(w_j, x_n)
        y_j = self.softmax(a_j, a, normalize=False)

        F_nabla = - (t_n[n_comp] - y_j) * x_n

        # F_nabla += conv

        F_nabla[active_idx_j == True] = 0.0
        F_nabla_proj = proj_w @ F_nabla

        # Momentum term
        momentum_w_j = w_j - w_j_prev
        momentum_w_j[active_idx_j == True] = 0.0  # Respect the activations from w_prev

        # Update rule
        w_j_prev = w_j.copy()
        w_j = w_j - lr * F_nabla_proj + self.beta * momentum_w_j

        # Check activation
        active_idx_local = w_j <= self.atol

        # Update active constraints
        active_idx_j[active_idx_local] = True

        # Set exactly zero for numerical stability
        w_j[active_idx_j == True] = 0.0

        w[n_comp], w_prev[n_comp], active_idx[n_comp] = w_j.squeeze(), w_j_prev.squeeze(), active_idx_j.squeeze()

        return self

    def __check_verbose(self):

        return not self.verbose or self.verbose < 0 or self.verbose > self.max_verb

    def __print_stats(self, w, active_idx, n_iter, stop_flags):
        if self.__check_verbose():
            return
        else:
            if n_iter % (self.max_verb - self.verbose + 1) == 0:
                headers = ['vars (n_iter: {})'.format(n_iter)] + ['w{}'.format(c) for c in range(self.n_classes)]

                a = ['activations'] + [str(np.sum(active_idx[c])) for c in range(self.n_classes)]
                l = [str(self.constr)] + [str(self.l1(w[c]) if self.constr == 'l1' else self.l2(w[c]))
                                          for c in range(self.n_classes)]
                f = ['stop flag'] + [stop_flags[c] for c in range(self.n_classes)]
                print(tabulate([a, l], headers=headers, floatfmt=".4f", tablefmt="fancy_grid"))

    def fit(self, X, y=None):

        """Fit the model with X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : None
            Ignored variable.
        Returns
        -------
        self : object
            Returns the instance itself.
        """

        if not self.clr_args:
            raise FoundViolation('Specify clr_args')
        else:
            lr_fun = CLR(**self.clr_args).get_lr_fun()

        self.classes_ = np.unique(y)
        self.n_classes = len(np.unique(y))

        self.n_samples, self.n_features = X.shape

        if self.n_classes == 2:
            y_endoced = np.zeros((self.n_samples, self.n_classes), dtype=np.float64)
            for i, cls in enumerate(np.unique(y)):
                y_endoced[y == cls, i] = 1
        else:
            y_endoced = self.lbin.fit_transform(y)

        if self.fit_intercept:
            X = np.hstack((X, np.ones((self.n_samples, 1))))
            self.n_features += 1

        self.timer.start()
        try:
            # import _pickle as cPickle
            # with open(r"logs/coef_.pkl", "rb") as input_file:
            #     coef_ = cPickle.load(input_file)
            # self.coef_ = coef_
            #
            # with open(r"logs/intercept_.pkl", "rb") as input_file:
            #     intercept_ = cPickle.load(input_file)
            # self.intercept_ = intercept_
            #
            # return self

            active_idx = np.zeros((self.n_classes, self.n_features), dtype=bool)

            # w = self.__halfnorm(dim=(self.n_classes, self.n_features), seed=self.seed, l2_normalize=True)
            w = np.vstack([self.__halfnorm(dim=(1, self.n_features), seed=c) for c in range(self.n_classes)])
            w_prev = w.copy() + 0.1 * np.random.normal(0, 1, w.shape)

            n_iter = 1
            stop_flags = self.n_classes * [False]
            try:
                while True:  # Epoch loop
                    for point_i in range(self.n_samples):

                        x_n = X[point_i][:, None]
                        t_n = y_endoced[point_i]

                        error = 0
                        lr = lr_fun(iteration=n_iter)
                        for n_comp in range(self.n_classes):

                            # if stop_flags[n_comp]:
                            #     continue

                            self.partial_fit(x_n=x_n, t_n=t_n, w=w, w_prev=w_prev,
                                             active_idx=active_idx, lr=lr, n_comp=n_comp)

                            error += self.__is_converged(w_j=w[n_comp], w_j_prev=w_prev[n_comp],
                                                         n_iter=n_iter, n_comp=n_comp,
                                                         stop_flags=stop_flags, check_poss=True)

                            # Print activations to tensor tb_args
                            if self.tb_args:
                                if self.image_shape:
                                    if self.fit_intercept:
                                        w_ = w[:, :-1].copy()
                                    else:
                                        w_ = w.copy()
                                    vmax, vmin = w_[n_comp].max(), w_[n_comp].min()
                                    w_ = (w_[n_comp] - vmin) / (vmax - vmin)
                                    self.tf_writer.add_image(tag='w{}'.format(n_comp),
                                                             img_tensor=w_.reshape(self.image_shape),
                                                             global_step=n_iter, dataformats='HW')

                                self.tf_writer.add_scalars(f'error_total', {'w': error / float(self.n_classes)}, n_iter)

                        n_iter += 1

                        # Print stats in console
                        self.__print_stats(w, active_idx, n_iter, stop_flags)

            except ExecStop as e:
                logging.warning(' / '.join([str(e), 'n_iter: {}'.format(n_iter)]))

        except FoundViolation as e:
            logging.warning(e)
        else:
            # Normalize to avoid small deviations from unit norm
            self._W = w
            if self.fit_intercept:
                self.coef_ = w[:, :-1]
                self.intercept_ = w[:, -1]
            else:
                self.coef_ = w

        self.timer.stop()

        return self

    def predict_proba(self, X):

        check_is_fitted(self, msg='nnLR not fitted.')

        if self.fit_intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))

        _activations = X @ self._W.T

        # _activations = X @ self.coef_.T + self.intercept_

        _activations_max = np.max(_activations, axis=1)
        _activations -= _activations_max[:, None]

        sigma = np.exp(_activations) / np.sum(np.exp(_activations), axis=1, keepdims=True)

        return sigma

    def predict(self, X):

        check_is_fitted(self, msg='nnLR not fitted.')
        sigma = self.predict_proba(X)
        y_pred = np.argmax(sigma, axis=1)

        return self.classes_[y_pred]
