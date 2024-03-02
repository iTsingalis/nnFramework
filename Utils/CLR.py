import numpy as np


class CLR:

    # Leslie Smith's paper Cyclical Learning Rates for Training Neural Networks
    def __init__(self, step_size=30, base_lr=1e-3, max_lr=6e-3, gamma=0.99994, decay=1e-3,
                 clr_type='triangular2_cycle'):

        self.step_size = step_size
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.gamma = gamma
        self.decay = decay
        self.clr_type = clr_type

    def get_lr_fun(self):

        if self.clr_type is 'triangular_exp_cycle':
            lr_fun = self.__get_triangular_exp_cycle_lr
            return lr_fun
        elif self.clr_type is 'triangular2_cycle':
            lr_fun = self.__get_triangular2_cycle_lr
            return lr_fun
        elif self.clr_type is 'triangular_cycle':
            lr_fun = self.__get_triangular_cycle_lr
            return lr_fun
        elif self.clr_type is 'triangular2':
            lr_fun = self.__get_triangular2_lr
            return lr_fun
        elif self.clr_type is 'triangular':
            lr_fun = self.__get_triangular_lr
            return lr_fun
        elif self.clr_type is 'exp_drop':
            lr_fun = self.__get_exp_drop_lr
            return lr_fun
        elif self.clr_type is 'exp':
            lr_fun = self.__get_exp_lr
            return lr_fun
        elif self.clr_type is 'time_step':
            lr_fun = self.__get_time_step_lr
            return lr_fun
        elif self.clr_type is 'fixed_step':
            lr_fun = self.__get_fixed_lr
            return lr_fun
        else:
            raise ValueError("Wrong CLR function type. Chose one of the following:  {'triangular', 'triangular2', "
                             "'triangular_cycle', 'triangular2_cycle', 'time_decay', 'exp', 'time_step', 'fixed_step'}")

    def __get_triangular_lr(self, iteration):

        """Given the inputs, calculates the lr that should be applicable for this iteration"""
        cycle = np.floor(1 + iteration / (2 * self.step_size))
        x = np.abs(iteration / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))

        return lr

    def __get_triangular2_lr(self, iteration):

        cycle = np.floor(1 + iteration / (2 * self.step_size))
        x = np.abs(iteration / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))

        return lr

    def __get_exp_lr(self, iteration):

        cycle = np.floor(1 + iteration / (2 * self.step_size))
        x = np.abs(iteration / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.gamma ** iteration

        return lr

    def __get_triangular_cycle_lr(self, iteration):

        clr_fn = lambda x: 0.5 * (1 + np.sin(x * np.pi / 2.))
        cycle = np.floor(1 + iteration / (2 * self.step_size))
        x = np.abs(iteration / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * clr_fn(cycle)

        return lr

    def __get_triangular_exp_cycle_lr(self, iteration):

        clr_fn = lambda x: 0.5 * (1 + np.sin(x * np.pi / 2.))
        cycle = np.floor(1 + iteration / (2 * self.step_size))
        x = np.abs(iteration / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * clr_fn(
            cycle) * self.gamma ** iteration

        return lr

    def __get_triangular2_cycle_lr(self, iteration):

        cycle = np.floor(1 + iteration / (2 * self.step_size))
        x = np.abs(iteration / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * 1 / (5 ** (iteration * 0.0001))

        return lr

    def __get_exp_drop_lr(self, iteration):

        # lr = self.base_lr * (1.0 / (1.0 + self.decay * iteration))

        lr = self.base_lr * self.decay ** int(iteration / self.step_size)

        return lr

    def __get_time_step_lr(self, iteration):

        lr = self.base_lr * (1.0 / (1.0 + self.decay * (iteration)))

        return lr

    def __get_fixed_lr(self, iteration=None):

        return self.base_lr
