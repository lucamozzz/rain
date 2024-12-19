import numpy as np
from scipy.stats import norm
from warnings import warn
import copy
import time


class ERICS:
    def __init__(self, n_param, window_mvg_average=50, window_drift_detect=50, beta=0.0001, base_model='probit',
                 init_mu=0, init_sigma=1, epochs=10, lr_mu=0.01, lr_sigma=0.01):
        # User-set ERICS-hyperparameters
        self.n_param = n_param
        self.M = window_mvg_average
        self.W = window_drift_detect
        self.beta = beta
        self.base_model = base_model

        # Default hyperparameters
        self.time_step = 0                                          # Current Time Step
        self.time_since_last_global_drift = 0                       # Time steps since last global drift detection
        self.time_since_last_partial_drift = np.zeros(n_param)      # Time steps since last partial drift detection
        self.global_drifts = []                                     # Time steps of all global drifts
        self.partial_drifts = []                                    # (time step, param.idx)-tuples of all partial drifts
        self.alpha = None                                           # Adaptive threshold for global concept drift detection
        self.partial_alpha = np.asarray([None] * self.n_param)      # Adaptive threshold for partial concept drift detection
        self.mu_w = np.ones((self.M, self.n_param)) * init_mu       # Parameter Mean in window
        self.sigma_w = np.ones((self.M, self.n_param)) * init_sigma # Parameter Variance in window
        self.param_sum = np.zeros((self.M - 1, self.n_param))       # Sum-expression for computation of moving average (see Eq. (8) in [1])
        self.global_info_ma = []                                    # Global moving average
        self.partial_info_ma = []                                   # Partial moving average

        # Parameters of FIRES (Probit) model according to [2]
        if self.base_model == 'probit':
            self.fires_mu = np.ones(self.n_param) * init_mu
            self.fires_sigma = np.ones(self.n_param) * init_sigma
            self.fires_epochs = epochs
            self.fires_lr_mu = lr_mu
            self.fires_lr_sigma = lr_sigma
            self.fires_labels = []                                          # Unique labels (fires requires binary labels)
            self.fires_encode_labels = True                                 # Indicator for warning message (auto-encoded labels)

        # ### ADD YOUR OWN MODEL PARAMETERS HERE ############################
        # if self.base_model == 'your_model':
        #   # define parameters
        #####################################################################

    def check_drift(self, x, y):
        if self.alpha is not None:
            self.alpha -= (self.alpha * self.beta * self.time_since_last_global_drift)
        for k in range(self.n_param):
            if self.partial_alpha[k] is not None:
                self.partial_alpha[k] -= (self.partial_alpha[k] * self.beta * self.time_since_last_partial_drift[k])

        self.time_since_last_global_drift += 1
        self.time_since_last_partial_drift += 1

        if self.base_model == 'probit':
            self.__update_probit(x, y)
        else:
            raise NotImplementedError('The base model {} has not been implemented.'.format(self.base_model))

        start = time.time()  # Start time drift detection
        self.__update_param_sum()                   # Update the sum expression for observations in a shifting window
        self.__compute_moving_average()             # Compute moving average in specified window
        g_drift, p_drift = self.__detect_drift()    # Detect concept drift

        self.time_step += 1

        return g_drift, p_drift

    def __update_param_sum(self):
        if self.base_model == 'probit':
            new_mu = copy.copy(self.fires_mu).reshape(1, -1)
            new_sigma = copy.copy(self.fires_sigma).reshape(1, -1)

        else:
            raise NotImplementedError('The base model {} has not been implemented.'.format(self.base_model))

        self.mu_w = self.mu_w[1:, :]
        self.sigma_w = self.sigma_w[1:, :]

        self.mu_w = np.concatenate((self.mu_w, new_mu))
        self.sigma_w = np.concatenate((self.sigma_w, new_sigma))

        for t in range(self.M - 1):
            self.param_sum[t, :] = (self.sigma_w[t + 1, :] ** 2 + (self.mu_w[t, :] - self.mu_w[t + 1, :]) ** 2) / self.sigma_w[t, :] ** 2

    def __compute_moving_average(self):
        partial_ma = np.zeros(self.n_param)
        global_score = np.zeros(self.M - 1)

        for k in range(self.n_param):
            partial_score = self.param_sum[:, k] - 1
            global_score += partial_score
            partial_ma[k] = np.sum(np.abs(partial_score)) / (2 * self.M)  # Add partial mov. avg. for parameter k

        global_ma = np.sum(np.abs(global_score)) / (2 * self.M)

        self.global_info_ma.append(global_ma)
        self.partial_info_ma.append(partial_ma)

    def __detect_drift(self):
        global_window_delta = None
        partial_window_delta = None

        if self.W < 2:
            self.W = 2
            warn('Sliding window for concept drift detection was automatically set to 2 observations.')

        if len(self.global_info_ma) < self.W:
            oldest_entry = len(self.global_info_ma)
        else:
            oldest_entry = self.W

        if oldest_entry == 1:  # In case of only one observation
            global_window_delta = copy.copy(self.global_info_ma[-1])
            partial_window_delta = copy.copy(self.partial_info_ma[-1])
        else:
            for t in range(oldest_entry, 1, -1):
                if t == oldest_entry:
                    global_window_delta = self.global_info_ma[-t+1] - self.global_info_ma[-t]  # newer - older
                    partial_window_delta = self.partial_info_ma[-t+1] - self.partial_info_ma[-t]
                else:
                    global_window_delta += (self.global_info_ma[-t+1] - self.global_info_ma[-t])
                    partial_window_delta += (self.partial_info_ma[-t+1] - self.partial_info_ma[-t])

        if self.alpha is None:
            self.alpha = np.abs(global_window_delta)  # according to Eq. (6) in [1] -> abs() is only required at t=0, to make sure that alpha > 0
        if None in self.partial_alpha:
            unspecified = np.isnan(self.partial_alpha.astype(float)).flatten()
            self.partial_alpha[unspecified] = np.abs(partial_window_delta[unspecified])

        g_drift = False
        if global_window_delta > self.alpha:
            g_drift = True
            self.global_drifts.append(self.time_step)
            self.time_since_last_global_drift = 0
            self.alpha = None

        p_drift = False
        partial_drift_bool = partial_window_delta > self.partial_alpha
        for k in np.argwhere(partial_drift_bool):
            p_drift = True
            self.partial_drifts.append((self.time_step, k.item()))
            self.time_since_last_partial_drift[k] = 0
            self.partial_alpha[k] = None

        return g_drift, p_drift

    ###########################################
    # BASE MODELS
    ##########################################
    def __update_probit(self, x, y):
        if y not in self.fires_labels:  # Add newly observed unique labels
            self.fires_labels.append(y)

        if tuple(self.fires_labels) != (-1, 1):  # Check if labels are encoded correctly
            #print('FIRES WARNING: The target variable will automatically be encoded as {-1, 1}.')
            pass

        if len(self.fires_labels) < 2:
            y = -1 if y == self.fires_labels[0] else y
        elif len(self.fires_labels) == 2:
            y = -1 if y == self.fires_labels[0] else 1
        else:
            raise ValueError('The target variable y must be binary.')

        for epoch in range(self.fires_epochs):
            # Shuffle the observations
            random_idx = np.random.permutation(len(x))
            x = x[random_idx]
            y_shuffled = y  # y is a single value, no need to shuffle

            # Iterative update of mu and sigma
            try:
                # Helper functions
                dot_mu_x = np.dot(x, self.fires_mu)
                rho = np.sqrt(1 + np.dot(x ** 2, self.fires_sigma ** 2))

                # Gradients
                nabla_mu = norm.pdf(y_shuffled / rho * dot_mu_x) * (y_shuffled / rho * x.T)
                nabla_sigma = norm.pdf(y_shuffled / rho * dot_mu_x) * (
                        - y_shuffled / (2 * rho ** 3) * 2 * (x ** 2 * self.fires_sigma).T * dot_mu_x)

                # Marginal Likelihood
                marginal = norm.cdf(y_shuffled / rho * dot_mu_x)

                # Reshape nabla_mu, nabla_sigma, and marginal to ensure correct dimensions
                nabla_mu = np.expand_dims(nabla_mu, axis=0) if nabla_mu.ndim == 1 else nabla_mu
                nabla_sigma = np.expand_dims(nabla_sigma, axis=0) if nabla_sigma.ndim == 1 else nabla_sigma
                marginal = np.expand_dims(marginal, axis=0) if marginal.ndim == 1 else marginal

                # Update parameters
                self.fires_mu += self.fires_lr_mu * np.mean(nabla_mu / marginal, axis=1)
                self.fires_sigma += self.fires_lr_sigma * np.mean(nabla_sigma / marginal, axis=1)
            except TypeError as e:
                raise TypeError('All features must be a numeric data type.') from e
            except IndexError as e:
                raise IndexError('Index out of bounds. Ensure the input arrays have correct dimensions.') from e
