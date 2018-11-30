""" Base classes for basis features and Poisson MLE estimator. """

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import FLOAT_DTYPES
import nlopt


class GenericBasisFeatures(BaseEstimator, TransformerMixin):
    """ 
    Generic base class for features generated from basis.
    Inherit from this class if you want to build your own custom features generator.
    """
    def __init__(self, omega=np.pi, g_min=1e-3, g_max=1e2, with_bias=True, whiten=False, fix_low_end=False):
        self.fix_low_end = fix_low_end
        self.whiten = whiten
        self.with_bias = with_bias
        self.g_max = g_max
        self.g_min = g_min
        self.omega = omega
        self.gamma_axis = None
        self.n_output_features = None
        self.scale = 1
        super(GenericBasisFeatures, self).__init__()

    @staticmethod
    def get_gamma_space(gamma_min, gamma_max, omega):
        """
        Calculates the points in the gamma space using exponential sampling.
        The first point is fixed to `gamma_min`, and spacing is given by resolution parameter `omega`.

        :param gamma_min: The lowest decay rate to consider
        :param gamma_max: The highest decay rate to consider
        :param omega: Resolution for exponential sampling
        :return: Vector of gamma points using exponential sampling
        """
        dg = np.pi / omega
        n_gamma = int(np.log(gamma_max / gamma_min) / dg)
        gamma_vec = np.asarray([gamma_min * np.exp(i * dg) for i in range(0, n_gamma + 1)])
        return gamma_vec.copy()

    @staticmethod
    def calculate_gamma_space(gamma_min, gamma_max, omega, fix_low_end):
        """
        More flexible method for computing points in gamma-space using exponential sampling.
        In the case of `fix_low_end = True`, the lowest decay rate (`gamma_min`) will be fixed.
        Otherwise, the points will be calculated as `gamma_vec[i]=gamma_min * np.exp((ig + 0.5) * dg)`,
        where `dg = pi/omega`.

        Parameters
        ----------
        gamma_min: float
            The lowest decay rate to consider
        gamma_max: float
            The highest decay rate to consider
        omega: float
            Resolution for exponential sampling

        fix_low_end: float
            Logical value - should the smallest decay rate be fixed?

        Returns
        -------
        ndarray
            Vector of gamma points using exponential sampling

        """
        if fix_low_end:
            gamma_vec = GenericBasisFeatures.get_gamma_space(gamma_min, gamma_max, omega)
            return gamma_vec.copy()

        dg = np.pi / omega
        gamma_range = np.log(gamma_max / gamma_min)

        M = np.int(np.floor((gamma_range * omega / np.pi) + 1))
        gamma_vec = np.asarray([gamma_min * np.exp((ig + 0.5) * dg) for ig in range(0, M)])

        return gamma_vec.copy()

    def calculate_scale(self, y):
        """
        In the case where one would like to use Gaussian statistics and scale variance to 1, the data should be scaled.
        This method scale vector to be used with the `fit` method for features generator.

        Warnings
        --------
        This means that the data one would like to fit should be scaled with the same vector too!

        Parameters
        ----------
        y:
            Data to be modelled.

        Returns
        -------
        None
            The `scale` attribute is changed in-place.
        """

        # in the case of pre-whitening, already scaled y (y/sqrt(y) is used
        # check for zeros
        tp_eps = np.finfo(y.dtype).eps
        zero_flag = np.absolute(y) < tp_eps
        self.scale = np.empty_like(y)
        if np.any(zero_flag):
            # there are critical values
            self.scale[zero_flag] = 1
            self.scale[~zero_flag] = np.absolute(y[~zero_flag]) ** -0.5

        else:
            # good to go for scaling
            self.scale = np.absolute(y) ** -0.5

        return None

    def fit(self, y=None):
        """
        `fit` method for the estimator/features.
        It is used to calculate the gamma_axis, number of output features and scale if needed.

        :param y:
        :return: None. The object is modified in-place.
        """
        self.gamma_axis = DeltaBasisFeatures.calculate_gamma_space(gamma_min=self.g_min, gamma_max=self.g_max,
                                                                   omega=self.omega, fix_low_end=self.fix_low_end)
        self.n_output_features = len(self.gamma_axis) + int(self.with_bias)

        if self.whiten:
            self.calculate_scale(y)

        return self


class DeltaBasisFeatures(GenericBasisFeatures):
    """ Class for creating delta-basis features."""
    def __init__(self, omega=np.pi, g_min=1e-3, g_max=1e2, with_bias=True, whiten=False, fix_low_end=False):
        super(DeltaBasisFeatures, self).__init__(omega, g_min, g_max, with_bias, whiten, fix_low_end)

    @staticmethod
    def _delta_basis(t, g):
        return np.exp(-t * g)

    def transform(self, X: np.ndarray):
        """
        Transform method for `DeltaBasisFeatures` class.

        :type X: np.ndarray
        :param X: vector of time points, given as `X=t[:, np.newaxis]`
        :return: `Xd`, matrix of features using delta-basis.
        """
        X = check_array(X, dtype=FLOAT_DTYPES)
        n_samples, n_features = X.shape

        # allocate output data
        Xd = np.empty((n_samples, self.n_output_features), dtype=X.dtype)
        Xd[:, 0] = 1
        start = int(self.with_bias)
        np.exp(-(X-X[0]) * self.gamma_axis, Xd[:, start:])

        if self.whiten:
            np.multiply(self.scale[:, np.newaxis], Xd[:, start:], out=Xd[:, start:])

        return Xd


class HistBasisFeatures(GenericBasisFeatures):
    """ Class for creating histogram-basis features."""
    def __init__(self,omega=np.pi, g_min=1e-3, g_max=1e2, with_bias=True, whiten=False, fix_low_end=False):
        super(HistBasisFeatures, self).__init__(omega, g_min, g_max, with_bias, whiten, fix_low_end)

    def fit(self, y=None):
        self.gamma_axis = GenericBasisFeatures.calculate_gamma_space(gamma_min=self.g_min, gamma_max=self.g_max,
                                                                     omega=self.omega, fix_low_end=self.fix_low_end)

        self.n_output_features = len(self.gamma_axis) - 1 + int(self.with_bias)

        if self.whiten:
            self.calculate_scale(y)

        return self

    def transform(self, X):
        """
        Transform method for histogram-basis features.

        :param X: Vector of time points, should be specified as `X=t[:, np.newaxis]`.
        :return: Matrix of features. Note that the number of output features is smaller than `self.gamma_axis`.
        """
        X = check_array(X, dtype=FLOAT_DTYPES)
        n_samples, n_features = X.shape

        # allocate output data
        Xd = np.empty((n_samples, self.n_output_features), dtype=X.dtype)
        Xd[:, 0] = 1
        start = int(self.with_bias)

        # take care of division by zero in the integral
        first_row = np.diff(self.gamma_axis)
        Xd[0, start:] = first_row

        Xd[1:, start:] = (np.exp(-(X[1:] - X[0])*self.gamma_axis[0:-1]) - np.exp(-(X[1:] - X[0])*self.gamma_axis[1:]))/(X[1:]-X[0])

        if self.whiten:
            np.multiply(self.scale[:, np.newaxis], Xd[:, start:], out=Xd[:, start:])

        return Xd


class TriangBasisFeatures(GenericBasisFeatures):
    """ Class for creating triangular-basis features."""
    def __init__(self, omega=np.pi, g_min=1e-3, g_max=1e2, with_bias=True, whiten=False, fix_low_end=False):
        super(TriangBasisFeatures, self).__init__(omega, g_min, g_max, with_bias, whiten, fix_low_end)

    def fit(self, y=None):
        self.gamma_axis = GenericBasisFeatures.calculate_gamma_space(gamma_min=self.g_min, gamma_max=self.g_max,
                                                                     omega=self.omega, fix_low_end=self.fix_low_end)
        self.n_output_features = len(self.gamma_axis) - 2 + int(self.with_bias) # should be -2, double check

        if self.whiten:
            self.calculate_scale(y)

        return self

    def transform(self, X):
        """
        Transform method for Histogram features base class.

        :param X: Vector of time points, should be specified as `X=t[:, np.newaxis]`.
        :return: Matrix of features. Note that the number of output features is smaller than `self.gamma_axis`.
        """
        X = check_array(X, dtype=FLOAT_DTYPES)
        n_samples, n_features = X.shape

        # allocate output data
        Xd = np.empty((n_samples, self.n_output_features), dtype=X.dtype)
        Xd[:, 0] = 1  # in case of a bias, put here all ones
        start = int(self.with_bias)

        # take care of division by zero in the integral
        first_row = []
        for i in range(1, len(self.gamma_axis)-1):
            first_row.append(0.5 * (self.gamma_axis[i+1] - self.gamma_axis[i-1]))
        first_row = np.array(first_row)
        Xd[0, start:] = first_row

        ll = np.empty_like(Xd[1:, start:])
        rr = np.empty_like(Xd[1:, start:])
        t_ = X[1:] - X[0]

        for i in range(1, len(self.gamma_axis)-1):
            a_ = self.gamma_axis[i-1]
            b_ = self.gamma_axis[i]
            c_ = self.gamma_axis[i+1]

            ll[:, i-1] = np.squeeze(((np.exp(-b_ * t_) * (1 - a_*t_ + b_*t_) - np.exp(-a_*t_)) / ((a_ - b_) * t_**2)))
            rr[:, i-1] = np.squeeze(((np.exp(-b_ * t_) * (1 - c_*t_ + b_*t_) - np.exp(-c_*t_)) / ((b_ - c_) * t_**2)))

        Xd[1:,start:] = (ll + rr).copy()

        if self.whiten:
            np.multiply(self.scale[:, np.newaxis], Xd[:, start:], out=Xd[:, start:])
        return Xd

################################################################






