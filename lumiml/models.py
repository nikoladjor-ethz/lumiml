import nlopt
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.linear_model.coordinate_descent import LinearModelCV
from sklearn.utils import check_X_y


class PoissonElasticNet(LinearModel, RegressorMixin):
    """
    Linear regression with combined L1 and L2 priors as regularizer.
    Objective function is the negative log-like loss of Poisson Maximum Likelihood.

    Parameters
    ----------
    alpha : float
        Constant that multiplies penalty terms. Optimal value is found using cross-validation.

    l1_ratio : float
        Mixing parameter for ElasticNet. It defines the ratio between L1 and L2 regularization.
        We refer to this parameter as :math:`\theta` in the paper.

    max_iter : int
        Maximum number of iterations for the NLopt optimization routine.

    tol : int
        Relative tolerance for the optimizer convergence.

    positive : bool, defaults to True
        When set to `True`, it forces coefficients to be positive

    intercept_guess : float
        Initial guess for background noise.

    fix_intercept: defaults to `False`
        When set to `True`, the intercept will be set to intercept_guess and not used in the optimization.

    method : nlopt.method
        Optimization method from NLopt package.

    warm_start : bool, default `False`.
        When set to `True`, initial guess will use `init_vector` as a starting point for optimization.

    init_vector : bool, default `None`.
        Initial guess vector for coefficients.

    Attributes
    ----------

    coef_ : array, shape(n_features)
        Vector of coefficients. It is the solution to the optimization problem. It is used together ith the basis,
        to reconstruct underlying decay rate distribution.

    """
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=10000, tol=1e-9, positive=True, intercept_guess=0.,
                 fix_intercept=False, method=nlopt.LN_NELDERMEAD, warm_start=False, init_vector=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.positive = positive
        self.method = method
        self.intercept_guess = intercept_guess
        self.fix_intercept = fix_intercept
        self.warm_start = warm_start
        self.init_vector = init_vector

        self.optimizer = None   # placeholder for defining optimizer later
        self.loc_optimizer_ = None   # placeholder for local optimizer (case of method=nlopt.AUGLAG)

        super(PoissonElasticNet, self).__init__()

    @staticmethod
    def loss_enet(X, y, reg_lambda, alpha, theta):
        """

        Parameters
        ----------
        X : np.ndarray
            Features matrix to calculate prediction
        y: np.ndarray
            Data being modelled.
        reg_lambda : float
            Regularization parameter
        alpha : float
            Ratio between the L1 and L2 penalties.
        theta : np.ndarray
            Vector of coefficients.

        Returns
        -------
        float
            Regularized log-likelihood loss.

        """
        y_hat = theta[0] + np.dot(X, theta[1:])
        n_samples = X.shape[0]
        l1_pen = reg_lambda * alpha * np.linalg.norm(theta[1:], 1)
        l2_pen = 0.5 * reg_lambda * (1 - alpha) * np.linalg.norm(theta[1:], 2) ** 2
        log_like_loss = np.sum(y_hat - y * np.log(y_hat)) / n_samples
        log_like_loss += l1_pen
        log_like_loss += l2_pen
        return log_like_loss

    @staticmethod
    def predict_(X, theta):
        return theta[0] + X.dot(theta[1:])

    @staticmethod
    def loglike_loss(X, y, theta):
        """

        Parameters
        ----------
        X
        y
        theta

        Returns
        -------
        float
            Log-likelihood loss.

        Warnings
        --------
        Ignored.

        """
        y_hat = PoissonElasticNet.predict_(X, theta)
        n_samples = X.shape[0]
        loss_ = np.sum(y_hat - y * np.log(y_hat)) / n_samples
        return loss_.copy()

    @staticmethod
    def optimization_loss(theta, grad, X, y, reg_lambda, alpha):
        """

        Parameters
        ----------
        theta : np.ndarray
            Vector of coefficients.
        grad
            Gradient, required for nlopt. Essentially, it is ignored, since we perform optimization without a gradient.
        X : np.ndarray
            Features matrix generated with certain basis.
        y : np.ndarray
            Data being modelled.
        reg_lambda : float
            Regularization parameter.
        alpha : float
            Ratio between the L1 and L2 penalties.

        Returns
        -------
        float
            Loss used for optimization.

        """
        if grad.size > 0:
            grad[:] = 0
        return PoissonElasticNet.loss_enet(X, y, reg_lambda, alpha, theta)

    def fit(self, X, y):
        """
        Fit the model using the optimizer algorithm defined in `self.method`.

        Parameters
        ----------
        X : ndarray
            Features matrix generated using a basis.
        y : ndarray, shape (n_samples)
            Target data.

        Notes
        -----

        The performance of the fit method will depend on the algorithm you choose. Please refer to
        NLopt reference page for more information.

        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        if self.warm_start and self.init_vector is not None:
            init_coef_ = self.init_vector
        elif self.warm_start and self.init_vector is None:
            print('Warm start without initial guess, defaulting to random guess')
            init_coef_ = 1 / (X.shape[1]) * y[0] * np.random.random(X.shape[1] + 1)
            init_coef_[0] = self.intercept_guess
        elif not self.warm_start:
            init_coef_ = 1 / (X.shape[1]) * y[0] * np.random.random(X.shape[1] + 1)
            init_coef_[0] = self.intercept_guess

        # define optimizer
        self.optimizer = nlopt.opt(self.method, n_features + 1)  # always fit intercept
        self.optimizer.set_ftol_rel(self.tol)
        self.optimizer.set_maxeval(self.max_iter)
        if self.fix_intercept:
            if self.positive:
                lb_vector = np.zeros((n_features + 1))
                lb_vector[0] = self.intercept_guess
                ub_vector = np.inf * np.ones_like(lb_vector)
                ub_vector[0] = self.intercept_guess
                self.optimizer.set_lower_bounds(lb_vector)
                self.optimizer.set_upper_bounds(ub_vector)
        else:
            if self.positive:
                self.optimizer.set_lower_bounds(0)

        # in case of AUGLAG, set the local optimizer
        self.loc_optimizer_ = nlopt.opt(nlopt.LN_SBPLX, n_features + 1)
        self.optimizer.set_local_optimizer(self.loc_optimizer_)

        min_func = lambda theta, grad: PoissonElasticNet.optimization_loss(theta, grad, X, y, self.alpha, self.l1_ratio)
        self.optimizer.set_min_objective(min_func)

        self.coef_ = self.optimizer.optimize(init_coef_)
        self.intercept_ = self.coef_[0]

        return self

    def predict(self, X):
        """

        Parameters
        ----------
        X: ndarray
            Features matrix

        Returns
        -------
        y_hat: ndarray
            Predicted values given calculated coefficients in `self.coef_`

        """
        return self.coef_[0] + X.dot(self.coef_[1:])

    def score(self, X, y, sample_weight=None):
        """

        Parameters
        ----------
        X: ndarray
            Feature matrix
        y: ndarray
            Target data
        sample_weight: ndarray, not used
            Weights for the samples, not used.

        Returns
        -------
        score: float
            Negated log-like loss in order to comply with sklearn API.

        """
        return -1 * PoissonElasticNet.loglike_loss(X, y, self.coef_)
