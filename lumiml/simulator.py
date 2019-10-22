import numpy as np
from scipy import special
from scipy import stats
from scipy import integrate


class DecayDistribution(object):
    ALLOWED_DISTRIBUTIONS = [
        'delta-like',
        'continuous',
        'stretched'
    ]

    def __init__(self, gamma_eval: np.ndarray, distribution_type='delta-like'):
        assert distribution_type in self.ALLOWED_DISTRIBUTIONS
        self.distribution_type = distribution_type

        assert isinstance(gamma_eval, np.ndarray)
        self.gamma_eval = gamma_eval

    def transform(self, time_scale):
        pass

    def _distribution_function(self):
        pass

    @staticmethod
    def _step(x):
        return np.asarray(x >= 0, dtype=np.int)


class DeltaDistribution(DecayDistribution):
    def __init__(self, gamma_eval, weights=None):
        super(DeltaDistribution, self).__init__(distribution_type='delta-like', gamma_eval=gamma_eval)

        if weights is None:
            self.weights = np.ones_like(gamma_eval)
        else:
            assert isinstance(weights, np.ndarray)
            assert len(weights) == len(gamma_eval)
            self.weights = weights

        self.n_exp = len(gamma_eval)

    def transform(self, time_scale):
        y_transform = (np.exp(- time_scale[:, np.newaxis] * self.gamma_eval) * self.weights).sum(axis=1)
        return y_transform.copy()

    def _distribution_function(self):
        return self.weights


class StretchedExponentialDistribution(DecayDistribution):
    def __init__(self, gamma_eval, beta_kww, tau_kww, n_sum_terms=10):
        super(StretchedExponentialDistribution, self).__init__(distribution_type='stretched', gamma_eval=gamma_eval)

        assert isinstance(n_sum_terms, int), \
            'n_sum_terms is not an integer:%r' % n_sum_terms
        self.n_sum_terms = n_sum_terms

        assert tau_kww > 0 and (0 < beta_kww < 1),\
            'Error in defining distribution parameters: tau_kww > 0 and 0<beta_kww<1.' \
            ' tau_kww:%r, beta_kww:%r' % (tau_kww, beta_kww)
        self.tau_kww = tau_kww
        self.beta_kww = beta_kww
        
    @staticmethod
    def _single_term(tau, bww, tww, k):
        bk = bww * k
        st = (((-1) ** k) / np.math.factorial(k)) * np.sin(np.pi * bk) * special.gamma(bk + 1) * ((tau / tww) ** bk)
        return st

    def _distribution_function(self):
        # TODO: consider normalizing with 1/gamma_eval to avoid post-normalization when plotting!
        tau = 1/self.gamma_eval
        beta = self.beta_kww
        tww = self.tau_kww
        kmax = self.n_sum_terms

        rho_k_ = [self._single_term(tau, beta, tww, k) for k in range(0, kmax)]
        rho_ = (-1 / np.pi) * np.sum(np.asarray(rho_k_), axis=0)
        return rho_.copy()

    def transform(self, time_scale):
        exponent = -(self._step(time_scale) * time_scale/self.tau_kww)**self.beta_kww
        y_transform = np.exp(exponent)
        return y_transform.copy()


class ContinuousDistribution(DecayDistribution):
    def __init__(self, gamma_eval, distribution=stats.lognorm, **kwargs):
        super(ContinuousDistribution, self).__init__(gamma_eval=gamma_eval)
        self.distribution = distribution
        self.kwargs = kwargs

    def _distribution_function(self):
        rho_ = self.distribution.pdf(self.gamma_eval, **self.kwargs)
        return rho_.copy()

    @staticmethod
    def _get_y(t, rho, gamma_eval):
        if t < 0:
            return 0
        else:
            return integrate.trapz(rho * np.exp(-t * gamma_eval), gamma_eval)

    def transform(self, time_scale):
        rho = self._distribution_function()
        get_y = lambda t: self._get_y(t, rho, self.gamma_eval)
        ss = np.array([int(x >= 0) for x in time_scale])
        y = np.asarray(list(map(get_y, time_scale)))
        y_transform = ss * y

        return y_transform.copy()


class SimulatedData(object):
    def __init__(self):
        self.background_corrected_data = None
        self.background_noise = None
        self.simulated = None
        self.analytical = None

    def set_data(self, analytical, simulated, background_noise, background_corrected_data):
        self.background_corrected_data = background_corrected_data
        self.background_noise = background_noise
        self.simulated = simulated
        self.analytical = analytical

    def save_to_file(self):
        pass


class Simulator(object):
    def __init__(self, distribution: DecayDistribution, time_scale, background_mean, snr, with_background=True):
        self.with_background = with_background
        self.snr = snr
        self.background_mean = background_mean
        assert isinstance(time_scale, np.ndarray)
        self.time_scale = time_scale

        self._filter_flag_time_scale = (self.time_scale >= 0).all()

        self.distribution = distribution

        self.data_simulated = SimulatedData()

    @staticmethod
    def step(x):
        return np.asarray(x >= 0, dtype=np.int)

    def simulate_data(self):
        max_counts = self.background_mean * (self.snr - 1)
        y_clean = self.step(self.time_scale) * self.distribution.transform(self.time_scale)

        y_analytic_max = max_counts * y_clean / max(y_clean)

        bcg_noise = np.random.poisson(self.background_mean, self.time_scale.shape[0])
        bcg_vec = self.background_mean * np.ones_like(self.time_scale)

        y_simulated = np.random.poisson(y_analytic_max + bcg_vec)
        y_simulated_bcg_corr = y_simulated - np.mean(bcg_noise)

        self.data_simulated.set_data(
            analytical=y_analytic_max,
            simulated=y_simulated,
            background_noise=bcg_noise,
            background_corrected_data=y_simulated_bcg_corr
        )

    def save_to_file(self):
        pass

    def new_time_scale(self):
        pass
        



