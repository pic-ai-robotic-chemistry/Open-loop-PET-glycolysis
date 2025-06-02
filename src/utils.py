import warnings
from scipy.stats import norm


def get_openai_key(var_name="openai_key"):
    with open("openai_api_key", 'r') as f:
        line = f.readline()
        return line.strip()


class AcquisitionFunction(object):

    def __init__(self, kind='ucb', kappa=2.576, xi=0, kappa_decay=1):

        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self.xi = xi
        self._iter_counter = 0
        assert kind in ['ucb', 'ei', 'poi'], "Invalid acquisition function"
        self.kind = kind

    def update_params(self):
        if self._kappa_decay < 1:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        elif self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        elif self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x.reshape(1, -1), return_std=True)
        return (mean + kappa * std)[0]

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x.reshape(1, -1), return_std=True)
        z = (mean - y_max - xi) / std
        return ((mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z))[0]

    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x.reshape(1, -1), return_std=True)
        z = (mean - y_max - xi) / std
        return (norm.cdf(z))[0]


