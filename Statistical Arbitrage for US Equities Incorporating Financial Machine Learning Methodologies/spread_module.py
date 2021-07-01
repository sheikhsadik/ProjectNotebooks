import numpy as np
import pandas as pd
from numba import jit


class SpreadModule:

    def __init__(self, pair, permnos):
        self.pair = pair
        self.permnos = permnos
        self.hedge_weights = pd.DataFrame()
        self.spreads = pd.DataFrame()

    @staticmethod
    @jit(nopython=True)
    def tls_coef(cov):
        e, v = np.linalg.eig(cov)
        sort_ix = e.argsort()[::-1]
        v = v[:, sort_ix]
        return v[0, 0]/v[1, 0]

    def get_hedge_ratio(self, method, hr_lkbk=252):

        if method is 'ols':
            ix_1 = self.permnos[0]; ix_2 = self.permnos[1]
            price_sds = self.pair.expanding(hr_lkbk).std()
            price_corr = self.pair.expanding(hr_lkbk).corr(self.pair[ix_1])[ix_2]
            self.hedge_weights = price_corr.multiply(price_sds[ix_1].divide(price_sds[ix_2]))
        elif method is 'tls':
            self.hedge_weights = self.pair.expanding(hr_lkbk).cov().dropna(axis=0).\
                groupby('date').apply(lambda x: self.tls_coef(x.values))
        else:
            print('Please input the correct hedge ratio method: ols or tls')
        return self.hedge_weights

    def get_spreads(self):

        self.spreads = self.pair[self.permnos[0]].subtract(
            self.pair[self.permnos[1]].multiply(self.hedge_weights, axis=0),
            axis=0
        ).dropna(axis=0)
        return self.spreads

    def get_ewm_zscore(self, spread_lkbk):

        return self.spreads.subtract(self.spreads.ewm(span=spread_lkbk).mean()).divide(
            self.spreads.ewm(span=spread_lkbk).std())
