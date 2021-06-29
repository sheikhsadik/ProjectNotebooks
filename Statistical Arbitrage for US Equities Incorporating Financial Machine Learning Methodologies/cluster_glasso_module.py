import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (12,10) # setting default figure size
pylab.style.use('ggplot') # setting style

from sklearn.covariance import GraphicalLassoCV
from sklearn.cluster import AgglomerativeClustering
from Backtester.MLHelper import Covariance
from Backtester.MLHelper import GapStatistic
import Backtester.SignalHelper as sh

from numba import jit
import warnings
warnings.filterwarnings("ignore")

########################################################################################
ret_df = pd.read_pickle('S:\\Github\\MyProjects\\MLStuff\\StatArb\\ret_df.pkl')
ret_df.drop(columns=['start', 'ending'], inplace=True)
ret_df = ret_df.pivot_table(values='ret',index='date', columns='permno')

start_date = '1960/01'
corr_lkbk = int(252*5)
min_periods = int(252*3)
test_l = int(252/2)
hr_lkbk = int(252)
spread_lkbk = int(21*3)
t_0 = 12000


ret_df = ret_df[start_date:].copy()
df = ret_df.iloc[t_0:(t_0 + corr_lkbk + test_l)].dropna(axis=1, how='all')
test_full_ids = df.iloc[corr_lkbk:].dropna(axis=1,how='any').columns
min_miss_train_ids = df.iloc[:corr_lkbk][::-1].iloc[:min_periods].isnull().sum()
train_full_ids = min_miss_train_ids.index[min_miss_train_ids == 0]
asset_ids = train_full_ids.intersection(test_full_ids)
df = df[asset_ids]
########################################################################################


class ClusterGlasso:

    def __init__(self, df, corr_lkbk, min_periods):

        self.df = df
        self.corr_lkbk = corr_lkbk
        self.min_periods = min_periods
        self.cov_mat = self.df[:self.corr_lkbk].cov(min_periods=self.min_periods)
        self.cov_mdl = Covariance(self.cov_mat)
        self.corr_den = self.cov_mdl.get_denoised_corr(self.corr_lkbk)
        self.cov_den = self.cov_mdl.corr_to_cov(self.corr_den, self.cov_mdl.vols)

        self.hi_clust = None
        self.pcorr_clust = {}
        self.pcorr_clust_rank = {}

    def get_glasso_pcorr(self, X):
        gl_model = GraphicalLassoCV(n_jobs=10,max_iter=500)
        gl_model.fit(X)
        inv_cov = gl_model.precision_
        par_corr_mat = self.get_inv_pcorr(inv_cov)
        return gl_model, par_corr_mat

    def get_inv_pcorr(self, inv_cov):
        diag_inv = np.sqrt(np.diag(inv_cov).reshape(-1, 1)) ** (-1)
        a = diag_inv @ diag_inv.T
        pcorr_mat = -(a * inv_cov)
        np.fill_diagonal(pcorr_mat, 1)
        return pcorr_mat

    def get_clusters(self, opt_k=None):
        cluster_mdl = AgglomerativeClustering()
        dist = self.cov_mdl.corr_distance(self.corr_den)
        if opt_k is None:
            gs = GapStatistic(cluster_mdl, dist)
            opt_k = gs.opt_k
        self.hi_clust = AgglomerativeClustering(n_clusters=opt_k).fit(dist)

        return self

    def get_cluster_partial_corr(self, corr_thresh=0.10):
        if self.hi_clust is not None:
            clusters = np.unique(self.hi_clust.labels_)
            for cluster in clusters:
                const = df.columns[self.hi_clust.labels_ == cluster]
                x = self.df.iloc[:self.corr_lkbk][const].dropna(axis=0,
                                                                how='any')
                _, pcorr_mat = self.get_glasso_pcorr(x.values)
                self.pcorr_clust['cluster_{}'.format(cluster)] = \
                    pd.DataFrame(pcorr_mat, index=const, columns=const)
                pair_df = self.cov_mdl.\
                    get_sorted_paircorr(self.pcorr_clust['cluster_{}'.format(cluster)])

                self.pcorr_clust_rank['cluster_{}'.format(cluster+1)] = \
                    pair_df[pair_df['corr'] >= corr_thresh].groupby('A1')['A2'].\
                        apply(lambda l: l.values.tolist()).to_dict()

            return self.pcorr_clust, self.pcorr_clust_rank
        else: print('Run get_clusters first...')



##### backtest module ################
@jit(nopython=True)
def get_trade_status(signal, stop_signal):
    status = [0]
    for ix in range(1, len(signal)):
        if (status[ix - 1] == 0) | \
                ((status[ix - 1] == 0) & (stop_signal[ix - 1] == 1)):
            status.append(signal[ix])
        elif (status[ix - 1] != 0) & (stop_signal[ix] == 1):
            status.append(0)
        elif (status[ix - 1] != 0) & (signal[ix] != 0):
            status.append(signal[ix])
        else:
            status.append(status[ix - 1])

    return status

# pair_selection_module


def get_cluster_glasso(df, corr_lkbk, min_periods, corr_thresh=0.10):

    clust_glasso = ClusterGlasso(df, corr_lkbk, min_periods)
    pcorr_clust, pcorr_clust_rank = clust_glasso.get_clusters(opt_k=5).\
        get_cluster_partial_corr(corr_thresh=corr_thresh)
    return pcorr_clust, pcorr_clust_rank


def pair_pass_test(train_df, train_residuals):

    coint_test = sh.test_coint_pairs(train_df)
    frac_coint_test = sh.frac_coint_test(train_df.values)['reject']
    hurst = sh.get_hurst_exponent(train_residuals)
    h_life = sh.calc_half_life(train_residuals.flatten())
    if np.any([*coint_test, frac_coint_test]):
        if hurst < 0.50:
            if h_life < test_l:
                return [y_id, X_id]
            else: return []
        else: return []
    else: return []


def form_pairs():

    pcorr_clust, pcorr_clust_rank = get_cluster_glasso(df, corr_lkbk, min_periods)

    selected_pairs = []
    hedge_weights_dict = {}
    z_spreads_dict = {}
    for cluster in pcorr_clust_rank.keys():

        if not pcorr_clust_rank[cluster]:
            continue
        for y_id in pcorr_clust_rank[cluster].keys():

            for X_id in pcorr_clust_rank[cluster][y_id]:

                permnos = [y_id, X_id]
                pair = df[permnos].dropna(axis=0, how='any'). \
                    apply(lambda x: 1 + x).cumprod()*1e3
                train_df = pair.iloc[:corr_lkbk]

                hedge_weights = get_hedge_ratio(pair,
                                                permnos,
                                                'tls',
                                                hr_lkbk=hr_lkbk)
                spreads = get_spreads(pair,
                                      permnos,
                                      hedge_weights)
                test = pair_pass_test(train_df,
                                      spreads.iloc[:corr_lkbk].values)

                if test:
                    z_spreads_dict['{}_{}'.format(y_id, X_id)] = get_ewm_zscore(spreads,
                                                                                spread_lkbk)
                    hedge_weights_dict['{}_{}'.format(y_id, X_id)] = hedge_weights.copy()
                    selected_pairs.append(test)




# spread module
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


pcorr_clust, pcorr_clust_rank = get_cluster_glasso(df, corr_lkbk, min_periods)

selected_pairs = []
hedge_weights_dict = {}
z_spreads_dict = {}
for cluster in pcorr_clust_rank.keys():

    if not pcorr_clust_rank[cluster]:
        continue
    for y_id in pcorr_clust_rank[cluster].keys():

        for X_id in pcorr_clust_rank[cluster][y_id]:

            permnos = [y_id, X_id]
            pair = df[permnos].dropna(axis=0, how='any'). \
                apply(lambda x: 1 + x).cumprod()*1e3
            train_df = pair.iloc[:corr_lkbk]

            spread_m = SpreadModule(pair, permnos)
            hedge_weights = spread_m.get_hedge_ratio('tls', hr_lkbk=hr_lkbk)
            spreads = spread_m.get_spreads()

            test = pair_pass_test(train_df,
                                  spreads.iloc[:corr_lkbk].values)

            if test:
                z_spreads_dict['{}_{}'.format(y_id, X_id)] = spread_m.get_ewm_zscore(spread_lkbk)
                hedge_weights_dict['{}_{}'.format(y_id, X_id)] = hedge_weights.copy()
                selected_pairs.append(test)

spread_m = SpreadModule(pair, permnos)
hedge_weights = spread_m.get_hedge_ratio('tls', hr_lkbk=hr_lkbk)
spreads = spread_m.get_spreads()
