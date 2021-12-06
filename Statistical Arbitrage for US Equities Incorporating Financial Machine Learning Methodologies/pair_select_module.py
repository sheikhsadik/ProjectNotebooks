import numpy as np
import pandas as pd

import Backtester.SignalHelper as sh
from StatArb.spread_module import SpreadModule
from StatArb.cluster_glasso_module import ClusterGlasso

class PairSelect:
    def __init__(self, df, train_lkbk=int(252*5), test_lkbk=int(252*2),
                 min_periods=int(252*3), opt_k=5,
                 hr_lkbk=int(252), spread_lkbk=int(21*3)):

        self.df = df
        self.train_lkbk = train_lkbk
        self.test_lkbk = test_lkbk
        self.min_periods = min_periods
        self.opt_k = opt_k
        self.hr_lkbk = hr_lkbk
        self.spread_lkbk = spread_lkbk

        self.selected_pairs = []
        self.hedge_weights_dict = {}
        self.spreads_dict = {}
        self.z_spreads_dict = {}
        self.pcorr_clust_rank = pd.DataFrame()

    def get_cluster_glasso(self, corr_thresh=0.10):
        clust_glasso = ClusterGlasso(self.df, self.train_lkbk, self.min_periods)
        pcorr_clust, pcorr_clust_rank = clust_glasso.get_clusters(opt_k=self.opt_k).\
            get_cluster_partial_corr(corr_thresh=corr_thresh)
        return pcorr_clust, pcorr_clust_rank

    def pair_pass_test(self, train_df, train_residuals):
        coint_test = sh.test_coint_pairs(train_df)
        frac_coint_test = sh.frac_coint_test(train_df.values)['reject']
        hurst = sh.get_hurst_exponent(train_residuals)
        h_life = sh.calc_half_life(train_residuals.flatten())
        if np.any([*coint_test, frac_coint_test]):
            if hurst < 0.50:
                if h_life < self.test_lkbk:
                    return train_df.columns.to_list()
                else: return []
            else: return []
        else: return []

    def form_pairs(self, corr_thresh=0.10, hr_method='tls', max_pairs=20):

        _, self.pcorr_clust_rank = self.get_cluster_glasso(corr_thresh=corr_thresh)

        for cluster in self.pcorr_clust_rank.keys():

            if not self.pcorr_clust_rank[cluster]:
                continue
            for y_id in self.pcorr_clust_rank[cluster].keys():

                for X_id in self.pcorr_clust_rank[cluster][y_id]:

                    permnos = [y_id, X_id]
                    pair = self.df[permnos].dropna(axis=0, how='any'). \
                               apply(lambda x: 1 + x).cumprod().dropna(axis=0).apply(np.log) * 1e3
                    train_df = pair.iloc[:self.train_lkbk ]

                    spread_m = SpreadModule(pair, permnos)
                    hedge_weights = spread_m.get_hedge_ratio(hr_method, hr_lkbk=self.hr_lkbk)
                    spreads = spread_m.get_spreads()
                    try:
                        test = self.pair_pass_test(train_df,
                                                   spreads.iloc[:self.train_lkbk].values)
                    except:
                        continue
                    if test:
                        self.spreads_dict['{}_{}'.format(y_id, X_id)] = spreads.copy()/1e3
                        self.z_spreads_dict['{}_{}'.format(y_id, X_id)] = spread_m.get_ewm_zscore(self.spread_lkbk)
                        self.hedge_weights_dict['{}_{}'.format(y_id, X_id)] = hedge_weights.copy()
                        self.selected_pairs.append(test)
        if len(self.selected_pairs) > max_pairs:
            self.selected_pairs = self.selected_pairs[:max_pairs]

