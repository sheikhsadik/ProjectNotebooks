import numpy as np
import pandas as pd

from sklearn.covariance import GraphicalLassoCV
from sklearn.cluster import AgglomerativeClustering
from Backtester.MLHelper import Covariance
from Backtester.MLHelper import GapStatistic

import warnings
warnings.filterwarnings("ignore")


class ClusterGlasso:

    def __init__(self, df, train_lkbk, min_periods):

        self.df = df
        self.train_lkbk = train_lkbk
        self.min_periods = min_periods
        self.cov_mat = self.df[:self.train_lkbk].cov(min_periods=self.min_periods)
        self.cov_mdl = Covariance(self.cov_mat)
        self.corr_den = self.cov_mdl.get_denoised_corr(self.train_lkbk)
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
                clust_bool = self.hi_clust.labels_ == cluster
                if np.sum(clust_bool) > 1:
                    const = self.df.columns[clust_bool]
                    x = self.df.iloc[:self.train_lkbk][const].dropna(axis=0,
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
        else:
            print('Run get_clusters first...')