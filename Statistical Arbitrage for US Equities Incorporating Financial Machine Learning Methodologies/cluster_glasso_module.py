
# import warnings
# from sklearn.exceptions import ConvergenceWarning
# warnings.simplefilter("ignore", category=ConvergenceWarning)

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.covariance import GraphicalLassoCV
from sklearn.cluster import AgglomerativeClustering

from Backtester.MLHelper import Covariance
from Backtester.MLHelper import GapStatistic
from sklearn.model_selection import TimeSeriesSplit
ts_cv = TimeSeriesSplit(n_splits=4)

class ClusterGlasso:

    def __init__(self, df, train_lkbk, min_periods, orthogonal=True):

        self.df = df
        self.train_lkbk = train_lkbk
        self.min_periods = min_periods
        if orthogonal:
            self.df_train = ClusterGlasso.get_orthogonal_data(self.df[:self.train_lkbk])
        else:
            self.df_train = self.df[:self.train_lkbk]
        self.cov_mat = self.df_train.cov(min_periods=self.min_periods)
        self.cov_mdl = Covariance(self.cov_mat)
        self.corr_den = self.cov_mdl.get_denoised_corr(self.train_lkbk)
        self.cov_den = self.cov_mdl.corr_to_cov(self.corr_den, self.cov_mdl.vols)

        self.hi_clust = None
        self.pcorr_clust = {}
        self.pcorr_clust_rank = {}

    def get_glasso_pcorr(self, X):

        gl_model = GraphicalLassoCV(n_jobs=8, cv=ts_cv, max_iter=1000)
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
        cluster_mdl = AgglomerativeClustering(linkage='single')
        dist = self.cov_mdl.corr_distance(self.corr_den)
        if opt_k is None:
            gs = GapStatistic(cluster_mdl, dist)
            opt_k = gs.opt_k
        self.hi_clust = AgglomerativeClustering(n_clusters=opt_k).fit(dist)

        return self

    @staticmethod
    def get_orthogonal_data(df, p_skip=1):

        X = df.dropna().copy()
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        pca = PCA(random_state=42)
        pca.fit(X_scaled)
        prin_components = pca.transform(X_scaled)

        e_vec = pca.components_
        X_recon = prin_components[:, p_skip:] @ e_vec[p_skip:, :]
        X_rescaled = pd.DataFrame(scaler.inverse_transform(X_recon),
                                  index=X.index, columns=X.columns)
        return X_rescaled

    def get_cluster_partial_corr(self, corr_thresh=0.10):
        if self.hi_clust is not None:
            clusters = np.unique(self.hi_clust.labels_)
            for cluster in clusters:
                clust_bool = self.hi_clust.labels_ == cluster
                if np.sum(clust_bool) > 1:
                    const = self.df.columns[clust_bool]
                    x = self.df_train[const].dropna(axis=0, how='any')
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