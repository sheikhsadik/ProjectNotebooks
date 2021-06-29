import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm


class OLS:

    def __init__(self, df, X_ids, y_id, missing='drop-fill'):

        self.X_ids = X_ids
        self.y_id = y_id
        self.missing = missing
        self.df = df

        if missing == 'drop-fill':
            self.df = self.df.dropna(axis=0, how='all').fillna(method='ffill').copy()

        self.N = self.df.shape[0]
        self.X_ = sm.add_constant(self.df[self.X_ids])
        self.X = self.X_.values
        self.y = self.df[self.y_id].values

    def fit(self, method, window=None):

        self.method = method
        self.window = window

        if method is not 'full':
            beta_arr, tStat_arr, rsq_arr, adjrsq_arr = self.rollexOLS(self.X, self.y, self.window, method=self.method)
            colIds = [['const'], self.X_ids]
            colIds_merged = list(itertools.chain(*colIds))

            self.betas = pd.DataFrame(beta_arr, index=self.df.index[self.window:], columns = colIds_merged)
            self.tStats = pd.DataFrame(tStat_arr, index=self.df.index[self.window:], columns = colIds_merged)
            self.r_squared = pd.DataFrame(np.hstack([rsq_arr, adjrsq_arr]), index=self.df.index[self.window:],
                                          columns = ['RSquared', 'AdjRSquared'])
            self.y_predict = self.X_.iloc[self.window:].multiply(self.betas, axis=0).sum(axis=1).to_frame().\
                rename(columns={0: 'Y_predict'})

        elif method is 'full':

            colIds = [['const'], self.X_ids]
            colIds_merged = list(itertools.chain(*colIds))

            beta_arr, tStat_arr, rsq_arr, adjrsq_arr = self.regress(self.X, self.y)
            self.stats_df = pd.DataFrame(np.vstack([beta_arr, tStat_arr]),
                                         columns = colIds_merged, index = ['Betas', 't-Stats'])
            self.stats_df.loc['RSquared'] = np.nan
            self.stats_df.loc['RSquared', 'const'] = rsq_arr
            self.stats_df.loc['AdjRSquared'] = np.nan
            self.stats_df.loc['AdjRSquared', 'const'] = adjrsq_arr
            self.y_predict = self.X_.dot(self.stats_df.loc['Betas'])


        return self

    def regress(self, X, y):

        k = X.shape[1]
        n = X.shape[0]

        b = np.linalg.inv(X.T @ X) @ X.T @ y
        y_hat = X @ b

        SSE = np.sum(((y - y_hat)**2))
        SST = np.sum(((y - np.mean(y))**2))
        r_squared = 1 - (SSE/SST)
        adj_r_squared = 1 - (1 - r_squared)*(n-1)/(n-k-1)

        var_eps = SSE/(n-k)
        vcov_mat = var_eps*np.linalg.inv(X.T @ X)
        se_b = np.sqrt(np.diag(vcov_mat))
        t_stats = b/se_b

        return b, t_stats, r_squared, adj_r_squared

    def rollexOLS(self, X, y, window, method='rolling'):

        ix_N = X.shape[0]
        K = X.shape[1]

        beta_arr = np.zeros(((ix_N - window), K)); tStat_arr = np.zeros(((ix_N - window), K))
        rsq_arr = np.zeros(((ix_N - window), 1)); adjrsq_arr = np.zeros(((ix_N - window), 1))

        if method is 'rolling':

            for ix in range(0, (ix_N-window)):
                b, t_stats, r_squared, adj_r_squared = self.regress(X[ix:(ix+(window+1)), :], y[ix: (ix+(window+1))])
                beta_arr[ix, :] = b; tStat_arr[ix, :] = t_stats
                rsq_arr[ix, :] = r_squared; adjrsq_arr[ix, :] = adj_r_squared

        elif method is 'expanding':

            for ix in range((window), ix_N):
                b, t_stats, r_squared, adj_r_squared = self.regress(X[0:ix, :], y[0:ix])
                beta_arr[(ix-window), :] = b; tStat_arr[(ix-window), :] = t_stats
                rsq_arr[(ix-window), :] = r_squared; adjrsq_arr[(ix-window), :] = adj_r_squared

        elif method is 'full':

            beta_arr, tStat_arr, rsq_arr, adjrsq_arr = self.regress(X, y)

        return beta_arr, tStat_arr, rsq_arr, adjrsq_arr













