import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.stats as scs
from scipy.stats import norm
from sklearn.neighbors.kde import KernelDensity
from sklearn.neighbors import kde
from sklearn.metrics import pairwise_distances

from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)

rand_matrix = np.random.rand(1000,1000,50)

class MetaLabels:

    def __init__(self, df=None, signal=None, span=None, n_days=None, h=None, sl_pt=None):

        self.df = df
        self.signal = signal
        self.span = span
        self.n_days = n_days
        self.h = h
        self.sl_pt = sl_pt

    def generate_labels(self):
        self.time_events = cusum_filter(self.df, self.h)
        target = df.pct_change().ewm(span=self.span).std()
        self.target = target.loc[self.time_events]
        self.t1 = vertical_barrier(self.df, self.n_days)
        self.events = get_events(self.df, self.time_events,
                                 self.sl_pt, self.target,
                                 self.signal, ret_thresh=self.h,
                                 t1=self.t1)

        self.out = get_bins(self.events, self.df)
        return self


    def cusum_filter(self, df, h, ret_price=False):
        time_events = []
        s_pos = 0
        s_neg = 0

        del_price = df.apply(np.log).diff()
        for ix in del_price.index[1:]:
            s_pos = max(0, s_pos + del_price.loc[ix])
            s_neg = min(0, s_neg + del_price.loc[ix])
            if s_neg < -h:
                s_neg = 0
                time_events.append(ix)
            elif s_pos > h:
                s_pos = 0
                time_events.append(ix)

        time_events = pd.DatetimeIndex(time_events)
        if ret_price:
            return df[time_events]
        else:
            return time_events

    def vertical_barrier(self, df, n_days):
        t1 = df.index.searchsorted(time_events + pd.Timedelta(days=n_days))
        t1 = t1[t1 < df.shape[0]]
        t1 = pd.Series(df.index[t1], index=time_events[:t1.shape[0]])

        return t1

    def triple_barrier_label(self, df, events, sl_pt, molecule):
        events_ = events.loc[molecule]
        out = events_[['t1']].copy(deep=True)
        out['lower'] = out['upper'] = pd.NaT
        if sl_pt[0] > 0:
            upper = sl_pt[0] * events_['target']
        else:
            upper = pd.Series(index=events.index)

        if sl_pt[1] > 0:
            lower = -sl_pt[1] * events_['target']
        else:
            lower = pd.Series(index=events.index)

        for loc, t1 in events_['t1'].fillna(df.index[-1]).iteritems():
            df_0 = df[loc:t1]
            df_0 = (df_0 / df[loc] - 1) * events_.loc[loc, 'signal']

            out.loc[loc, 'lower'] = df_0[df_0 < lower[loc]].index.min()
            out.loc[loc, 'upper'] = df_0[df_0 > upper[loc]].index.min()

        return out

    def get_events(self, df, time_events, sl_pt, target, signal, ret_thresh, t1=False):
        target = target.loc[time_events]
        target = target[target > ret_thresh]

        if t1 is False: t1 = pd.Series(pd.NaT, index=time_events)
        if signal is None:
            signal = pd.Series(1., index=target.index)
            sl_pt = [sl_pt[0], sl_pt[0]]
        else:
            signal = signal[target.index]
            sl_pt = sl_pt[:2]

        events = pd.concat({
            't1': t1,
            'target': target,
            'signal': signal
        }, axis=1).dropna(subset=['target'])
        molecule = events.index
        df_0 = triple_barrier_label(df, events, sl_pt, molecule)
        events['t1'] = df_0.dropna(how='all').min(axis=1)
        if signal is None:
            events = events.drop('signal', axis=1)

        return events

    def barrier_indicator(self, out_df):

        store = []
        for ix in range(len(out_df)):
            date = out_df.index[ix]
            ret = out_df.loc[date, 'ret']
            tgt = out_df.loc[date, 'target']

            if (ret > 0.0) and (ret > tgt):
                store.append(1)
            elif (ret < 0.0) and (ret < -tgt):
                store.append(-1)
            else:
                store.append(0)
        out_df['bin'] = store

        return out_df

    def get_bins(self, events, df):

        # align prices with events
        events_ = events.dropna(subset=['t1'])
        prices = events_.index.union(events['t1'].values).drop_duplicates()
        prices = df.reindex(prices, method='bfill')

        # create data frame for labels
        out_df = pd.DataFrame(index=events_.index)
        out_df['ret'] = np.log(prices[events_['t1'].values].values) - np.log(prices.loc[events_.index])
        out_df['target'] = events_['target'].copy()
        # meta label returns
        if 'signal' in events_:
            out_df['ret'] = out_df['ret'] * events['signal']

        # vertical barrier
        out_df = barrier_indicator(out_df)
        # label incorrect events with zeros
        if 'signal' in events_:
            out_df['gl'] = out_df['bin'].copy()
            out_df.loc[out_df['gl'] <= 0, 'gl'] = 0
            out_df['signal'] = events['signal']

        # change to simple returns
        out_df['ret'] = np.exp(out_df['ret']) - 1

        return out_df



class Covariance:

    def __init__(self, cov):
        self.cov_mat = cov
        self.vols = self.diag_vols(self.cov_mat)
        self.corr_mat = self.cov_to_corr(self.cov_mat)


    def get_pca(self,C):

        eig_val, eig_vec = np.linalg.eigh(C)
        ix = eig_val.argsort()[::-1]
        eig_val, eig_vec = np.diagflat(eig_val[ix]), eig_vec[:, ix]
        return eig_val, eig_vec


    def marcenko_pastor_pdf(self, var, q, pts=1000):

        eig_min = var * (1 - np.sqrt(q)) ** 2
        eig_max = var * (1 + np.sqrt(q)) ** 2

        eig_val = np.linspace(eig_min, eig_max, pts)
        pdf = q * np.sqrt((eig_max - eig_val) * (eig_val - eig_min)) * (1 / (2 * np.pi * eig_val * var))
        return pdf, eig_val


    def fit_kde(self, x_fit, bw, x_eval=None):
        kde = KernelDensity(bandwidth=bw).fit(x_fit.reshape(-1, 1))
        if x_eval is None: x_eval = np.unique(x_fit).reshape(-1, 1)
        if len(x_eval.shape) == 1: x_eval = x_eval.reshape(-1, 1)
        if np.any(np.isnan(x_eval)):
            x_eval[:] = 0
        pdf = np.exp(kde.score_samples(x_eval))
        return pdf


    def eps_pdfs(self,var, eig_val, q, bw, pts=1000):
        pdf_theo, eig_val_theo = self.marcenko_pastor_pdf(var, q, pts)
        pdf_emp = self.fit_kde(eig_val, bw, eig_val_theo)
        return np.sum((pdf_theo - pdf_emp) ** 2)


    def get_max_eig_val(self, eig_val, q, bw=0.25):
        opt_var = minimize(lambda *x: self.eps_pdfs(*x), 0.5,
                           args=(eig_val, q, bw,), bounds=((1e-5, 1 - 1e-5),))
        if opt_var['success']:
            var = opt_var.x
        else:
            var = 1
        eig_max = var * (1 + np.sqrt(q)) ** 2
        return eig_max, var


    def get_denoised_corr(self,  M=60000):

        q = self.corr_mat.shape[0] / M
        eig_val, eig_vec = self.get_pca(self.corr_mat)

        eig_max, var = self.get_max_eig_val(np.diag(eig_val), 1, bw=0.01)
        n_replace = int(eig_val.shape[0] - np.diag(eig_val)[::-1].searchsorted(eig_max))

        eig_val_ = np.diag(eig_val).copy()
        eig_val_[n_replace:] = eig_val_[n_replace:].sum() / (eig_val_.shape[0] - n_replace)
        eig_val_ = np.diag(eig_val_)

        corr_new = eig_vec @ eig_val_ @ eig_vec.T
        sds = np.sqrt(np.diag(corr_new))
        corr_denoised = corr_new / np.outer(sds, sds)
        corr_denoised[corr_denoised < -1] = -1
        corr_denoised[corr_denoised > 1] = 1

        return corr_denoised


    def corr_distance(self, corr):
        return ((1 - corr) / 2)**(0.5)


    def cov_to_corr(self,cov):
        vols = np.sqrt(np.diag(cov)).reshape(-1, 1)
        corr = ((1 / vols) @ (1 / vols).T) * cov
        return corr


    def corr_to_cov(self,corr, vols):
        vols = vols.reshape(-1, 1)
        return ((vols) @ (vols).T) * corr


    def diag_vols(self,cov):
        return np.sqrt(np.diag(cov))


    def factor_cov_decomp(self, n_pc):
        e, v = np.linalg.eigh(self.cov_mat)
        idx = e.argsort()[::-1]
        e = e[idx]
        v = v[:, idx]
        L = v[:, :n_pc] @ np.diag(np.sqrt(e[:n_pc]))
        syst_cov = L @ L.T
        idio_cov = self.cov_mat.values - syst_cov
        return syst_cov, idio_cov


    def get_sorted_paircorr(self,corr_mat):

        triu_ix = np.triu_indices_from(corr_mat, 1)
        pair_df = pd.DataFrame(index=range(len(triu_ix[0])))
        pair_df['A1'] = list(corr_mat.columns[triu_ix[0]])
        pair_df['A2'] = list(corr_mat.columns[triu_ix[1]])
        pair_df['corr'] = corr_mat.values[triu_ix]
        #     pair_df['corr'] = ((1 - pair_df['corr']) / 2)**(0.5)
        pair_df.sort_values('corr', ascending=False, inplace=True)
        return pair_df


class GapStatistic:
    def __init__(self, model, dist, k_max=15, n_randoms=30):

        self.model = model
        self.dist = dist
        self.k_max = k_max
        self.n_randoms = n_randoms
        gap, clust_measure, opt_k = self.gap_statistic(self.model,
                                                       self.dist,
                                                       k_max=self.k_max,
                                                       n_randoms=self.n_randoms)
        self.gap = gap
        self.clust_measure = clust_measure
        self.opt_k = opt_k

    def inertia(self, labels, data):
        W = [
            np.mean(pairwise_distances(data[labels == label]))
            for label in np.unique(labels)
        ]
        return np.mean(W)


    def gap_statistic(self, model, data, k_max=10, n_randoms=10):
        e_w_k_rand = []
        sd_k = []
        w_k = []
        gap = []
        for k in range(1, k_max + 1):
            w_k_rand = []
            model.n_clusters = k
            for ix in range(n_randoms):
                rand_data = rand_matrix[:data.shape[0], :data.shape[1], ix]
                np.fill_diagonal(rand_data, 0)
                labels = model.fit_predict(rand_data)
                w_k_rand.append(np.log(self.inertia(labels, rand_data)))
            e_w_k_rand.append(np.mean(w_k_rand))
            sd_k.append(np.std(w_k_rand) * np.sqrt(1 + 1 / n_randoms))
            data_labels = model.fit_predict(data)
            w_k.append(self.inertia(data_labels, data))
            gap.append(e_w_k_rand[k - 1] - w_k[k - 1])

        clust_measure = (np.array(gap)[:-1] - (np.array(gap)[1:] - np.array(sd_k)[1:]))
        clust_range = np.arange(1, k_max)
        opt_k = clust_range[clust_measure > 0][0]

        return gap, clust_measure, opt_k
