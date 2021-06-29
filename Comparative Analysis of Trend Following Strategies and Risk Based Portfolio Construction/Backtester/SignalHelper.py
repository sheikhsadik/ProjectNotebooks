import warnings
warnings.filterwarnings("ignore")

import numpy as np
from datetime import datetime
import pandas as pd
from scipy.stats import geom
from scipy import stats

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR

import pykalman as pk
from ta.momentum import kama, rsi
from ta.volatility import BollingerBands

import Portfolio.PortHelper as ph
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

from numba import jit
import pdblp
# con = pdblp.BCon(debug=False, port=8194, timeout=50000)
# con.start()
lmts = importr('LongMemoryTS')
utils = importr('utils')

critical_values = {
    0: {
        .90: 13.4294,
        .95: 15.4943,
        .99: 19.9349
    },
    1: {
        .90: 2.7055,
        .95: 3.8415,
        .99: 6.6349
    }
}

def get_bbg_data(tickers, flds, startdt, enddt=None, frequency='DAILY'):

    if enddt is None:
        enddt = datetime.strftime(datetime.today(), '%Y%m%d')
    data = con.bdh(tickers, flds=flds, start_date=startdt,
                   end_date=enddt, elms=[('periodicitySelection', frequency)])
    data.columns = data.columns.droplevel(1)
    return data[tickers]


def moving_average(df, lkbk, method='sma'):

    if method is 'sma': return df.rolling(lkbk).mean()
    elif method is 'ema': return df.ewm(span=lkbk).mean()
    elif method is 'kama': return kama(df, lkbk,
                                       int(lkbk * 0.2),
                                       int(lkbk * 3.0))
    elif method is 'frama': return frama(df, batch=int(lkbk/ 2))
    else: print('Invalid method, please enter one of sma, ema, kama, frama')


def tsmom(asset, lkbk):
    tsmom = asset.apply(np.log).diff(1).rolling(lkbk).sum()
    return tsmom


def tsmom_smooth(asset, lkbk, smooth_period, smooth_method='sma', rx=False):

    global tsm_smooth
    P_s_t = moving_average(asset, smooth_period, method=smooth_method)
    P_s_tm = P_s_t.shift(lkbk)

    if rx is False: tsm_smooth = P_s_t - P_s_tm
    elif rx is True: tsm_smooth = (P_s_t - P_s_tm)/P_s_tm
    return tsm_smooth


def tsmom_diff(asset, lkbk):
    tsmom_diff = asset.diff(lkbk)
    return tsmom_diff


def mult_tsmom(asset, lkbks, weights):

    sig_lkbks = pd.concat([tsmom(asset, lkbk) for lkbk in lkbks], axis=1)

    if weights is None:
        sig_lkbks.columns = lkbks
        return sig_lkbks.dropna()
    elif weights is 'equal':
        return sig_lkbks.apply(np.sign).mean(axis=1).dropna()
    else:
        return sig_lkbks.apply(np.sign).dot(weights).dropna()


def mult_tsmomsmooth(asset, lkbks, smooth_period, smooth_method, rx, weights):

    sig_lkbks = pd.concat([tsmom_smooth(asset, lkbk, smooth_period,
                                        smooth_method, rx) for lkbk in lkbks], axis=1)

    if weights is None:
        sig_lkbks.columns = lkbks
        return sig_lkbks.dropna()
    elif weights is 'equal':
        return sig_lkbks.apply(np.sign).mean(axis=1).dropna()
    else:
        return sig_lkbks.apply(np.sign).dot(weights).dropna()


def ma_crossover(asset, lkbk_pair, smooth_method='sma'):

    ma_1 = moving_average(asset, lkbk_pair[0], method=smooth_method)
    ma_2 = moving_average(asset, lkbk_pair[1], method=smooth_method)
    mac_trend = 2 * (ma_2 - ma_1) / (lkbk_pair[1] - lkbk_pair[0])
    return -mac_trend


def breakout(asset, lkbk, smooth_method='sma'):
    if smooth_method is None:
        return asset.div(asset.shift(lkbk), axis=0) - 1
    else:
        return asset.div(moving_average(asset, lkbk,
                                        smooth_method).shift(1)) - 1


def avg_breakout(asset, lkbks, sign=True):

    breakout = pd.concat({lkbk: asset - asset.shift(lkbk) for lkbk in lkbks}, axis=1).dropna()
    if sign is True: breakout = breakout.apply(np.sign).mean(axis=1)
    elif not sign: breakout = breakout.mean(axis=1)
    return breakout


def maavg_breakout(asset, lkbks, smooth_method='sma', sign=True, prcnt=True):

    if prcnt is True:
        breakout = pd.concat({lkbk: (asset/moving_average(asset, lkbk, smooth_method).shift(1) - 1) \
                              for lkbk in lkbks}, axis=1).dropna()
    else:
        breakout = pd.concat({lkbk: asset - moving_average(asset, lkbk, smooth_method).shift(1) \
                              for lkbk in lkbks}, axis=1).dropna()
    if sign is True: breakout = breakout.apply(np.sign).mean(axis=1)
    elif not sign: breakout = breakout.mean(axis=1)
    return breakout


def ols_trend(asset, return_corr=False):
    t = np.arange(1, len(asset) + 1)
    slope, _, r, _, std_err = stats.linregress(t, np.log(asset))
    if return_corr:return r
    else: return slope/std_err


def coef_deter(prices):

    X = prices.dropna().div(prices.dropna().iloc[0])
    r2 = np.corrcoef(x=np.log(X.values.flatten()),
                     y=np.arange(1,
                                 len(X) + 1))[0, 1]**2
    return r2


def dvr(prices, t):

    r2 = coef_deter(prices)
    sharpe = sharpe_ratio(prices.pct_change(1), t)
    return sharpe * r2


def var_ratio(returns, alpha):
    return np.divide(returns.mean(), returns.quantile(alpha) * (-1))


def cvar_ratio(returns, alpha):
    return np.divide(returns.mean(),
                     returns[returns < returns.quantile(alpha)].\
                     abs().mean())


def maxloss_ratio(returns):
    return np.divide(returns.mean(), np.abs(returns.min()))


def avgdd_ratio(cum_returns):
    return np.divide(
        cum_returns.pct_change(1).mean(),
        drawdown(cum_returns.apply(np.log)).mean())


def hilo_diff(prices):

    del_h = np.divide(prices.max(), prices[-1])
    del_l = np.divide(prices[-1], prices.min())
    diff_h = np.divide(del_h, prices.diff().std())
    diff_l = np.divide(del_l, prices.diff().std())

    return np.subtract(diff_l, diff_h)


def gpr(returns):

    return np.divide(returns.loc[returns > 0].sum(),
                     returns.loc[returns < 0].abs().sum())


def omega_ratio(returns):
    mu = returns.mean()
    return np.divide(returns[returns > mu].sum(),
                     returns[returns < mu].abs().sum())


def ulcer_ratio(cum_returns):

    return np.divide(
        cum_returns.pct_change(1).mean(),
        (np.sqrt(drawdown(cum_returns.apply(np.log))**2).mean()))


def rolling_window(a, window_size):
    shape = (a.shape[0] - window_size + 1, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    return np.lib.stride_tricks.as_strided(a,
                                           shape=shape,
                                           strides=strides)


def frama(X, batch=10, sc=200):

    w = np.log(2/(sc+1))
    window = batch * 2
    n_nulls = X.isnull().sum()
    p = X.dropna()

    highs_int = p.rolling(batch).max()
    lows_int = p.rolling(batch).min()

    N1 = highs_int.subtract(lows_int) / batch
    N2 = N1.shift(batch)
    highs = p.rolling(window).max()
    lows = p.rolling(window).min()
    N3 = highs.subtract(lows) / window

    D = (np.log(N1 + N2) - np.log(N3)) / np.log(2)

    alpha = np.exp(w* (D - 1))
    alpha = np.clip(alpha, 0.01, 1.0).values
    frama_x = p.values
    for i, a in enumerate(alpha):
        cl = p.values[i]
        if i < window:
            continue
        frama_x[i] = cl * a + (1 - a) * frama_x[i - 1]

    return pd.Series(np.hstack([np.repeat(np.nan, n_nulls),
                                frama_x.flatten()]),
                     name=X.name,
                     index=X.index)


def eff_ratio(X, lkbk):
    return (X - X.shift(lkbk)).abs().\
            divide((X - X.shift(1)).abs().\
                   rolling(lkbk).sum(), axis=0)

def rsi_wrap(df, lkbk, window=14, smooth_method='sma'):

    if lkbk < window:
        raise ValueError('lookback period must be greater than window')

    n_nans = df.isnull().sum()

    ma_vec = moving_average(df, lkbk, method=smooth_method).shift(1)

    rsi_vec = rsi(df.dropna(), window=window)
    rsi_series = np.hstack([np.repeat(np.nan, n_nans), rsi_vec])

    signal = np.where((df.sub(ma_vec) > 0) & (rsi_series < 70),1, 0)
    signal = np.where((df.sub(ma_vec) < 0) & (rsi_series > 30),
                      -1, signal).astype(float)
    nan_ix = np.isnan(rsi_series)
    signal[nan_ix] = np.repeat(np.nan, np.sum(nan_ix))

    return pd.Series(signal, index=df.index, name=df.name)


def bollinger_bands_wrap(df, window, window_dev, signal=True):

    idx = df.name[0]
    df = df.to_frame('close')
    indicator_bb = BollingerBands(close=df['close'],
                                  window=window,
                                  window_dev=window_dev)

    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    if signal:
        df['sign'] = np.where((df['close'] > df['bb_bbm']) & \
                              (df['close'] < df['bb_bbh']), 1, 0)
        df['sign'] = np.where((df['close'] < df['bb_bbm']) & \
                              (df['close'] > df['bb_bbl']), -1,
                              df['sign'])
        bb = df['sign'].copy()
        bb.name = idx

        return bb
    else:
        return df


def macd(df, long, short, signal=9):

    ema_long = df.ewm(span=long).mean()
    ema_short = df.ewm(span=short).mean()
    macd = ema_short.sub(ema_long, axis=0)
    macd_signal = macd.sub(macd.ewm(span=signal).mean().shift(1))
    return macd_signal


def kalman_smoother(x, burn_in=21):

    n_nans = x.isnull().sum()
    x_ = x.dropna().values
    kf = pk.KalmanFilter(transition_matrices=[1], observation_matrices=[1],
                         initial_state_mean=0, initial_state_covariance=1,
                         observation_covariance=1, transition_covariance=0.01)
    state_mus, _ = kf.filter(x_)
    state_mus = state_mus
    # return np.hstack([np.repeat(np.nan, n_nans+burn_in),
    #                   state_mus.flatten()[burn_in:]])
    return pd.Series(np.hstack([np.repeat(np.nan, n_nans+burn_in),
                                state_mus.flatten()[burn_in:]]),
                     index=x.index, name=x.name)


@jit(nopython=True)
def hp_filter(X, lmbda=1e4):
    X = X[~np.isnan(X)]
    T = X.shape[0]
    K = np.zeros((T - 2, T))
    I = np.eye(T)
    for ix in range(K.shape[0]): K[ix, ix:ix + 3] = np.array([1, -2, 1])
    return (np.linalg.inv(I + lmbda * (K.T @ K)) @ X)[-1]


def ewma_risk(x, lmbda, tau):

    nan_ix = np.isnan(x)
    rx = x[~nan_ix]
    eps = np.array(ph.EWMAEpsSeries(lmbda, rx)).flatten()
    return np.hstack([x[nan_ix], np.sqrt(ph.EWMAVarSeries(lmbda, eps)) * np.sqrt(tau)])


def man_macross(X, coms, pw=63, sw=252):
    s_k = coms[0]
    l_k = coms[1]
    y_k = (X.ewm(com=s_k).mean().sub(X.ewm(com=l_k). \
                                     mean())).div(X.rolling(pw).std())
    z_k = y_k.div(y_k.rolling(sw).std())
    #     r_z = z_k.multiply((-(1/4)*(z_k**2)).apply(np.exp))/0.89
    return z_k


def cum_returns(returns):
    returns = returns.apply(lambda x: np.log(1+x))
    return (1 + returns.cumsum(axis=0))


def drawdown(cum_returns):
    eq_curve = np.exp(cum_returns)
    return eq_curve/eq_curve.cummax() - 1


def max_drawdown(cum_returns):
    drawdown_curve = drawdown(cum_returns)
    return np.max(np.abs(drawdown_curve))


def ann_mean(returns, t):
    return returns.mean()*t


def ann_vol(returns, t):
    return returns.std()*np.sqrt(t)


def skewness(returns, t):
    return returns.skew() * (1/np.sqrt(t))


def kurtosis(returns, t):
    return returns.kurtosis() * (1/t)


def sharpe_ratio(returns, t):
    return np.divide(returns.mean()*np.sqrt(t),
                     returns.std())


def t_ratio(returns):
    return returns.mean()*np.sqrt(len(returns.dropna()))/returns.std()


def sortino_ratio(returns, t):
    downside = returns[returns < 0]
    return np.divide(returns.mean()*np.sqrt(t), downside.std())


def calmar_ratio(returns, t):
    cum_rets = cum_returns(returns)
    try:
        return (returns.mean()*t)/max_drawdown(cum_rets)
    except:
        return np.nan


def get_corrs(y_df, x_df, start=None, end=None):

    if (start is not None) & (end is not None): y_df = y_df[start:end].copy()
    return pd.concat([pd.concat([y_df[idx], x_df], axis=1).dropna().corr().iloc[0, 1:]\
                      for idx in y_df.columns], axis=1)


def get_drawdown_stats(returns, t):

    cum_performance = cum_returns(returns)
    drawdown_curve = drawdown(cum_performance)
    max_dd = max_drawdown(cum_performance)
    ddmax_end = drawdown_curve[(drawdown_curve == -max_dd)].index[0]
    mdd_series = drawdown_curve.loc[:ddmax_end]
    ddmax_start = mdd_series.index[(mdd_series == 0.0).values][-1]

    maxdd_returns = returns[ddmax_start:ddmax_end][1:]
    maxdd_duration = len(maxdd_returns)
    maxdd_sharpe = maxdd_returns.mean()*np.sqrt(t)/maxdd_returns.std()

    maxdd_stats = pd.Series([max_dd, maxdd_duration/t, maxdd_sharpe],
                            index=['Max Drawdown',
                                   'Max Drawdown Duration',
                                   'Max Duration Sharpe'])

    return maxdd_stats


def get_stats(returns, t):

    agg_ids = ['Mean', 'Volatility', 'Skewness', 'Kurtosis', 'Strategy Sharpe Ratio',
               't-Stats', 'Sortino Ratio', 'Calmar Ratio']

    stats_agg = pd.Series([ann_mean(returns, t), ann_vol(returns, t), skewness(returns, t),
                           kurtosis(returns, t), sharpe_ratio(returns, t), t_ratio(returns),
                           sortino_ratio(returns, t), calmar_ratio(returns, t)], index=agg_ids)

    return stats_agg


def exponential_risk(X_df, risk_lkbk, tau):

    lmbda = (risk_lkbk - 1) / (risk_lkbk + 1)
    # apply(lambda x: ewma_risk(x.values, lmbda, tau))
    return X_df.ewm(span=risk_lkbk, min_periods=5).std()*np.sqrt(tau)


def rolling_risk(X_df, risk_lkbk, tau):
    return X_df.rolling(risk_lkbk).std().multiply(np.sqrt(tau), axis=0)


def tma(y):
    w_t = 1 / np.array(list(reversed(np.arange(1, len(y) + 1))))
    return np.sum(w_t * y) / np.sum(w_t)


def cutoff_rule(x, cutoffs, signs):

    sign_arr = np.zeros(x.shape)
    for ix in range(x.shape[0]):
        if x[ix] <= cutoffs[0]:
            sign_arr[ix] = signs[0]
        elif x[ix] >= cutoffs[1]:
            sign_arr[ix] = signs[-1]
        elif (x[ix] > cutoffs[0]) and (x[ix] < cutoffs[1]):
            sign_arr[ix] = 0
        elif (np.isnan(x[ix])) or (np.isinf(x[ix])):
            sign_arr[ix] = np.nan

    return sign_arr


def cs_rank_signal(x, buckets=[33, 66]):

    x_sigs = np.zeros(x.shape)
    x_rank = x[~np.isnan(x)]
    x_prcntile = [stats.percentileofscore(x_rank,
                                          val,
                                          kind='rank') \
                  for val in x_rank]
    x[~np.isnan(x)] = x_prcntile
    x_sigs[np.isnan(x)] = np.nan
    x_sigs[x <= buckets[0]] = -1
    x_sigs[x >= buckets[1]] = 1

    return x_sigs


@jit(nopython=True)
def get_sb_idx(L, M, T, N):

    """
    Computes the stationary block indices for each path
    based on the expected block length

    :param Eb: M x T array of sample block lengths
    :param M: Number of paths
    :param T: Number of periods per path
    :param N: Length of the original data
    :return: An array of indices for each path
    """
    Idx = np.zeros((M, T))

    for i in range(0, M):
        a = 0
        unif_rand = np.random.uniform(0, 1, T)
        for j in range(0, T):
            if (a > T - 1):
                break
            ii = int(N * unif_rand[j])
            for k in range(0, L[i, j]):
                Idx[i, a] = ii
                ii += 1
                if (ii > (N - 1)):
                    ii = (N - ii)
                a += 1
                if (a > T - 1):
                    break
    return Idx.T


def get_sb_samples(data, Eb=None, M=1000, T=252):

    """
    Given an array of returns, the function returns a N x k x M array
    with sampled paths based on Stationary block bootstrap

    :param data: N X k array of returns
    :param Eb: Expected block Length;
               heuristic approach: int(0.17 x T)
               or the authors optimal approach: http://www.math.ucsd.edu/~politis/SBblock-revER.pdf
    :param M: Number of paths
    :param T: Number of periods per path
    :return: A multi-dim array of sampled returns for each path
    """
    N = data.shape[0]
    k = data.shape[1]

    if Eb is None:
        prob = 1 / int(0.17 * T)
    else: prob = 1/Eb
    L = geom.rvs(prob, size=(M, T))
    mult_samp_arr = np.zeros((T, k, M))
    sb_idx = get_sb_idx(L, M, T, N).astype(int)
    for m in range(M): mult_samp_arr[:, :, m] = data[sb_idx[:, m], :]

    return mult_samp_arr



def get_R_s(x):

    r_t = x[1:] - x[:-1]
    mu = np.sum(r_t)/(len(r_t)-1)
    r_t_dm = r_t - mu
    y_t = r_t_dm.cumsum()
    R_n = np.max(y_t) - np.min(y_t)
    s_n = np.std(r_t, ddof=1)
    if (R_n != 0.0) & (s_n != 0.0):
        R_s = R_n/s_n
        return R_s
    else: return 0.0


def get_hurst_exponent(x, get_h=True):

    if len(x) < 21:
        raise ValueError("log price series length must be greater than or equal to 21")
    min_size = 10
    max_size = len(x) + 1
    window_sizes = list(map(lambda x: int(10**x),
                            np.arange(np.log10(min_size),
                                      np.log10(max_size),
                                      0.25)))
    window_sizes.append(len(x))
    R_s = []
    for w in window_sizes:
        rs = []
        for start in range(0, len(x), w):
            if (start+w) > len(x):
                break
            r_s = get_R_s(x[start:(start+w)])
            if r_s != 0:
                rs.append(r_s)
        R_s.append(np.mean(rs))
    H = np.polyfit(np.log10(window_sizes), np.log10(R_s), 1)[0]
    if get_h:
        return H
    else:
        C = 2**(2*H - 1) - 1
        return H, C, np.log10(R_s), np.log10(window_sizes)


def calc_half_life(x):
    y_t = np.subtract(x[1:], x[:-1])
    X_t = np.vstack([np.ones(len(x[:-1])), x[:-1]]).T

    coefs = np.linalg.inv(X_t.T @ X_t) @ X_t.T @ y_t
    return -np.log(2) / coefs[1]


def calc_coint_stats(X, results=True):

    var = VAR(X)
    lags = var.select_order()
    k_ar_diff = lags.selected_orders['aic']

    cj0 = coint_johansen(X, det_order=0, k_ar_diff=k_ar_diff)
    t1, p1 = coint(X[:, 0], X[:, 1], trend='c')[:2]
    t2, p2 = coint(X[:, 1], X[:, 0], trend='c')[:2]
    if results:
        return [
            t1, p1, t2, p2, k_ar_diff, *cj0.lr1
        ]
    else:
        return results, cj0


def test_coint_pairs(X, alpha=0.05):

    trace0_cv = critical_values[0][
        1 - alpha]  # critical value for 0 cointegration relationships
    trace1_cv = critical_values[1][
        1 - alpha]  # critical value for 1 cointegration relationship

    X = X.div(X.iloc[0])
    eps = 1e-5 * np.random.rand(X.shape[0], 2)
    X = (X.pct_change() + eps).apply(lambda x: 1 + x).fillna(1).cumprod()
    t1, p1, t2, p2, k_ar_diff, trace0, trace1 = calc_coint_stats(X.values)

    trace_sig = ((trace0 > trace0_cv) &(trace1 > trace1_cv)).astype(int)
    eg_sig = (p1 < alpha) & (p2 < alpha).astype(int)

    return trace_sig, eg_sig


def frac_coint_test(X, m_peri=25, alpha=0.05):
    """
    Semiparametric residual-based test for fractional
    cointegration by Chen, Hurvich (2003). Returns test
    statistic, critical value and testing decision.
    Null hypothesis: no fractional cointegration

    """

    numpy2ri.activate()
    n, k = X.shape
    X_r = robjects.r.matrix(X, nrow=n, ncol=k)

    coint_test = lmts.FCI_CH06(X_r,
                               m_peri=robjects.Vector(m_peri),
                               alpha=alpha,
                               m=robjects.Vector(np.floor(len(X_r) ** 0.65)))
    CH_dict = {}
    for name, val in coint_test.items():
        CH_dict[name] = np.array(val)[0]
    numpy2ri.deactivate()

    return CH_dict
