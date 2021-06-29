import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import Backtester.SignalHelper as sh

import dask.dataframe as dd
from dask.multiprocessing import get
import multiprocessing
import sys

n_parts = int(multiprocessing.cpu_count() * 1.5)
sys.modules['__main__'].__spec__ = None

class signal_init:

    def __init__(self):

        self.signal_final = None
        self.signal_raw = None

    def sigprocess_aggsignals(self, weights='equal', signal_override=True):

        if signal_override is False: self.signal_final = self.signal_raw.copy()

        if self.signal_final.shape[1] is 1:
            raise ValueError('Signal must be more than one dimensional to aggregate')

        if weights is 'equal':
            self.signal_agg = self.signal_final.mean(axis=1)
        else:
            if type(weights) is np.ndarray:
                self.signal_agg = self.signal_final.dot(weights)
            else:
                print("No Array of weights founds, pass an array of weights!")

        self.signal_final = pd.DataFrame(self.signal_agg, columns=['RX_Asset'])

    def sigprocess_tszscore(self, sd_lkbk, method='rolling', demean=True,
                            clip=(None, None), signal_override=True):

        if signal_override is False: self.signal_final = self.signal_raw.copy()

        if method is 'rolling':
            if demean is True:
                X_dm = self.signal_final - self.signal_final.rolling(sd_lkbk).mean()
                Z = (X_dm / self.signal_final.rolling(sd_lkbk).std()).dropna(how='all')

            elif demean is False:
                Z = (self.signal_final / self.signal_final.rolling(sd_lkbk).std()).dropna(how='all')

        elif method is 'expanding':
            if demean is True:
                X_dm = self.signal_final - self.signal_final.expanding(sd_lkbk).mean()
                Z = (X_dm / self.signal_final.expanding(sd_lkbk).std()).dropna(how='all')

            elif demean is False:
                Z = (self.signal_final / self.signal_final.expanding(sd_lkbk).std()).dropna(how='all')

        self.sp_zscore = Z.clip(clip[0], clip[1])
        self.signal_final = self.sp_zscore.copy()

    def sigprocess_cszscore(self, clip=(None, None), signal_override=True):

        if signal_override is False:
            self.signal_final = self.signal_raw.copy()

        Z = self.signal_final.sub(self.signal_final.mean(axis=1), axis=0).\
            div(self.signal_final.std(axis=1), axis=0)

        self.sp_zscore = Z.clip(clip[0], clip[1])
        self.signal_final = self.sp_zscore.copy()

    def sigprocess_percentiles(self, method='full', lkbk=None, signal_override=True):

        if signal_override is False: self.signal_final = self.signal_raw
        if (method is not 'full') and (lkbk is None):
            raise ValueError('Please provide lookback period')

        if method is 'full':
            X_prcntile = self.signal_final.rank(pct=True)
        elif method is 'rolling':
            X_prcntile = self.signal_final.rolling(lkbk).apply(lambda x: x.rank(pct=True)[-1])
        elif method is 'expanding':
            X_prcntile = self.signal_final.expanding(lkbk).apply(lambda x: x.rank(pct=True)[-1])

        self.signal_final = X_prcntile

    def sigprocess_qcut(self, method='full', lkbk=None, nbins=None, signal_override=True):

        if signal_override is False: self.signal_final = self.signal_raw
        if nbins is None: nbins = 5
        if (method is not 'full') and (lkbk is None):
            raise ValueError('Please provide lookback period')

        if method is 'full':
            X_binned = self.signal_final.apply(pd.qcut, args=(nbins, False, False, 3, 'drop'))
        elif method is 'rolling':
            X_binned = self.signal_final.rolling(lkbk). \
                apply(lambda x: pd.qcut(x, nbins, False, False, 3, 'drop')[-1])
        elif method is 'expanding':
            X_binned = self.signal_final.expanding(lkbk). \
                apply(lambda x: pd.qcut(x, nbins, False, False, 3, 'drop')[-1])

        self.signal_final = X_binned

    def sigprocess_quantiles(self, method, lkbk, prcntile_list=None, signal_override=True):

        if prcntile_list is None: prcntile_list = [0.20, 0.40, 0.60, 0.80]
        if signal_override is False: self.signal_final = self.signal_raw

        if method not in ['expanding', 'rolling']:
            raise ValueError('Please provide appropriate method, one of expanding or rolling')

        if method is 'expanding':
            X_window = self.signal_final.expanding(lkbk)
        if method is 'rolling':
            X_window = self.signal_final.rolling(lkbk)

        X_qs = pd.concat({prcntile: X_window.quantile(prcntile) for prcntile in prcntile_list}, axis=1)
        self.signal_final = X_qs

    def sigprocess_masmooth(self, smooth_lkbk, smooth_method='rolling', signal_override=True):

        if signal_override is False: self.signal_final = self.signal_raw.copy()

        if smooth_method is 'rolling':
            X = self.signal_final.rolling(smooth_lkbk).mean()
        elif smooth_method is 'exponential':
            X = self.signal_final.ewm(span=smooth_lkbk).mean()
        elif smooth_method is 'time':
            X = self.signal_final.rolling(smooth_lkbk).apply(sh.tma)

        self.signal_final = X.copy()

    def sigprocess_sizefunc(self, d_type='unbounded', z_clip=None,
                            cutoff_args=None, signal_override=True):

        if signal_override is False: self.signal_final = self.signal_raw.copy()
        if (d_type is 'custom') and (cutoff_args is None):
            raise ValueError("Please provide cutoff arguments with 'cutoffs' and corresponding 'sign'")

        if d_type is 'unbounded':
            if cutoff_args is not None:
                sp_sign = self.signal_final.apply(lambda x: sh.cutoff_rule(x,
                                                                           cutoff_args['cutoffs'],
                                                                           cutoff_args['sign']), axis=0)
            else:
                sp_sign = np.sign(self.signal_final)

        elif d_type is 'percentile':
            sp_sign = self.signal_final.apply(lambda x: -1 + 2 * x)

        elif d_type is 'zscore':
            if z_clip is None: raise ValueError('Please provide Z clip')
            sp_sign = self.signal_final.apply(lambda x: x.clip(-z_clip, z_clip) / z_clip)

        self.signal_final = sp_sign.copy()

    def sigprocess_resampler_misc(self, resample_freq='B', fna_method='ffill', multiplier=1.0,
                                  smooth_lkbk=None, signal_override=True):

        if signal_override is False: self.signal_final = self.signal_raw.copy()

        if resample_freq is None:
            X_resamp = self.signal_final * multiplier
        else:
            X_resamp = self.signal_final.resample(resample_freq).fillna(method=fna_method) * multiplier

        if smooth_lkbk is None:
            self.signal_resamp_z = X_resamp
        else:
            self.signal_resamp_z = X_resamp.rolling(smooth_lkbk).mean()

        self.signal_final = self.signal_resamp_z.copy()

    def sigprocess_cs_rankweights(self,signal_override=True):

        if signal_override is False: self.signal_final = self.signal_raw.copy()
        rank_df = self.signal_final.rank(axis=1)
        rank_dev = rank_df.sub(rank_df.mean(axis=1), axis=0)
        c_f = 2 / rank_dev.apply(lambda x: np.sum(np.abs(x)), axis=1)
        self.signal_final = rank_dev.multiply(c_f, axis=0)

    def sigprocess_cs_ranksignal(self, buckets=[33,66]):

        ddf = dd.from_pandas(self.signal_final, npartitions=n_parts)
        signal_df = ddf.map_partitions(lambda df: df.apply(sh.cs_rank_signal,
                                                         args=(buckets,),
                                                         axis=1,
                                                         raw=True)). \
            compute(scheduler='processes')
        self.signal_final = signal_df.copy()

    def signal_mapper(self, sig_assets, lkbk, func, args=(), raw=False):

        return sig_assets.map_overlap(lambda df: df.rolling(lkbk,min_periods=5). \
                              apply(func, args=args, raw=raw), lkbk, 0). \
                              compute(scheduler='processes')

    def gen_signal(self):

        raise NameError("No signal class/ signal DataFrame found!")


class Signals:
    class signal_userdefined(signal_init):

        def __init__(self, signal_raw):

            if signal_raw.index.dtype == 'datetime64[ns]':
                self.signal_raw = signal_raw
                self.signal_final = self.signal_raw.copy()
            else:
                raise TypeError("Signal index should be in datetime format!")

            # print('Raw signal processed!')

    class signal_mom_tsmom(signal_init):

        def __init__(self, sig_assets, lkbk):
            self.sig_assets = sig_assets
            self.lkbk = lkbk

        def gen_signal(self):
            self.signal_raw = self.sig_assets.apply(lambda x: sh.tsmom(x, self.lkbk))
            self.signal_final = self.signal_raw.copy()

            # print('Raw signal processed!')

            return self

    class signal_mom_tsmomsmooth(signal_init):

        def __init__(self, sig_assets, lkbk, smooth_period, smooth_method, rx=False):
            self.sig_assets = sig_assets
            self.lkbk = lkbk
            self.smooth_period = smooth_period
            self.smooth_method = smooth_method
            self.rx = rx

        def gen_signal(self):
            self.signal_raw = self.sig_assets.apply(lambda x: sh.tsmom_smooth(x, self.lkbk,
                                                                              self.smooth_period,
                                                                              self.smooth_method,
                                                                              self.rx))
            self.signal_final = self.signal_raw.copy()

            # print('Raw signal processed!')

            return self

    class signal_mom_mult_tsmom(signal_init):

        def __init__(self, sig_assets, lkbks, weights=None):

            self.sig_assets = sig_assets
            self.lkbks = lkbks
            self.weights = weights

            if (self.sig_assets.shape[1] > 1) & (self.weights is None):
                raise ValueError('Please provide array weights!')

        def gen_signal(self):

            if self.sig_assets.shape[1] > 1:
                self.signal_raw = self.sig_assets.apply(lambda x: sh.mult_tsmom(x, self.lkbks,
                                                                                self.weights))
            else:
                self.signal_raw = sh.mult_tsmom(self.sig_assets,
                                                self.lkbks,
                                                self.weights)
            self.signal_final = self.signal_raw.copy()

            # print('Raw signal processed!')

            return self

    class signal_mom_mult_tsmomsmooth(signal_init):

        def __init__(self, sig_assets, lkbks, smooth_period, smooth_method, rx=False, weights=None):

            self.sig_assets = sig_assets
            self.lkbks = lkbks
            self.smooth_period = smooth_period
            self.smooth_method = smooth_method
            self.rx = rx
            self.weights = weights

            if (self.sig_assets.shape[1] > 1) & (self.weights is None):
                raise ValueError('Please provide array weights!')

        def gen_signal(self):

            if self.sig_assets.shape[1] > 1:
                self.signal_raw = self.sig_assets.apply(lambda x: sh.mult_tsmomsmooth(x, self.lkbks,
                                                                                      self.smooth_period,
                                                                                      self.smooth_method,
                                                                                      self.rx,
                                                                                      self.weights))
            else:
                self.signal_raw = sh.mult_tsmomsmooth(self.sig_assets,
                                                      self.lkbks,
                                                      self.smooth_period,
                                                      self.smooth_method,
                                                      self.rx,
                                                      self.weights)

            self.signal_final = self.signal_raw.copy()

            # print('Raw signal processed!')

            return self

    class signal_mom_macrossover(signal_init):

        def __init__(self, sig_assets, lkbk_pair, smooth_method):
            self.sig_assets = sig_assets
            self.lkbk_pair = lkbk_pair
            self.smooth_method = smooth_method

            if self.lkbk_pair[1] <= self.lkbk_pair[0]:
                raise ValueError("lkbk_pair[1] should be greater than lkbk_pair[0]")

        def gen_signal(self):
            self.signal_raw = self.sig_assets.apply(lambda x: sh.ma_crossover(x,
                                                                              self.lkbk_pair,
                                                                              self.smooth_method))
            self.signal_final = self.signal_raw.copy()

            # print('Raw signal processed!')

            return self

    class signal_mom_breakout(signal_init):

        def __init__(self, sig_assets, lkbk, smooth_method):
            self.sig_assets = sig_assets
            self.lkbk = lkbk
            self.smooth_method = smooth_method

        def gen_signal(self):
            self.signal_raw = self.sig_assets.apply(lambda x: sh.breakout(x,
                                                                          self.lkbk,
                                                                          self.smooth_method))
            self.signal_final = self.signal_raw.copy()

            return self


    class signal_mom_avgbreakout(signal_init):

        def __init__(self, sig_assets, lkbks, sign):
            self.sig_assets = sig_assets
            self.lkbks = lkbks
            self.sign = sign

        def gen_signal(self):
            self.signal_raw = self.sig_assets.apply(lambda x: sh.avg_breakout(x, self.lkbks, self.sign))
            self.signal_final = self.signal_raw.copy()

            # print('Raw signal processed!')

            return self

    class signal_mom_maavgbreakout(signal_init):

        def __init__(self, sig_assets, lkbks, smooth_method, sign, prcnt):
            self.sig_assets = sig_assets
            self.lkbks = lkbks
            self.smooth_method = smooth_method
            self.sign = sign
            self.prcnt = prcnt

        def gen_signal(self):
            self.signal_raw = self.sig_assets.apply(lambda x: sh.maavg_breakout(x,
                                                                                self.lkbks,
                                                                                self.smooth_method,
                                                                                self.sign,
                                                                                self.prcnt))
            self.signal_final = self.signal_raw.copy()

            # print('Raw signal processed!')

            return self

    class signal_mom_olstrend(signal_init):

        def __init__(self, sig_assets, lkbk):
            self.sig_assets = sig_assets
            self.lkbk = lkbk
            self.dprices = dd.from_pandas(self.sig_assets, npartitions=n_parts)

        def gen_signal(self):
            self.signal_raw = self.signal_mapper(self.dprices, self.lkbk,
                                                 sh.ols_trend, args=(False,))
            # self.sig_assets.rolling(self.lkbk).apply(sh.ols_trend)
            self.signal_final = self.signal_raw.copy()

            # print('Raw signal processed')

            return self

    class signal_mom_rsi(signal_init):

        def __init__(self, sig_assets, lkbk, window, smooth_method):
            self.sig_assets = sig_assets
            self.lkbk = lkbk
            self.window = window
            self.smooth_method = smooth_method

        def gen_signal(self):
            self.signal_raw = self.sig_assets.apply(sh.rsi_wrap,
                                                    args=(self.lkbk,
                                                          self.window,
                                                          self.smooth_method))
            self.signal_final = self.signal_raw.copy()

            return self

    class signal_mom_bollingerbands(signal_init):

        def __init__(self, sig_assets, window, window_dev):
            self.sig_assets = sig_assets
            self.window = window
            self.window_dev = window_dev

        def gen_signal(self):
            self.signal_raw = self.sig_assets.apply(sh.bollinger_bands_wrap,
                                                    args=(self.window, self.window_dev,))
            self.signal_final = self.signal_raw.copy()

            return self

    class signal_mom_macd(signal_init):

            def __init__(self, sig_assets, lkbk_pair):

                self.sig_assets = sig_assets
                self.lkbk_pair = lkbk_pair
                if self.lkbk_pair[1] <= self.lkbk_pair[0]:
                    raise ValueError("lkbk_pair[1] should be greater than lkbk_pair[0]")

            def gen_signal(self):

                self.signal_raw = sh.macd(self.sig_assets,
                                          self.lkbk_pair[1],
                                          self.lkbk_pair[0])
                self.signal_final = self.signal_raw.copy()

                return self

    class signal_mom_sharpe(signal_init):

        def __init__(self, sig_assets, lkbk):
            self.sig_assets = sig_assets
            self.lkbk = lkbk
            self.returns = self.sig_assets.pct_change(1)
            self.dreturns = dd.from_pandas(self.returns, npartitions=n_parts)

        def gen_signal(self):

            self.signal_raw = self.signal_mapper(self.dreturns, self.lkbk,
                                                 sh.sharpe_ratio, args=(1,))
            self.signal_final = self.signal_raw.copy()

            return self

    class signal_mom_omega(signal_init):

        def __init__(self, sig_assets, lkbk):
            self.sig_assets = sig_assets
            self.lkbk = lkbk
            self.returns = self.sig_assets.pct_change(1)
            self.dreturns = dd.from_pandas(self.returns, npartitions=n_parts)

        def gen_signal(self):

            self.signal_raw = self.signal_mapper(self.dreturns, self.lkbk,
                                                 sh.omega_ratio)
            self.signal_final = self.signal_raw.copy()

            return self

    class signal_mom_sortino(signal_init):

        def __init__(self, sig_assets, lkbk):
            self.sig_assets = sig_assets
            self.lkbk = lkbk
            self.returns = self.sig_assets.pct_change(1)
            self.dreturns = dd.from_pandas(self.returns, npartitions=n_parts)

        def gen_signal(self):

            self.signal_raw = self.signal_mapper(self.dreturns, self.lkbk,
                                                 sh.sortino_ratio, args=(1,))
            self.signal_final = self.signal_raw.copy()

            return self

    class signal_mom_calmar(signal_init):

        def __init__(self, sig_assets, lkbk):
            self.sig_assets = sig_assets
            self.lkbk = lkbk
            self.returns = self.sig_assets.pct_change()
            self.dreturns = dd.from_pandas(self.returns, npartitions=n_parts)

        def gen_signal(self):
            self.signal_raw = self.signal_mapper(self.dreturns, self.lkbk,
                                                 sh.calmar_ratio, args=(1,))
            self.signal_final = self.signal_raw.copy()

            return self

    class signal_mom_dvr(signal_init):

        def __init__(self, sig_assets, lkbk):
            self.sig_assets = sig_assets
            self.lkbk = lkbk
            self.dprices = dd.from_pandas(self.sig_assets, npartitions=n_parts)

        def gen_signal(self):

            self.signal_raw = self.signal_mapper(self.dprices, self.lkbk,
                                                 sh.dvr, args=(1,))
            self.signal_final = self.signal_raw.copy()

            return self

    class signal_mom_var(signal_init):

        def __init__(self, sig_assets, lkbk):
            self.sig_assets = sig_assets
            self.lkbk = lkbk
            self.returns = self.sig_assets.pct_change(1)
            self.dreturns = dd.from_pandas(self.returns, npartitions=n_parts)

        def gen_signal(self):
            self.signal_raw = self.signal_mapper(self.dreturns, self.lkbk,
                                                 sh.var_ratio, args=(0.01,))
            self.signal_final = self.signal_raw.copy()

            return self

    class signal_mom_cvar(signal_init):

        def __init__(self, sig_assets, lkbk):
            self.sig_assets = sig_assets
            self.lkbk = lkbk
            self.returns = self.sig_assets.pct_change(1)
            self.dreturns = dd.from_pandas(self.returns, npartitions=n_parts)

        def gen_signal(self):
            self.signal_raw = self.signal_mapper(self.dreturns, self.lkbk,
                                                 sh.cvar_ratio, args=(0.01,))
            self.signal_final = self.signal_raw.copy()

            return self

    class signal_mom_maxloss(signal_init):

        def __init__(self, sig_assets, lkbk):
            self.sig_assets = sig_assets
            self.lkbk = lkbk
            self.returns = self.sig_assets.pct_change(1)
            self.dreturns = dd.from_pandas(self.returns, npartitions=n_parts)

        def gen_signal(self):
            self.signal_raw = self.signal_mapper(self.dreturns, self.lkbk,
                                                 sh.maxloss_ratio)

            self.signal_final = self.signal_raw.copy()

            return self

    class signal_mom_hilodiff(signal_init):

        def __init__(self, sig_assets, lkbk):
            self.sig_assets = sig_assets
            self.lkbk = lkbk
            self.dprices = dd.from_pandas(self.sig_assets, npartitions=n_parts)

        def gen_signal(self):
            self.signal_raw = self.signal_mapper(self.dprices, self.lkbk,
                                                 sh.hilo_diff)
            self.signal_final = self.signal_raw.copy()

            return self

    class signal_mom_ulcer(signal_init):

        def __init__(self, sig_assets, lkbk):
            self.sig_assets = sig_assets
            self.lkbk = lkbk
            self.dprices = dd.from_pandas(self.sig_assets, npartitions=n_parts)

        def gen_signal(self):
            self.signal_raw = self.signal_mapper(self.dprices, self.lkbk,
                                                 sh.ulcer_ratio)
            self.signal_final = self.signal_raw.copy()

            return self

    class signal_mom_effratio(signal_init):

        def __init__(self, sig_assets, lkbk):
            self.sig_assets = sig_assets
            self.lkbk = lkbk

        def gen_signal(self):
            self.signal_raw = self.sig_assets.apply(sh.eff_ratio,
                                                    args=(self.lkbk,))
            self.signal_final = self.signal_raw

            return self

    class signal_mom_kalmanfilter(signal_init):

        def __init__(self, sig_assets, burn_in=21):
            self.sig_assets = sig_assets
            self.burn_in = burn_in

        def gen_signal(self):

            self.signal_raw = self.sig_assets.div(self.sig_assets.\
                                                  apply(sh.kalman_smoother,
                                                        args=(self.burn_in,)).shift(1)) - 1
            self.signal_final = self.signal_raw.copy()

            return self

    class signal_mom_hpfilter(signal_init):

        def __init__(self, sig_assets, lkbk=21):
            self.sig_assets = sig_assets
            self.lkbk = lkbk
            self.dprices = dd.from_pandas(self.sig_assets, npartitions=n_parts)

        def gen_signal(self):

            self.signal_hp = self.dprices.map_overlap(lambda df: df.rolling(self.lkbk). \
                                                       apply(sh.hp_filter, args=(), raw=True),
                                                       self.lkbk, 0). \
                compute(scheduler='processes')

            self.signal_raw = self.sig_assets.div(self.signal_hp.shift(1))-1

            self.signal_final = self.signal_raw.copy()

            return self


    class signal_mom_mancross(signal_init):

        def __init__(self, sig_assets, coms):
            self.sig_assets = sig_assets
            self.coms = coms

        def gen_signal(self):

            self.signal_raw = self.sig_assets.apply(sh.man_macross,
                                                    args=(self.coms,))
            self.signal_final = self.signal_raw.copy()

            return self


    class signal_get_data():

        def __init__(self, tickers, fields, startdate, enddate=None, frequency='DAILY'):

            self.tickers = tickers
            self.fields = fields
            self.startdate = startdate
            self.enddate = enddate
            self.frequency = frequency

        def pull(self, resample_fill=True):

            data = sh.get_bbg_data(self.tickers, self.fields,
                                   self.startdate, self.enddate,
                                   self.frequency)

            if (resample_fill is True) and (self.frequency is 'DAILY'):
                return data.resample('B').last().fillna(method='ffill', limit=5)
            elif (resample_fill is True) and (self.frequency is 'WEEKLY'):
                return data.resample('W').last().fillna(method='ffill', limit=2)
            elif (resample_fill is True) and (self.frequency is 'MONTHLY'):
                return data.resample('M').last().fillna(method='ffill', limit=1)
            elif (resample_fill is True) and (self.frequency is 'QUARTERLY'):
                return data.resample('M').last()
            elif (resample_fill is True) and (self.frequency is 'YEARLY'):
                return data.resample('A').last()
            else:
                return data.fillna(method='ffill')
