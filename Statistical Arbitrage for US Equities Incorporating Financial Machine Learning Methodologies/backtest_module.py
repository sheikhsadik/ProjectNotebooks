import numpy as np
import pandas as pd
from StatArb.pair_select_module import PairSelect
from StatArb.estimators_module import Estimators
import Backtester.SignalHelper as sh

class Backtest:

    def __init__(self, df,
                 train_lkbk=int(252*5), test_lkbk=int(252*2),
                 min_periods=int(252*3), opt_k=5,
                 hr_lkbk=int(252), spread_lkbk=int(21*3),
                 corr_thresh=0.10, hr_method='tls', cv_splits=4):

        self.df = df
        self.train_lkbk = train_lkbk
        self.test_lkbk = test_lkbk
        self.min_periods = min_periods
        self.opt_k = opt_k
        self.hr_lkbk = hr_lkbk
        self.spread_lkbk = spread_lkbk
        self.cv_splits = cv_splits
        self.corr_thresh = corr_thresh
        self.hr_method = hr_method

        self.pair_select = None
        self.pairs_opt = None
        self.strat_dict = {}

    def get_pairs(self):

        self.pair_select = PairSelect(self.df, self.train_lkbk,
                                      self.test_lkbk, self.min_periods,
                                      self.opt_k, self.hr_lkbk,
                                      self.spread_lkbk)
        self.pair_select.form_pairs(self.corr_thresh, self.hr_method)
        self.pairs_opt = self.pair_select.selected_pairs

        return self

    def get_pair_strat(self, meta_labelling=True):

        for pair_ids in self.pairs_opt:

            pair_str = '{}_{}'.format(pair_ids[0], pair_ids[1])
            pair_df = self.df[pair_ids].apply(lambda x: 1 + x).cumprod().\
                          iloc[-self.test_lkbk:].apply(np.log)

            z_spreads = self.pair_select.z_spreads_dict[pair_str]
            h_ratio = self.pair_select.hedge_weights_dict[pair_str].\
                          iloc[-self.test_lkbk:]
            if meta_labelling:
                pair_spreads = self.pair_select.spreads_dict[pair_str].to_frame('spread')
                meta_features = self.get_meta_features(pair_spreads, z_spreads)
                meta_labels = self.get_meta_labels(meta_features,
                                                   self.test_lkbk,
                                                   n_splits=self.cv_splits)
                meta_labels = meta_labels.to_frame('meta_label')
            else:
                meta_labels = pd.DataFrame(1, index=h_ratio.index,
                                           columns=['meta_label'])

            self.strat_dict[pair_str] = self.get_strat_ret(pair_df,
                                                           z_spreads.iloc[-self.test_lkbk:],
                                                           meta_labels,
                                                           h_ratio)
        return self

    def score_to_status(self, z_spreads, dev=1.5, stop_long=0.5, stop_short=0.75):

        sig_df = z_spreads.to_frame('z_spreads')
        sig_df['signal'] = 0
        sig_df['signal'] = np.where(sig_df['z_spreads'] < -dev, 1,
                                    sig_df['signal'])
        sig_df['signal'] = np.where(sig_df['z_spreads'] > dev, -1,
                                    sig_df['signal'])
        sig_df['stop_signal'] = 0
        sig_df['status'] = 0
        stop_idx = (sig_df['z_spreads'] > - stop_long) & (sig_df['z_spreads'] < stop_short)
        sig_df['stop_signal'] = np.where(stop_idx, 1, sig_df['stop_signal'])

        signal = sig_df['signal'].values
        stop_signal = sig_df['stop_signal'].values
        sig_df['status'] = sh.get_trade_status(signal, stop_signal)

        return sig_df['status']

    def get_meta_features(self, pair_spreads, z_spreads):

        cols = []
        params_dict = {

            'vol_lkbks': [21, 63],
            'skew_lkbks': [21, 63],
            'kurt_lkbks': [21, 63],
            'hurst_lkbks': [21, 63],
            'acf_lags': [1, 5, 10]
        }

        vol_df = pd.concat(
            {'vol_{}'.format(lkbk): pair_spreads.diff().\
                rolling(lkbk).std() for lkbk in params_dict['vol_lkbks']
            },
            axis=1)

        skew_df = pd.concat(
            {'skew_{}'.format(lkbk): pair_spreads.diff().\
                rolling(lkbk).skew() for lkbk in params_dict['skew_lkbks']
             },
            axis=1)

        kurt_df = pd.concat(
            {'kurt_{}'.format(lkbk): pair_spreads.diff().\
                rolling(lkbk).kurt() for lkbk in params_dict['kurt_lkbks']
             },
            axis=1)

        hurst_df = pd.concat(
            {'hurst_{}'.format(lkbk): pair_spreads.\
                rolling(lkbk).apply(sh.get_hurst_exponent, raw=True)
             for lkbk in params_dict['hurst_lkbks']
             },
            axis=1)

        acf_df = pd.concat(
            {'acf_{}'.format(lag): pair_spreads.diff().\
                rolling(21).apply(lambda x: x.autocorr(lag=lag), raw=False)
             for lag in params_dict['acf_lags']
             },
            axis=1)

        meta_df = pd.concat([vol_df, skew_df,
                             kurt_df, hurst_df,
                             acf_df], axis=1).shift(1)

        for l1, l2 in meta_df.columns: cols.append('{}_{}'.format(l1, l2))
        meta_df = meta_df.droplevel(axis=1, level=0)
        meta_df.columns = cols

        meta_df['spread_ret'] = pair_spreads.diff().apply(lambda x: np.exp(x) - 1)
        meta_df['spread_sign'] = meta_df['spread_ret'].apply(np.sign)
        meta_df['z_spreads'] = z_spreads.shift(1)
        meta_df['status'] = self.score_to_status(meta_df['z_spreads'])
        meta_df['label'] = np.where(meta_df['spread_sign'] == meta_df['status'], 1, 0)

        meta_df.drop(columns=['spread_ret', 'spread_sign'],
                     inplace=True,
                     axis=1)
        meta_df.dropna(axis=0, inplace=True)

        return meta_df

    def get_meta_labels(self, meta_df, test_lkbk, n_splits=4):

        train_meta = meta_df.iloc[:-test_lkbk]
        test_meta = meta_df.iloc[-test_lkbk:]

        y_train_0 = train_meta.loc[:, 'label'].values
        ix = y_train_0.cumsum() > 0
        X_train = train_meta.loc[:, :'status'].values[ix]
        y_train = y_train_0[ix]

        X_test = test_meta.loc[:, :'status'].values
        y_test = test_meta.loc[:, 'label'].values

        ml_models = Estimators(n_splits=n_splits)
        ml_models.fit(X_train, y_train)
        test_pred = ml_models.predict(X_test, y_test)
        ensemble_pred = pd.DataFrame(test_pred).mean(axis=1)
        ensemble_pred.index = test_meta.index
        ensemble_pred.index.name = 'date'

        return ensemble_pred

    def spread_return_bg1(self, log_prices, h_ratio, position, tickers):
        return position.multiply(((1 / h_ratio).multiply(log_prices[tickers[0]], axis=0) -\
                                  log_prices[tickers[1]]).diff(), axis=0)

    def spread_return_bl1(self, log_prices, h_ratio, position, tickers):
        return position.multiply((log_prices[tickers[0]] - h_ratio. \
                                  multiply(log_prices[tickers[1]], axis=0)).diff(), axis=0)

    def get_strat_ret(self, pair_df, z_spreads,
                      meta_labels, h_ratio, dev=1.5,
                      stop_long=0.50, stop_short=0.75):

        test_df = pd.DataFrame()
        pair_ids = pair_df.columns
        test_df[pair_ids] = pair_df
        test_df['z_spreads'] = z_spreads
        test_df['h_ratio'] = h_ratio

        test_df['status'] = self.score_to_status(test_df['z_spreads'],
                                                 dev, stop_long, stop_short)
        test_df['status'] = test_df['status'].shift(1)
        meta_labels['meta_label'] = np.where(meta_labels['meta_label'] > 0.50, 1, 0)
        test_df['status'] = test_df['status'].\
                multiply(meta_labels['meta_label'].squeeze(), axis=0)

        test_df['dn_bool'] = test_df['h_ratio'] > 1.0
        test_df['strat_rets'] = 0.0
        test_df.loc[test_df['dn_bool'], 'strat_rets'] = \
            self.spread_return_bg1(test_df[pair_ids],
                                   test_df['h_ratio'],
                                   test_df['status'],
                                   pair_ids)

        test_df.loc[~test_df['dn_bool'], 'strat_rets'] = \
            self.spread_return_bl1(test_df[pair_ids],
                                   test_df['h_ratio'],
                                   test_df['status'],
                                   pair_ids)

        return test_df['strat_rets'].apply(lambda x: np.exp(x) - 1)
