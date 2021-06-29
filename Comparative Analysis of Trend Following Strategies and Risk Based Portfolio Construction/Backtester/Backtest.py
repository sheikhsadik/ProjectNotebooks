import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib.gridspec as gridspec

import Backtester.SignalHelper as sh
from Backtester.SignalLibrary import Signals
import Backtester.MiscData as mdata

mpl.style.use('ggplot')
mpl.rcParams['figure.figsize'] = (12, 10)
mpl.rcParams['axes.grid'] = True


class backtest:

    def __init__(self, strategy, test, cash=None, port_weights=None, rebalancing='B'):

        print('Initiating Backtester')

        self.test = test
        self.cash = cash
        self.port_weights = port_weights
        if strategy is None:
            self.no_sigs = True
            self.strategy = Signals.signal_userdefined(pd.DataFrame(np.ones(self.test.shape[0]),
                                                                    index=self.test.index,
                                                                    columns=['RX_Asset']))
        else:
            self.no_sigs = False
            self.strategy = strategy
            self.strategy.signal_final = self.strategy.signal_final.dropna(how='all')

        if type(self.strategy.signal_final) is pd.core.series.Series:
            self.strategy.signal_final = self.strategy.signal_final.to_frame().rename(columns={0: 'RX_Asset'})

        self.n_sigs = self.strategy.signal_final.shape[1]

        self.rb_n = rebalancing
        if self.rb_n is 'B':
            self.t = 261
        elif self.rb_n is 'W':
            self.t = 52
        elif self.rb_n is 'M':
            self.t = 12
        elif self.rb_n is 'Q':
            self.t = 4
        elif self.rb_n is 'A':
            self.t = 1
        else:
            raise ValueError('Provide appropriate rebalancing type')

        self.test_freq = self.test.index.inferred_freq
        if self.test_freq == 'B':
            self.tau = 261
        elif self.test_freq == 'W-SUN':
            self.tau = 52
        elif self.test_freq == 'M':
            self.tau = 12
        elif self.test_freq == 'Q':
            self.tau = 4
        elif self.test_freq == 'A':
            self.tau = 1
        else:
            raise TypeError('Index frequency must be of an appropriate type')

        self.lead_lags = [-65, -22, -5, -1, 0, 1, 5, 22, 65, 131, 261, 522]
        self.corr_data = mdata.corr_data.resample(self.rb_n).last().dropna()
        self.corr_tickers = mdata.corr_tickers

        self.corr_rets = self.corr_data[self.corr_tickers[:-2]].apply(np.log).diff(1). \
            rename(columns={self.corr_tickers[0]: 'S&P500', self.corr_tickers[1]: 'US10Y',
                            self.corr_tickers[2]: 'CAD10Y', self.corr_tickers[3]: 'GOLD',
                            self.corr_tickers[4]: 'BE05Y'})
        self.corr_rets['ACWI'] = self.corr_data['M2WD Index'].apply(np.log).diff(1).sub(self.corr_data['US0003M Index'].
                                                                                        divide(self.t * 100))
        self.corr_rets['6040_US'] = 0.6 * self.corr_rets['S&P500'] + 0.4 * self.corr_rets['US10Y']
        self.corr_rets.dropna(inplace=True)

    def get_excess_returns(self):

        self.test_TR = self.test.pct_change(1).replace([np.inf, -np.inf], np.nan)
        #self.test.apply(np.log).diff(1).replace([np.inf, -np.inf], np.nan)
        if self.test_TR.shape[1] is 1:
            rx_ids = ['RX_Asset']
        else:
            rx_ids = ['RX_{}'.format(test_id) for test_id in self.test_TR.columns]

        if self.n_sigs == self.test_TR.shape[1]:
            self.strategy.signal_final.columns = rx_ids

        if self.cash is None:
            self.test_df = self.test_TR.copy()
            self.test_df.columns = rx_ids
            self.test_ids = rx_ids
        else:
            self.cash = self.cash.shift(1).divide(self.tau * 100, axis=0)
            self.test_df = pd.concat([self.test_TR, self.cash], axis=1)

            if self.cash.shape[1] is 1:
                self.test_df = self.test_df.sub(self.test_df.iloc[:, -1], axis=0).iloc[:, :-1]
                self.test_df.columns = rx_ids

            elif self.cash.shape[1] > 1:
                self.test_df[rx_ids] = pd.DataFrame(self.test_df[self.test_TR.columns].values -
                                                    self.test_df[self.cash.columns].values,
                                                    index=self.test_df.index, columns=rx_ids)
            self.test_ids = rx_ids

        if self.port_weights is not None:
            self.test_ids = ['RX_Asset']

            if (type(self.port_weights) is np.ndarray) and (len(self.port_weights.shape) == 1):
                self.test_df = self.test_df[rx_ids].dot(self.port_weights).to_frame().\
                    rename(columns={0: self.test_ids[0]})

            else:
                self.port_weights.columns = rx_ids
                self.test_df = self.test_df.multiply(self.port_weights).sum(skipna=False, axis=1).to_frame().\
                    rename(columns={0: self.test_ids[0]})

        self.test_df = self.test_df.dropna(how='all')
        self.l_assets = self.test_df.shape[0]
        self.n_assets = self.test_df.shape[1]

    def get_risk_size(self, risk_target, method, risk_lkbk):

        self.risk_target = risk_target
        self.risk_lkbk = risk_lkbk

        signal = self.strategy.signal_final.copy()

        print("Calculating risk for each strategy...")
        if (self.n_sigs > 1) and (self.n_assets == 1):
            strat_rets = signal.multiply(self.test_df.squeeze(), axis='index').dropna(how='all')
        else:
            strat_rets = self.test_df.multiply(signal.squeeze(), axis='index').dropna(how='all')

        print('Risk method: {}'.format(method))
        if method is 'exponential':
            self.risk_df = sh.exponential_risk(strat_rets, self.risk_lkbk, self.tau)
        elif method is 'rolling':
            self.risk_df = sh.rolling_risk(strat_rets, self.risk_lkbk, self.tau)

        print('Calculating risk size with risk target: {:.1%}...'.format(self.risk_target))
        self.risk_size_df = self.risk_target / self.risk_df
        self.risk_size_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    def get_position_size(self, strat_type, decay, unitrisk=False):

        if strat_type == 'Q':
            signal = self.strategy.signal_final.copy()
        elif strat_type == 'C':
            signal = self.strategy.signal_final.copy().apply(lambda x: x/np.sum(np.abs(x)), axis=1)

        if unitrisk is True:
            risksize_df = self.unitrisk_size_df.copy()
        else:
            risksize_df = self.risk_size_df.copy()

        if (signal.shape[1] > 1) and (risksize_df.shape[1] == 1):
            return signal.shift(decay).multiply(risksize_df.squeeze(), axis='index')
        else:
            return risksize_df.multiply(signal.shift(decay).squeeze(), axis='index')

    def get_strategy_returns(self, strat_type, decay, return_misc=True, unitrisk=False):

        position_size_df = self.get_position_size(strat_type, decay, unitrisk=unitrisk)
        position_size_rbn = position_size_df.resample(self.rb_n).last()
        test_rebalanced = self.test_df.apply(lambda x: (1+x)).cumprod().\
            groupby(pd.Grouper(freq=self.rb_n)).last().pct_change(1)
            #self.test_df.groupby(pd.Grouper(freq=self.rb_n)).sum(min_count=1)
        if strat_type == 'C':
            test_rebalanced = test_rebalanced.dropna()
        strat_df = position_size_rbn.shift(1).multiply(test_rebalanced.squeeze(), axis='index')

        if strat_type == 'C':
            strat_df = strat_df.sum(axis=1, skipna=False).to_frame()
            strat_df.columns = ['RX_Asset']

        if return_misc is True:
            return position_size_df, position_size_rbn, test_rebalanced, strat_df
        else:
            return strat_df

    def get_agg_stats(self, strat_df):

        strat_ix = self.strat_df.index
        test_df = self.test_rebalanced[strat_ix[0]:]

        if self.strat_type == 'Q':
            sr_test = test_df.apply(sh.sharpe_ratio, args=(self.t,)).\
                to_frame().rename(columns={0: 'Test Asset Sharpe'}).T
        elif self.strat_type == 'C':
            sr_test = test_df.mean(axis=1).to_frame().apply(sh.sharpe_ratio, args=(self.t,)).\
                to_frame().rename(columns={0: 'Test Asset Sharpe'}, index={0: 'RX_Asset'}).T

        if self.no_sigs is False:
            sr_raw = self.SR_decay_df[0].to_frame().rename(columns={0: 'Signal Sharpe'}).T
            temp_stats = pd.concat([strat_df.apply(lambda x: sh.get_stats(x, self.t)), sr_raw, sr_test,
                                    strat_df.apply(lambda x: sh.get_drawdown_stats(x, self.t))], axis=0).dropna()
        else:
            temp_stats = pd.concat([strat_df.apply(lambda x: sh.get_stats(x, self.t)), sr_test,
                                    strat_df.apply(lambda x: sh.get_drawdown_stats(x, self.t))], axis=0).dropna()

        temp_corrs = sh.get_corrs(strat_df, self.corr_rets)

        return pd.concat([temp_stats, temp_corrs], axis=0, keys=['Performance Stats', 'Correlations'])

    def get_subperiod_stats(self, start, end):

        if self.strat_df is None:
            print("Run 'get_performance' first!!")
        else:
            strat_df = self.strat_df[start:end].copy()

            return self.get_agg_stats(strat_df)

    def get_performance(self, strat_type='Q', risk_target=0.10, method_risksize='rolling', risk_lkbk=32, decay=0):

        self.method_risksize = method_risksize
        self.risk_target = risk_target
        self.risk_lkbk = risk_lkbk
        self.strat_type = strat_type

        print('Calculating excess returns...')
        self.get_excess_returns()

        print('Backtesting: # of Signals: {0}, # of Assets {1}'.format(self.n_sigs, self.n_assets))
        print('Initiating risk sizing...')
        if (self.n_assets == 1) and (self.n_sigs > 1):

            unit_cols = self.strategy.signal_final.columns
            unit_dim = (self.l_assets, self.n_sigs)
            self.unitrisk_size_df = pd.DataFrame(np.ones(unit_dim),
                                                 index=self.test_df.index,
                                                 columns=unit_cols)
        else:
            unit_cols = self.test_ids
            self.unitrisk_size_df = pd.DataFrame(np.ones(self.test_df.shape),
                                                 index=self.test_df.index,
                                                 columns=unit_cols)

        if risk_target is None:
            self.risk_size_df = self.unitrisk_size_df.copy()
        else:
            self.get_risk_size(self.risk_target, self.method_risksize, self.risk_lkbk)

        print('Calculating position size, rebalanced position and strategy performance')
        self.position_size_df, self.position_size_rbn, self.test_rebalanced, self.strat_df = \
            self.get_strategy_returns(self.strat_type, decay, return_misc=True)

        print('Calculating performance statistics...')
        if self.no_sigs is False:
            self.SR_decay_df = pd.concat({

                ll: self.get_strategy_returns(self.strat_type, ll, return_misc=False, unitrisk=True).
                    apply(lambda x: x.mean() * np.sqrt(self.t) / x.std()) for ll in self.lead_lags

            }, axis=1)
        self.strat_stats_agg = self.get_agg_stats(self.strat_df)

        print('Done!')

    def get_performance_tearsheet(self, asset_id='RX_Asset'):

        if (self.strat_df.shape[1] > 1) & (asset_id == 'RX_Asset'):
            print('Please enter asset ID!')

        else:
            print('Generating tearsheet...')

            tempstrat_df = pd.DataFrame(index=self.strat_df.index)
            tempstrat_df['cum_logret'] = sh.cum_returns(self.strat_df[[asset_id]])
            tempstrat_df.dropna(inplace=True)
            tempstrat_df['logret'] = self.strat_df[[asset_id]]

            tempstrat_mu = tempstrat_df['logret'].mean()
            tempstrat_vol = tempstrat_df['logret'].std()

            tempstrat_df['time_ix'] = np.arange(1, len(tempstrat_df) + 1)
            tempstrat_df['trend'] = (1 + tempstrat_mu * tempstrat_df['time_ix'])
            tempstrat_df['rp1sd'] = tempstrat_df['trend'] + tempstrat_vol * np.sqrt(tempstrat_df['time_ix'])
            tempstrat_df['rp2sd'] = tempstrat_df['trend'] + 2 * tempstrat_vol * np.sqrt(tempstrat_df['time_ix'])
            tempstrat_df['rm1sd'] = tempstrat_df['trend'] - tempstrat_vol * np.sqrt(tempstrat_df['time_ix'])
            tempstrat_df['rm2sd'] = tempstrat_df['trend'] - 2 * tempstrat_vol * np.sqrt(tempstrat_df['time_ix'])
            tempstrat_df['uw'] = tempstrat_df[['cum_logret']].apply(sh.drawdown)

            rolling = tempstrat_df['logret'].rolling(self.t)
            tempstrat_df['rolling_sharpe'] = np.sqrt(self.t) * (rolling.mean() / rolling.std())
            if self.strat_type == 'C':
                position_size = self.position_size_rbn.shift(1).dropna().copy()
            else:
                tempstrat_df['position_size'] = self.position_size_rbn[[asset_id]].shift(1).dropna()

            plt.figure(figsize=(15, 22))
            gs = gridspec.GridSpec(7, 4, wspace=0.25, hspace=0.5)

            ax_equity = plt.subplot(gs[:2, :])
            ax_position = plt.subplot(gs[2, :])
            ax_sharpe = plt.subplot(gs[3, :])
            ax_drawdown = plt.subplot(gs[4, :])
            ax_monthly_returns = plt.subplot(gs[5:, :2])

            colors = ['royalblue', 'lime', 'red', 'lime', 'red', 'grey']
            lstyle = ['-', '-', '-', '-', '-', '--']
            cum_elements = ['cum_logret', 'rp1sd', 'rp2sd', 'rm1sd', 'rm2sd', 'trend']
            tempstrat_df[cum_elements].plot(color=colors, lw=2, ax=ax_equity, style=lstyle, fontsize=12, legend=False)
            ax_equity.set_title('Cumulative Performance: {}'.format(asset_id))

            if self.strat_type == 'C':
                position_size.plot(color=colors, lw=2, ax=ax_position, linestyle='-.', fontsize=12)
            else:
                tempstrat_df['position_size'].plot(color=colors, lw=2, ax=ax_position, linestyle='-.', fontsize=12)
            ax_position.axhline(0, color='black', lw=2)
            ax_position.set_title('Position Size')

            tempstrat_df['rolling_sharpe'].plot(color='darkgreen', lw=2, ax=ax_sharpe, fontsize=12)
            ax_sharpe.axhline(tempstrat_df['logret'].mean() * np.sqrt(self.t) / tempstrat_df['logret'].std(),
                              color='darkorange', linestyle='--', lw=2)
            ax_sharpe.set_title('Rolling Sharpe Ratio: {:.0f} year'.format(1))

            tempstrat_df['uw'].plot(color='red', kind='area', lw=2, ax=ax_drawdown, alpha=0.3, fontsize=12)
            ax_drawdown.set_title('Drawdown')

            (self.strat_df[[asset_id]] * 100).plot(kind='hist', density=True, color='#0504aa', alpha=0.7,
                                                   edgecolor='black', bins=20, ax=ax_monthly_returns)
            ax_monthly_returns.set_title('Strategy Returns, Rebalancing: {}'.format(self.rb_n))
            ax_monthly_returns.set_xlabel('%')

            if self.no_sigs is False:
                ax_decay = plt.subplot(gs[5:, 2:])
                self.SR_decay_df.loc[[asset_id]].T.plot(kind='bar', ax=ax_decay, color='darkkhaki')
                ax_decay.set_title('SR Decay')
                ax_decay.set_xlabel('Sharpe')
                ax_decay.set_ylabel('# of Days')
