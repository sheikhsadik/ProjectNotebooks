import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import Backtester.SignalHelper as sh


class generate_miscprice():

    def __init__(self):
        print('Pulling miscellaneous price data, this may take a while...')
        self.corr_tickers = ['SP1 A:03_0_R Index', 'TY1 A:03_0_R Comdty', 'CN1 A:03_0_R Comdty',
                             'GC1 A:03_0_R Comdty', 'BSWIZU05 Index', 'M2WD Index', 'US0003M Index']
        self.ix_0 = '19700101'

    def gen_miscdata(self):

        self.corr_data = sh.get_bbg_data(self.corr_tickers, ['PX_LAST'], self.ix_0)
        return self.corr_tickers, self.corr_data


corr_tickers, corr_data = generate_miscprice().gen_miscdata()
print('Done!')

