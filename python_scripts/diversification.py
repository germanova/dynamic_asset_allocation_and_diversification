import numpy as np
import pandas as pd
from datetime import timedelta
import time
from python_scripts.data_and_descriptives import have_na
from python_scripts.plot_backtest import _plot_backtest
from pandas.testing import assert_frame_equal
from python_scripts.markowitz import markowitz_opt

HRP = True
try:
    from python_scripts.hierachical_risk_parity import hrp
except ImportError:
    HRP = False
    print("Hierarchical Risk Parity not found. Skipping...")


class TriggerSimulation:
    def __init__(self, data,  exit_type='cap',
                 rebal_type='markowitz',
                 safe_asset='cash_bank', threshold=0.2,
                 window=180, rebal=30, start_value=10000, t_c=1,
                 t_c_type='fixed',
                 are_returns=True, plot=True, **kwargs):
        '''
        Parameters:
        data: pd.DataFrame with the assets returns or prices, if using prices, returns parameters must be False
        exit_type: type of strategy, could be 'cap' for a cap in returns (cap profit per asset),
            'floor' for a floor in returns (limits losses per asset),
            'cap_floor' for both a floor and a cap in returns,
            'trailing_floor' for a trailing stop
            'no_cap_floor' for no exit
        rebal_type: rebalancing method, currently could be 'ew' for equal-weighted or 'markowitz',
            in case of 'markowitz' extra parameters are: gamma (risk-aversion parameter) and w_bounds (weight bounds, tuple of floats)
            Hierarchical Risk Parity can also be used with 'hrp' if you have an implementation that retrieves the weights for given covariance and correlation matrices
        threshold: float when using cap and floor, a list[float, float] when using cap_floor.
            It is the target % return that will define the rebalancing for the cap and the floor
        window: parameter that uses a rolling window for estimations, only used in for certain strategies
        rebal: step for rebalancing, every each rebal the strategy will do rebalancing
        start_value: start value of the strategy
        t_c: transaction cost in $ if 'fixed', in % if 'pct' (i.e. 0.1% = 0.001)
        t_c_type: either 'fixed' for a fixed monetary amount per transaction or
        'pct' for a cost representing a pct of the transaction value
        are_returns: if True, data are returns, if False must be prices
        plot: if True, perfomance metrics will be shown
        '''

        if not HRP and rebal_type == 'hrp':
            raise ImportError(
                "A Hierarchical Risk Parity library is required to use this feature."
                "The hrp() function should receive both covariance and correlation matrices. Please install it.")

        self.data = data.copy(deep=True)
        self.exit_type = exit_type
        self.rebal_type = rebal_type
        self.safe_asset = safe_asset
        self.threshold = threshold
        self.window = window
        self.rebal = rebal
        self.start_value = start_value
        self.t_c = t_c
        self.t_c_type = t_c_type
        self.are_returns = are_returns
        self.plot = plot

        if self.rebal_type == 'markowitz':
            self.gamma = kwargs.get('gamma', 0.8)
            print(f'using gamma = {self.gamma}')
            self.w_bounds = kwargs.get('w_bounds', (0.0, 1.0))
            print(f'using weight bounds = {self.w_bounds}')

    def update_data(self, new_data, check_tickers=True):
        '''
        Update returns data while making sure that dates are consistent,
        new data must start on the same date and must finish in date greater than the current one
        new data until the final date of the past data must be the same
        '''

        start_date = self.data.index[0]
        end_date = self.data.index[-1]

        assert start_date == new_data.index[0]
        assert end_date < new_data.index[-1]

        if self.safe_asset == 'cash_bank':
            new_data['cash_bank'] = 0.0
        else:
            assert self.safe_asset in new_data.columns
            # put safe asset at the end
            new_data[self.safe_asset] = new_data.pop(self.safe_asset)

        new_r_sub = new_data.loc[new_data.index <= end_date,]
        if check_tickers:
            assert_frame_equal(self.data, new_r_sub)
        else:
            assert_frame_equal(self.data, new_r_sub, check_names=False)

        assert not have_na(new_data)

        self.data = new_data.copy(deep=True)
        print('data updated, running backtest...')

        backtest_results = self.trigger_simulation()

        return backtest_results

    def _trailing_floor(self, da, da_max):
        '''
        Check if trailing floor is broken per asset
        '''

        return da/da_max - 1 < -self.threshold

    def _cap(self, da, da_rebal):
        '''
        Check if cap is exceeded per asset
        '''
        return da/da_rebal - 1 > self.threshold

    def _floor(self, da, da_rebal):
        '''
        Check if floor is broken per asset
        '''
        return da/da_rebal - 1 < -self.threshold

    def _cap_floor(self, da, da_rebal):
        '''
        Check if cap is exceeded per asset or floor is broken per asset
        '''
        return (da/da_rebal - 1 > self.threshold[0]) | (da/da_rebal - 1 < -self.threshold[1])

    def _no_cap_floor(self, da, **kwargs):
        """
        For doing just the diversification part (no cap or floor)
        """
        return da.copy(), 0

    def _ew_rebal_w(self, **kwargs):
        '''
        Computes Equal Weighted weights leaving 0% for cash
        Cash must be the last asset
        '''

        active_assets = len(self.data.columns) - 1
        # assign 0 to safe asset
        w = pd.Series([1/active_assets] *
                      active_assets + [0.0], index=self.data.columns)

        return w

    def _hrp_rebal_w(self, t, **kwargs):
        '''
        Computes the Hierarchical Risk Parity weights leaving 0% for cash
        Cash must be the last asset

        t: current date of estimation
        '''
        # Rolling Data
        sub_returns = self.data.loc[t-timedelta(days=self.window):t,
                                    ~self.data.columns.isin([self.safe_asset])]

        # rolling covariance mat
        cov = sub_returns.cov()
        # rolling correlation mat
        corr = sub_returns.corr()
        w = hrp(corr, cov)

        safe_asset_w = pd.Series([0], index=[self.safe_asset])
        w = pd.concat([w, safe_asset_w])

        return w

    def _markowitz_rebal_w(self, t, **kwargs):
        '''
        Computes Markowitz weights leaving 0% for cash
        Cash must be the last asset
        '''
        # Rolling Data
        sub_returns = self.data.loc[t-timedelta(days=self.window):t,
                                    ~self.data.columns.isin([self.safe_asset])]

        mu = np.array(sub_returns.mean())
        sig = sub_returns.cov()
        w_init = np.ones(sub_returns.shape[1]) / sub_returns.shape[1]

        w = markowitz_opt(mu, sig, w_init, gamma=self.gamma,
                          w_bounds=(self.w_bounds, )*sub_returns.shape[1])

        w = pd.Series(np.append(w, 0.0), index=self.data.columns)

        return w

    def _ew_trigger_w(self, da, da_rebal, da_max, **kwargs):
        """
        Check if there are any assets that activates the trigger and updates dollar allocation
        Also updates rebal value, useful for recording with type of rebalancing was performed
        """

        abs_strategies = {'cap': self._cap, 'floor': self._floor,
                          'cap_floor': self._cap_floor, 'trailing_floor': self._trailing_floor}

        trigger_type = abs_strategies[self.exit_type]
        # make assignation to cash if asset had more return than threshold since last rebal date
        # in the case of cash == 0, there will be an np.nan, however it will be a false boolean

        # [:-1] to exclude cash allocation
        da_no_cash = da[:-1]
        if self.exit_type == 'trailing_floor':
            ret_rebal = np.where(
                da_no_cash == 0, False, trigger_type(da_no_cash, da_max[:-1]))
        else:
            ret_rebal = np.where(
                da_no_cash == 0, False, trigger_type(da_no_cash, da_rebal[:-1]))

        rebal = np.zeros_like(ret_rebal, dtype=int)
        rebal[ret_rebal] = 2
        if ret_rebal.sum() > 0:
            rebal = np.append(rebal, 2)  # for cash
        else:
            rebal = np.append(rebal, 0)  # for cash

        ret_rebal = da_no_cash * ret_rebal
        ret_rebal = np.append(ret_rebal, -ret_rebal.sum())
        da_new = (da - ret_rebal).copy()

        return da_new, rebal

    def _fixed_t_c_rebal(self, w_t_1, w, value):
        """
        Return fixed transaction costs
        """

        # add transactions costs
        if self.safe_asset == 'cash_bank':
            # do not incur in transaction costs for cash
            w_t_1.iloc[-1] = w.iloc[-1]

        if self.t_c_type == 'fixed':
            t_c = np.where(w != w_t_1, self.t_c, 0)

        elif self.t_c_type == 'pct':
            t_c = np.where(w != w_t_1, value * (w-w_t_1).abs() * self.t_c, 0)

        return t_c

    def _fixed_t_c_exit(self, rebal, da_t_1, da):
        """
        Return fixed transaction costs
        """
        # add transactions costs
        if self.t_c_type == 'fixed':
            t_c = self.t_c * (rebal == 2)

        elif self.t_c_type == 'pct':
            t_c = da_t_1 * (rebal == 2) * self.t_c

        if self.safe_asset == 'cash_bank':
            t_c[-1] = 0

        elif self.t_c_type == 'pct':  # cost for safe asset buy
            t_c[-1] = (da[-1] - da_t_1[-1]) * (rebal[-1] == 2) * self.t_c

        return t_c

    def _trigger_backtest(self) -> None:
        '''
        Executes the backtesting for a strategy based on trigger (estimates weights based a trigger or rebalancing dates)

        '''
        value = self.start_value
        # init weights on t-1
        w_t_1 = pd.Series(
            np.zeros(self.data.shape[1]), index=self.data.columns)
        for t in self.weights.index:

            # rebal at the start of the day
            if t in self.rebaldates:
                w = self.rebal_strategy(t=t)  # includes 0% to safe asset

                t_c = self._fixed_t_c_rebal(w_t_1, w, value)
                # end of the day dollar allocation
                # dollar allocation in t (dollar_allocation_t *(1+r_t))
                da = ((value - t_c.sum()) * w)@np.diag(self.data.loc[t, :]+1)
                w = da/da.sum()  # end of the day vector of weights

                # store dollar allocation at then time of rebalancing
                da_rebal = da.copy()
                da_max = da.copy()
                # store value of rebalancing as 1 (rebalancing on rebalancing date)
                rebal = np.where(t_c > 0, 1, 0)

            else:
                # end of the day dollar allocation in t (dollar_allocation_t *(1+r_t))
                da_t_1 = da.copy()
                da_t_1 = da_t_1 @ np.diag(self.data.loc[t, :]+1)

                da_max = np.maximum(da_max, da)  # for the trailing floor

                da, rebal = self.trigger_strategy(
                    da=da_t_1, da_rebal=da_rebal, da_max=da_max)

                # add transactions costs
                t_c = self._fixed_t_c_exit(rebal, da_t_1, da)
                da[-1] -= t_c.sum()  # subtract costs to alloc on safe asset

                # weights drifts
                w = da/da.sum()  # end of the day vector of weights

                w = pd.Series(w, index=self.data.columns)

            w_t_1 = w.copy()  # update weights at t-1
            value = da.sum()
            self.dollar_allocation.loc[t] = da
            self.weights.loc[t] = w
            self.rebalancing.loc[t] = rebal
            self.transaction_costs.loc[t] = t_c

    def _check_tc_type(self):
        '''
        Check if transaction costs parameters are valid

        '''

        allowed_t_c = ['fixed', 'pct']

        if self.t_c < 0:
            print(f'transaction cost must be >= 0')
            return False

        if self.t_c_type not in allowed_t_c:
            print(f'transaction cost type must not in {allowed_t_c}')
            return False

        if self.t_c_type == 'pct' and self.t_c >= 1:
            print(f'transaction cost is >= 1')
            return False

        return True

    def _check_capped_exit_type(self) -> bool:
        '''
        Check if allocation type and threshold data is coherent
        '''

        if self.exit_type in ['cap', 'floor', 'trailing_floor']:
            if isinstance(self.threshold, float):
                print(
                    f'backtesting {self.exit_type} strategy using a threshold (% return) of {self.threshold}')
            else:
                print(f'threshold is not of {float}')
                return False

        elif self.exit_type in ['cap_floor']:
            if isinstance(self.threshold[0], float) and isinstance(self.threshold[1], float) and len(self.threshold) == 2:
                print(
                    f'backtesting {self.exit_type} strategy using thresholds (% return) of {self.threshold}')
            else:
                print(
                    'threshold does not have the correct format, check if its a list with 2 floats')
                return False
        elif self.exit_type in ['no_cap_floor']:
            return True
        else:
            print(f'{self.exit_type} strategy not found')
            return False

        return True

    def _check_rebal_type(self) -> bool:
        allowed_methods = ['ew', 'markowitz', 'hrp']
        if self.rebal_type not in allowed_methods:
            print(
                f'rebalancing method not found {float}, try {allowed_methods}')
            return False

        return True

    def _check_cash_asset(self) -> bool:
        '''
        Check if cash asset data is coherent
        '''

        if self.safe_asset == 'cash_bank':
            self.data['cash_bank'] = 0.0
        elif self.safe_asset in self.data.columns:
            # put safe asset at the end
            self.data[self.safe_asset] = self.data.pop(self.safe_asset)
        else:
            print('safe asset not found in data')
            return False

        print(f'using {self.safe_asset} as safe asset...')
        return True

    def trigger_simulation(self):
        '''
        Executes the simulation of a trigger strategy (estimates weights based on a trigger or rebalancing date)
        '''

        start_time = time.time()

        if not self._check_tc_type():
            return None

        if have_na(self.data) or not self._check_capped_exit_type() or not self._check_rebal_type():
            return None

        cash_check = self._check_cash_asset()
        if not cash_check:
            return None

        if not self.are_returns:
            self.data = self.data.pct_change().dropna()

        if self.exit_type in ['cap', 'floor', 'cap_floor', 'trailing_floor']:
            self.trigger_strategy = self._ew_trigger_w
        elif self.exit_type == 'no_cap_floor':
            self.trigger_strategy = self._no_cap_floor

        if self.rebal_type == 'ew':
            self.rebal_strategy = self._ew_rebal_w
        elif self.rebal_type == 'hrp':
            self.rebal_strategy = self._hrp_rebal_w
        elif self.rebal_type == 'markowitz':
            self.rebal_strategy = self._markowitz_rebal_w

        rebaldates = [i for i in range(
            self.window, len(self.data.index), self.rebal)]
        self.rebaldates = self.data.iloc[rebaldates].index
        self.start_date = self.rebaldates[0]

        self.weights = pd.DataFrame().reindex_like(
            pd.DataFrame(self.data.loc[self.start_date:]))
        self.dollar_allocation = self.weights.copy(deep=True)
        self.rebalancing = self.weights.copy(deep=True)
        self.transaction_costs = self.weights.copy(deep=True)

        self._trigger_backtest()

        first_date = pd.DataFrame(
            self.start_value, index=[self.start_date - timedelta(days=1)], columns=[self.exit_type])
        portfolio_value = pd.DataFrame(self.dollar_allocation.sum(
            axis=1), index=self.weights.index, columns=[self.exit_type])
        self.portfolio_value = pd.concat([first_date, portfolio_value])
        self.portfolio_value.index.name = 'Date'

        backtest_results = {
            "portfolio_value": self.portfolio_value,
            "dollar_allocation": self.dollar_allocation,
            "weights": self.weights,
            "rebalancing": self.rebalancing,
            "transaction_costs": self.transaction_costs
        }

        if self.plot:
            _plot_backtest(self.data, self.exit_type,  self.start_value, self.start_date,
                           self.portfolio_value, self.dollar_allocation, self.weights)

        print("--- %s seconds ---" % (time.time() - start_time))
        return backtest_results
