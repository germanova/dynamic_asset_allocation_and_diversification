import numpy as np
import pandas as pd
from datetime import timedelta
import time
from python_scripts.data_and_descriptives import have_na
from python_scripts.plot_backtest import _plot_backtest


class TriggerSimulation:
    def __init__(self, data,  allocation_type='ew_cap',
                 safe_asset='cash_bank', threshold=0.2,
                 window=180, rebal=30, start_value=1,
                 returns=True, plot=True):
        '''
        Parameters:
        data: pd.DataFrame with the assets returns or prices, if using prices, returns parameters must be False
        allocation_type: type of strategy, could be 'ew_cap' for an Equal Weighted with a cap in returns (cap profit per asset),
            'ew_floor' for an Equal Weighted with a floor in returns (limits losses per asset) or
            'ew_cap_floor' for and Equal Weighted with both a floor and a cap in returns
        threshold: float when using ew_cap and ew_floor, a list[float, float] when using ew_cap_floor.
            It is the target % return that will define the rebalancing for the cap and the floor
        window: parameter that uses a rolling window for estimations, only used in for certain strategies 
        rebal: step for rebalancing, every each rebal the strategy will do rebalancing
        start_value: start value of the strategy
        returns: if True, data are returns, if False must be prices
        plot: if True, perfomance metrics will be shown
        '''

        self.data = data
        self.allocation_type = allocation_type
        self.safe_asset = safe_asset
        self.threshold = threshold
        self.window = window
        self.rebal = rebal
        self.start_value = start_value
        self.returns = returns
        self.plot = plot

    def _ew_cap(self, da, da_rebal):
        '''
        Check if cap is exceeded per asset
        '''
        return da/da_rebal - 1 > self.threshold

    def _ew_floor(self, da, da_rebal):
        '''
        Check if floor is broken per asset
        '''
        return da/da_rebal - 1 < -self.threshold

    def _ew_cap_floor(self, da, da_rebal):
        '''
        Check if cap is exceeded per asset or floor is broken per asset
        '''
        return (da/da_rebal - 1 > self.threshold[0]) | (da/da_rebal - 1 < -self.threshold[1])

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

    def _ew_trigger_w(self, da, da_rebal, **kwargs):
        """
        Check if there are any assets that activates the trigger and updates dollar allocation
        Also updates rebal value, useful for recording with type of rebalancing was performed
        """

        abs_strategies = {'ew_cap': self._ew_cap, 'ew_floor': self._ew_floor,
                          'ew_cap_floor': self._ew_cap_floor}

        trigger_type = abs_strategies[self.allocation_type]
        # make assignation to cash if asset had more return than threshold since last rebal date
        # in the case of cash == 0, there will be an np.nan, however it will be a false boolean

        # [:-1] to exclude cash allocation
        da_no_cash = da[:-1]
        ret_rebal = trigger_type(da_no_cash, da_rebal[:-1])
        if ret_rebal.sum() > 0:
            rebal = 2
        else:
            rebal = 0
        ret_rebal = da_no_cash * ret_rebal
        ret_rebal = np.append(ret_rebal, -ret_rebal.sum())
        da -= ret_rebal

        return da, rebal

    def _trigger_backtest(self) -> None:
        '''
        Executes the backtesting for a strategy based on trigger (estimates weights based a trigger or rebalancing dates)

        '''
        value = self.start_value

        for t in self.weights.index:

            # rebal at the start of the day
            if t in self.rebaldates:

                w = self.rebal_strategy()  # includes 0% to safe asset
                # end of the day dollar allocation
                # dollar allocation in t (dollar_allocation_t *(1+r_t))
                da = (value * w)@np.diag(self.data.loc[t, :]+1)
                w = da/da.sum()  # end of the day vector of weights

                # store dollar allocation at then time of rebalancing
                da_rebal = da.copy()
                # store value of rebalancing as 1 (rebalancing for rebalancing date)
                rebal = 1

            else:
                # end of the day dollar allocation in t (dollar_allocation_t *(1+r_t))
                da = da @ np.diag(self.data.loc[t, :]+1)

                da, rebal = self.trigger_strategy(da=da, da_rebal=da_rebal)
                # weights drifts
                w = da/da.sum()  # end of the day vector of weights

            value = da.sum()
            self.dollar_allocation.loc[t] = da
            self.weights.loc[t] = w
            self.rebalancing.loc[t] = rebal

    def _check_capped_allocation_type(self) -> bool:
        '''
        Check if allocation type and threshold data is coherent
        '''

        if self.allocation_type in ['ew_cap', 'ew_floor']:
            if isinstance(self.threshold, float):
                print(
                    f'backtesting {self.allocation_type} strategy using a threshold (% return) of {self.threshold}')
            else:
                print(f'threshold is not of {float}')
                return False

        elif self.allocation_type in ['ew_cap_floor']:
            if isinstance(self.threshold[0], float) and isinstance(self.threshold[1], float) and len(self.threshold) == 2:
                print(
                    f'backtesting {self.allocation_type} strategy using thresholds (% return) of {self.threshold}')
            else:
                print(
                    'threshold does not have the correct format, check if its a list with 2 floats')
                return False
        else:
            print(f'{self.allocation_type} strategy not found')
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

        if have_na(self.data) or not self._check_capped_allocation_type():
            return None

        cash_check = self._check_cash_asset()
        if not cash_check:
            return None

        if not self.returns:
            self.data = self.data.pct_change().dropna()

        if self.allocation_type in ['ew_cap', 'ew_floor', 'ew_cap_floor']:
            self.rebal_strategy = self._ew_rebal_w
            self.trigger_strategy = self._ew_trigger_w

        rebaldates = [i for i in range(
            self.window, len(self.data.index), self.rebal)]
        self.rebaldates = self.data.iloc[rebaldates].index
        self.start_date = self.rebaldates[0]

        self.weights = pd.DataFrame().reindex_like(
            pd.DataFrame(self.data.loc[self.start_date:]))
        self.dollar_allocation = self.weights.copy(deep=True)
        self.rebalancing = self.weights.copy(deep=True)

        self._trigger_backtest()

        first_date = pd.DataFrame(
            self.start_value, index=[self.start_date - timedelta(days=1)], columns=[self.allocation_type])
        portfolio_value = pd.DataFrame(self.dollar_allocation.sum(
            axis=1), index=self.weights.index, columns=[self.allocation_type])
        self.portfolio_value = pd.concat([first_date, portfolio_value])
        self.portfolio_value.index.name = 'Date'

        backtest_results = {
            "portfolio_value": self.portfolio_value,
            "dollar_allocation": self.dollar_allocation,
            "weights": self.weights,
            "rebalancing": self.rebalancing
        }

        if self.plot:
            _plot_backtest(self.data, self.allocation_type,  self.start_value, self.start_date,
                           self.portfolio_value, self.dollar_allocation, self.weights)

        print("--- %s seconds ---" % (time.time() - start_time))
        return backtest_results
