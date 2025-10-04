import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import time
from python_scripts.data_and_descriptives import normalize_data, have_na
from pandas.testing import assert_frame_equal


class DAASimulation:
    def __init__(self, data: pd.DataFrame, strategy: str = 'rdd', m: float | pd.DataFrame = 3,
                 kappa: float = 0.7, rebal: int = 30, start_value: int = 1, window: int = 252,
                 sat_core: bool = True, w_bounds: list[float, float] = [0, 1],
                 returns: bool = False, plot: bool = True):
        '''
        Parameters:
        data: satellite and core historic data
        strategy: rdd for relative max drawdown control, edd for excesss max drawdown control,
        mdd for max drawdown control and cppi for constant proportion portfolio insurance
        m: multiplier, could be either fixed or a dynamic multiplier. The dynamic multiplier must come with 
        an upper bound label as 'upper_bound' and a lower bound label as 'lower_bound', 
        and the multiplier value label as 'm_value'. Multiplier data must have a pd.datetime() index
        start_value: start value for the strategy
        kappa: k parameter
        window: for specifing the rolling window size, 0 for a growth window
        rebal: rebalacing frequency
        sat_core: if True the first series is the satellite, otherwise is the second
        w_bounds: lower and upper weights bounds
        returns: if data are returns
        plot: if want to plot the strategy
        '''
        self.data = data
        self.strategy = strategy
        self.m = m
        self.kappa = kappa
        self.rebal = rebal
        self.start_value = start_value
        self.window = window
        self.sat_core = sat_core
        self.w_bounds = w_bounds
        self.returns = returns
        self.plot = plot

    def update_data(self, new_data, check_tickers=True, dynamic_multiplier=None):
        '''
        Update returns data while making sure that dates are consistent,
        new data must start on the same date and must finish in date greater than the current one
        new data until the final date of the past data must be the same
        '''

        if self.dynamic_m:
            assert isinstance(dynamic_multiplier, pd.DataFrame)
            assert self.m.index[0] == dynamic_multiplier.index[0]
            assert self.m.index[-1] < dynamic_multiplier.index[-1]
            assert_frame_equal(
                self.m, dynamic_multiplier.loc[dynamic_multiplier.index <= self.m.index[-1],])
            assert not have_na(dynamic_multiplier)

        start_date = self.data.index[0]
        end_date = self.data.index[-1]

        assert start_date == new_data.index[0]
        assert end_date < new_data.index[-1]

        new_r_sub = new_data.loc[new_data.index <= end_date,]
        if check_tickers:
            assert_frame_equal(self.data, new_r_sub)
        else:
            assert_frame_equal(self.data, new_r_sub, check_names=False)
        # make sure columns are on the same order
        assert (self.data.columns == new_data.columns).all()

        assert not have_na(new_data)

        self.data = new_data.copy(deep=True)
        if self.dynamic_m:
            self.m = dynamic_multiplier.copy(deep=True)

        print('data updated, running backtest...')

        backtest_results = self.daa_simulation()

        return backtest_results

    def _sub_history(self, date: pd.Timestamp):
        '''
        Subset the history data for floor computation
        '''
        history_sub = self.history.loc[self.history.index <=
                                       date, self.account_name]
        return history_sub

    def _mdd_floor(self, account_value: float, date: pd.Timestamp, **kwargs) -> float:
        """
        Updates MDD floor, can use a rolling window

        Parameters come from daa_strategies parameters
        """
        history_sub = self._sub_history(date)

        if self.window == 0:

            account_max = history_sub.max()
            account_drawdown = account_value / account_max - 1
            self.z.loc[date] = account_drawdown

            floor_value = self.kappa * account_max  # MDD floor

        else:

            account_max = history_sub.loc[date -
                                          timedelta(days=self.window):date].max()

            account_drawdown = account_value / account_max - 1
            self.z.loc[date] = account_drawdown
            # Rolling MDD Floor
            floor_value = self.kappa * account_max

        return floor_value

    def _cppi_floor(self, date: pd.Timestamp, **kwargs) -> float:
        """
        Updates CPPI floor

        Parameters come from daa_strategies parameters
        """
        core_value = self.core.loc[date]

        self.z.loc[date] = core_value
        # End of the day floor value
        floor_value = self.kappa*core_value

        return floor_value

    def _rdd_floor(self, account_value: float, date: pd.Timestamp, **kwargs) -> float:
        """
        Updates z (portfolio value/benchmark value), floor and cushion

        Parameters come from daa_strategies parameters
        """

        core_value = self.core.loc[date]

        # End of the day floor value
        # using previous day close = today's pre-open portfolio value/core value
        current_z = account_value/core_value
        self.relative_value.loc[date] = current_z

        rel_val_max = self.relative_value.max().iloc[0]
        floor_value = self.kappa*rel_val_max*core_value  # RDD Floor

        # relative drawdown
        self.z.loc[date] = current_z/rel_val_max - 1
        return floor_value

    def _edd_floor(self, account_value: float, date: pd.Timestamp, **kwargs) -> float:
        """ 
        Updates z (portfolio drawdown - benchmark drawdown), floor and cushion

        Parameters come from daa_strategies parameters
        """
        history_sub = self._sub_history(date)

        core_sub = self.core[self.core.index <= date]

        core_value = core_sub.loc[date]

        core_max = core_sub.max()
        account_max = history_sub.max()

        floor_value = self.kappa * \
            (account_max/core_sub.loc[history_sub.idxmax()]
             )*core_value  # EDD Floor

        # Excess Drawdown
        account_drawdown = account_value / account_max - 1
        core_drawdown = core_value / core_max - 1
        self.z.loc[date] = account_drawdown - core_drawdown

        return floor_value

    def _end_of_day_estimation(self, sat_w: float, account_value: float, date: pd.Timestamp) -> list[float, float]:
        ''' 
        Computes end of the day weights and account value

        Parameters come from daa_strategies parameters
        '''

        core_w = 1-sat_w
        risky_alloc = account_value*sat_w*(1+self.sat_ret.loc[date])
        safe_alloc = account_value*core_w*(1+self.core_ret.loc[date])
        account_value = risky_alloc + safe_alloc
        sat_w = risky_alloc/account_value

        return account_value, sat_w

    def _check_allowed_strategies(self) -> bool:
        '''
        Check if strategy is allowed

        Parameters come from daa_strategies parameters
        '''

        allowed_strategies = ['rdd', 'edd', 'mdd', 'cppi']

        if self.strategy in allowed_strategies:
            return True

        else:
            print(
                f'Strategy {self.strategy} not found, please select between {allowed_strategies}')
            return False

    def _retrieve_plot_names(self) -> None:
        '''
        Define strategy plot titles and legends

        Parameters come from daa_strategies parameters
        '''

        metric_name = {'rdd': 'Relative Drawdown',
                       'edd': 'Excess Drawdown',
                       'mdd': 'Drawdown',
                       'cppi': 'Benchmark'}

        plot_name = self.strategy.upper()
        self.plot_titles = [f'{plot_name} Strategy Evolution',
                            f'{metric_name[self.strategy]} Evolution', f'{plot_name} Strategy Floor']
        self.plot_legends = [[f'{plot_name} Strategy', 'Satellite', 'Core'],
                             [f'{plot_name} Strategy', 'Floor']]

    def _strategy_summary(self) -> None:
        '''
        Prints the strategy summary

        Parameters come from daa_strategies parameters
        '''

        strategy_name = {'rdd': 'Relative Maximum Drawdown',
                         'edd': 'Excess Maximum Drawdown',
                         'mdd': 'Maximum Drawdown',
                         'cppi': 'Constant Proportion Portfolio Insurance'}

        if isinstance(self.m, pd.DataFrame):
            print_m = 'dynamic multiplier'
        else:
            print_m = self.m
        print(
            f'Summary: \n strategy: {strategy_name[self.strategy]} \n m = {print_m} \n kappa: {self.kappa} \n \
            rebalancing frequency: {self.rebal} \n weight bounds: {self.w_bounds}')

    def _get_core_sat(self) -> None:
        '''
        Store core and satellite returns based on parameters

        Parameters come from daa_strategies parameters
        '''

        if self.sat_core:
            indexes = (0, 1)
        else:
            indexes = (1, 0)

        if self.returns:
            self.sat_ret = self.data.iloc[:, indexes[0]]
            self.core_ret = self.data.iloc[:, indexes[1]]

            norm_data = normalize_data(
                self.data, are_returns=True, base=self.start_value, plot=False, return_data=True)
            self.satellite = norm_data.iloc[:, indexes[0]]
            self.core = norm_data.iloc[:, indexes[1]]

        else:
            self.satellite = self.data.iloc[:, indexes[0]]
            self.core = self.data.iloc[:, indexes[1]]
            self.sat_ret = self.satellite.pct_change().dropna()
            self.core_ret = self.core.pct_change().dropna()

    def _get_history_and_z(self) -> None:
        '''
        Creates history and z dataframes for storing backtest data

        Parameters come from daa_strategies parameters
        '''

        history_index = self.core.index

        if self.dynamic_m:
            history_columns = [
                self.account_name, 'sat_w', 'cushion', 'floor', 'imp_m', 'rebal']
            start_in_zero = ['sat_w', 'cushion', 'floor', 'imp_m', 'rebal']
        else:
            history_columns = [self.account_name, 'sat_w', 'cushion', 'floor']
            start_in_zero = ['sat_w', 'cushion', 'floor']

        self.history = pd.DataFrame(
            columns=history_columns, index=history_index)
        first_row = self.history.index[0]
        self.history.loc[first_row, self.account_name] = self.start_value
        self.history.loc[first_row, start_in_zero] = 0
        self.history = self.history.astype(
            {self.account_name: 'float', 'sat_w': 'float', 'cushion': 'float', 'floor': 'float'})

        self.z = pd.DataFrame(columns=['z'], index=history_index)
        if self.strategy in ['cppi']:
            # z value is the benchmark for CPPI so starts in start_value
            self.z.iloc[0] = self.start_value
        else:
            # z value for RDD is relative drawdown, for EDD is excess drawdown, for MDD is drawdown, all starts in 0
            self.z.iloc[0] = 0

    def _plot_strategy(self) -> None:
        '''
        Plot strategy backtest and metrics

        Parameters come from daa_strategies parameters
        '''
        self._retrieve_plot_names()

        fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(25, 15))

        ax1[0].plot(pd.concat([self.backtest_result['history']
                    [self.account_name], self.backtest_result['satellite'], self.backtest_result['core']], axis=1))
        ax1[0].set_title(self.plot_titles[0])
        ax1[0].legend(self.plot_legends[0], loc="upper right")

        ax1[1].plot(self.backtest_result['z'])
        ax1[1].set_title(self.plot_titles[1])

        ax2[0].plot(self.backtest_result['history']['sat_w'].iloc[1:])
        ax2[0].set_title('Satellite Allocation')

        ax2[1].plot(self.backtest_result['history']
                    [[self.account_name, 'floor']].iloc[1:])
        ax2[1].set_title(self.plot_titles[2])
        ax2[1].legend(self.plot_legends[1], loc="upper right")

    def _check_dynamic_multiplier(self) -> bool:
        '''
        Check that the dynamic multiplier data satifies the defined scheme

        Parameters come from daa_strategies parameters
        '''

        allowed_columns = ['lower_bound', 'm_value', 'upper_bound']
        check_allowed_columns = set(
            allowed_columns).difference(set(self.m.columns))
        if len(check_allowed_columns) > 0:
            print(f'{check_allowed_columns} missing in dynamic multiplier data')
            return False
        if have_na(self.m):
            return False

        return True

    def _get_floor_and_cushion(self, account_value: str, i: pd.DatetimeIndex) -> tuple[float, float]:
        '''
        Computes the floor and cushion of the strategy during the backtesting

        Parameters come from daa_strategies parameters
        '''

        floor_calculation = {'rdd': self._rdd_floor, 'edd': self._edd_floor,
                             'mdd': self._mdd_floor, 'cppi': self._cppi_floor}

        floor = floor_calculation[self.strategy]

        floor_value = floor(account_value=account_value, date=i)

        # End of the day cushion value
        cushion = (account_value - floor_value)/account_value

        return cushion, floor_value

    def _static_m_backtest(self) -> None:
        '''
        Executes the backtest for a static multiplier

        Parameters come from daa_strategies parameters
        '''

        # initialize parameters for backtesting
        rebaldates = [i for i in range(0, len(self.sat_ret.index), self.rebal)]
        rebaldates = self.sat_ret.iloc[rebaldates].index

        account_value = self.start_value
        floor_value = self.kappa*self.start_value
        cushion = (account_value - floor_value)/account_value

        for i in self.sat_ret.index:

            if i in rebaldates:

                sat_w = self.m*cushion  # use estimated multiplier
                # long_only constraints & bounds
                sat_w = np.minimum(sat_w, self.w_bounds[1])
                sat_w = np.maximum(sat_w, self.w_bounds[0])

            # END OF DAY
            account_value, sat_w = self._end_of_day_estimation(
                sat_w, account_value, i)
            self.history.loc[i, [self.account_name, 'sat_w']] = [
                account_value, sat_w]

            cushion, floor_value = self._get_floor_and_cushion(
                account_value, i)

            self.history.loc[i, ['cushion', 'floor']] = [cushion, floor_value]

    def _dynamic_m_backtest(self) -> None:
        '''
        Executes the backtest for a dynamic multiplier

        Parameters come from daa_strategies parameters
        '''
        account_value = self.start_value
        floor_value = self.kappa*self.start_value
        cushion = (account_value - floor_value)/account_value

        # start with something higher than the upper bound so that will trigger the rebalancing at start
        imp_m = self.m.iloc[0].loc['upper_bound'] + 1
        threshold = 0

        for i in self.sat_ret.index:

            # risk based rebalancing (implied multiplier lower than lower bound or higher than upper bound)
            if self.m.loc[i, 'upper_bound'] < imp_m or self.m.loc[i, 'lower_bound'] > imp_m \
                    or (threshold > self.rebal and self.rebal != 0):
                sat_w = self.m.loc[i, 'm_value'] * \
                    cushion  # use estimated multiplier
                # long_only constraints & bounds
                sat_w = np.minimum(sat_w, self.w_bounds[1])
                sat_w = np.maximum(sat_w, self.w_bounds[0])

                self.history.loc[i, 'rebal'] = 1

                threshold = 0

            else:
                threshold += 1

            # END OF DAY
            account_value, sat_w = self._end_of_day_estimation(
                sat_w, account_value, i)
            self.history.loc[i, [self.account_name, 'sat_w']] = [
                account_value, sat_w]

            cushion, floor_value = self._get_floor_and_cushion(
                account_value, i)

            imp_m = sat_w/cushion

            self.history.loc[i, ['cushion', 'floor', 'imp_m']] = [
                cushion, floor_value, imp_m]

        self.history['rebal'] = self.history['rebal'].fillna(0)

    def daa_simulation(self) -> dict:
        """
        Backtest of Dynamic Asset Allocation Strategies
        """

        start_time = time.time()

        # checks for data and strategy inputs
        if have_na(self.data) | (not self._check_allowed_strategies()):
            return None

        # checks for multiplier data
        if isinstance(self.m, pd.DataFrame):
            self.dynamic_m = True
            if not self._check_dynamic_multiplier():
                return None
        elif not isinstance(self.m, float):
            print('if m is not a pd.DataFrame, it must be a float')
            return None
        else:
            self.dynamic_m = False

        self._strategy_summary()

        self.account_name = self.strategy + '_account'

        self._get_core_sat()

        self._get_history_and_z()

        # used for storing the relative value between the portfolio and the benchmark, used for the floor computation
        self.relative_value = self.z.copy(deep=True)

        # backtest
        if self.dynamic_m:
            self._dynamic_m_backtest()
        else:
            self._static_m_backtest()

        self.backtest_result = {
            "history": self.history,
            "z": self.z,
            "core": self.core,
            "satellite": self.satellite
        }

        if self.plot:
            self._plot_strategy()

        print("--- %s seconds ---" % (time.time() - start_time))

        return self.backtest_result
