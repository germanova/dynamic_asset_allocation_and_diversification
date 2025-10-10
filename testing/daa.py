
import json
import pickle
from pandas.testing import assert_frame_equal
from pandas.testing import assert_series_equal
import pandas as pd
from python_scripts.dynamic_asset_allocation import DAASimulation

# Python script for testing
# use python -m testing.daa

returns = pd.read_csv('testing/returns.csv', index_col=0)
returns.index = pd.to_datetime(returns.index)
with open(r'testing\parameters\rdd_parameters.json', 'r') as f:
    parameters = json.load(f)

with open(r'testing\results\rdd.pkl', 'rb') as f:
    testing_backtest_results = pickle.load(f)


def daa_test(returns: pd.DataFrame, parameters: dict, testing_backtest_results: dict[pd.DataFrame]) -> bool:

    daa_simulation = DAASimulation(data=returns, strategy=parameters['strategy'],
                                   m=parameters['m'], kappa=parameters['kappa'], rebal=parameters['rebal'],
                                   start_value=parameters['start_value'],
                                   window=parameters['window'], sat_core=parameters['sat_core'],
                                   w_bounds=parameters['w_bounds'], returns=parameters['returns'],
                                   plot=False)
    backtest_results = daa_simulation.daa_simulation()

    i = 0
    for key in backtest_results:
        print(f'Testing {key} data...')
        if isinstance(testing_backtest_results[key], pd.Series):
            try:
                assert_series_equal(
                    testing_backtest_results[key], backtest_results[key])
            except AssertionError as e:
                i += 1
                print(f"Comparison failed for: {key}")
                print(e)  # This prints the detailed message explaining the mismatch

        elif isinstance(testing_backtest_results[key], pd.DataFrame):
            try:
                assert_frame_equal(
                    testing_backtest_results[key], backtest_results[key])
            except AssertionError as e:
                i += 1
                print(f"Comparison failed for: {key}")
                print(e)

    if i > 0:
        print('Test failed')
        return False

    else:
        print('Test passed')
        return True


daa_test(returns[['XLK', 'AUSF']], parameters, testing_backtest_results)
