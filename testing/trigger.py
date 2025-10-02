
import json
import pickle
from pandas.testing import assert_frame_equal
import pandas as pd
from python_scripts.diversification import TriggerSimulation

# Python script for testing
# use python -m testing.trigger

returns = pd.read_csv('testing/returns.csv', index_col=0)
returns.index = pd.to_datetime(returns.index)
with open(r'testing\parameters\ew_trigger_parameters.json', 'r') as f:
    parameters = json.load(f)

with open(r'testing\results\ew_trigger.pkl', 'rb') as f:
    testing_backtest_results = pickle.load(f)


def trigger_test(returns: pd.DataFrame, parameters: dict, testing_backtest_results: dict[pd.DataFrame]) -> bool:

    trigger_simulation = TriggerSimulation(returns, allocation_type=parameters['allocation_type'], safe_asset=parameters['safe_asset'],
                                           threshold=parameters['threshold'], window=parameters['window'], rebal=parameters['rebal'])
    backtest_results = trigger_simulation.trigger_simulation()

    i = 0
    for key in backtest_results:
        try:
            assert_frame_equal(
                testing_backtest_results[key], backtest_results[key])
        except AssertionError as e:
            i += 1
            print(f"Comparison failed for: {key}")
            print(e)  # This prints the detailed message explaining the mismatch

    if i > 0:
        print('Test failed')
        return False

    else:
        print('Test passed')
        return True


trigger_test(returns, parameters, testing_backtest_results)
