import pandas as pd
import matplotlib.pyplot as plt
from python_scripts.data_and_descriptives import normalize_data


def _plot_backtest(data: pd.DataFrame, allocation_type: str,  start_value: int, start_date: pd.DatetimeIndex,
                   portfolio_value: pd.DataFrame, dollar_allocation: pd.DataFrame, weights: pd.DataFrame) -> None:
    '''
    Plot the strategy perfomance and metrics

    '''

    allocation_type = allocation_type.upper()
    data = data.loc[start_date:,]
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(25, 15))

    ax1[0].plot(portfolio_value)
    ax1[0].set_title(f'{allocation_type} Strategy')

    norm_data = normalize_data(data, True, start_value, False, True)
    norm_data = pd.concat([norm_data, portfolio_value], axis=1, join='inner')
    ax1[1].plot(norm_data)
    ax1[1].set_title(f'{allocation_type} Strategy Components')
    ax1[1].legend([x.upper() for x in norm_data.columns], loc="upper left")

    ax2[0].plot(dollar_allocation)
    ax2[0].set_title('End of the Day Dollar Allocation')
    ax2[0].legend([x.upper()
                  for x in dollar_allocation.columns], loc="upper left")

    ax2[1].plot(weights)
    ax2[1].set_title('End of the Day Weights')
    ax2[1].legend([x.upper() for x in weights.columns], loc="upper left")
