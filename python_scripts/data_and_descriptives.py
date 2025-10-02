import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yfin
from datetime import timedelta
from scipy.stats import skew, kurtosis, normaltest, norm
from typing import Callable
import scipy.cluster.hierarchy as sch
import numpy as np

import matplotlib.pylab as pylab
from matplotlib import rcParamsDefault


def config_plt() -> None:
    '''
    Initial config for matplotlib
    '''
    pylab.rcParams.update(rcParamsDefault)
    plt.rcParams['figure.figsize'] = (16, 9)
    plt.style.use('ggplot')


config_plt()


def config_subplot() -> None:
    '''
    Config for subplots in matplotlib
    '''
    params = {'legend.fontsize': 'xx-large',
              'figure.figsize': (16, 9),
              'axes.labelsize': 'xx-large',
              'axes.titlesize': 'xx-large',
              'xtick.labelsize': 'xx-large',
              'ytick.labelsize': 'xx-large'}
    pylab.rcParams.update(params)


def plot_distribution(returns: pd.DataFrame, tickers: list | None = None,
                      overlap: bool = False, alpha: int = 0.7) -> None:
    '''
    Plot the distribution of the selected returns

    parameters:
    tickers: if None it will use of the data in returns, must match returns columns
    overlap: if you want distributions to show overlapped
    alpha: transparency parameter of matplotlib
    '''

    if tickers:
        returns = returns[tickers]
    else:
        tickers = returns.columns.tolist()

    if overlap:
        for i in tickers:
            ticker_data = returns[i]
            plt.hist(ticker_data, edgecolor='black', alpha=alpha, label=i)
    else:
        plt.hist(returns, edgecolor='black', alpha=alpha,
                 label=tickers)

    # Add labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of Values')
    plt.legend()  # Add legend to differentiate between histograms
    plt.show()


def fix_na_data_yahoo(df: pd.DataFrame, max_na: int = 3) -> pd.DataFrame:
    """
    Fix NA from Yahoo Finance using the following rules:
    1. Drop rows filled with NA
    2. Drop columns that have more than max_na
    3. Use backward fill for columns with NA less than max_na

    Parameters:
    max_na: the maximum number of NA's allowed in a column,
      above this parameter, columns will be removed
    """
    # Drop rows filled with NA
    df = df.dropna(axis=0, how='all')
    old_dates = set(df.index)
    rem_na = old_dates.difference(df.index)
    if len(rem_na) > 0:
        print(f'Total of {len(rem_na)} rows filled with NA: {rem_na}')
    old_tickers = set(df.columns)
    # Drop columns that have more than max_na
    df = df.dropna(axis=1, thresh=len(df) - max_na)
    rem_na = old_tickers.difference(set(df.columns))
    if len(rem_na) > 0:
        print(
            f'Total of {len(rem_na)} tickers dropped: {rem_na} \
              using a threshold of max NA of: {max_na}')
    rem_na = df.loc[:, df.isna().any()].columns
    if len(rem_na) > 0:
        print(
            f'Total of {len(rem_na)} remaining tickers with NA: {rem_na.tolist()}, using backward fill...')
    # Use backward fill for columns with NA less than max_na
    df = df.bfill()

    return df


def have_na(data: pd.DataFrame) -> bool:
    '''
    Check if DataFrame has NA's

    '''
    cols_na = sum(data.isna().sum() > 0)
    if cols_na > 0:
        print(f'data contains {cols_na} columns with NA')
        return True
    else:
        return False


def data_yahoo(tickers: list, freq: str, start: str, end: str, data: list = ['Adj Close'],
               verbose: bool = False) -> pd.DataFrame | None:
    '''
    Attempts to retrieve requested yfinance data

    parameters:
    start: format 'yyyy-mm-dd' 
    end: format 'yyyy-mm-dd' 
    freq: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    data: ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    verbose: if retrieve NA data

    '''

    try:
        df = yfin.download(tickers, start=start, end=end,
                           interval=freq, auto_adjust=False)
        if len(data) == 1:
            data = data[0]
        df = df[data]

        if verbose == True:
            print(df.isna().sum())
        return df
    except:
        print('could not retrieve data')
        pass


def normalize_data(data: pd.DataFrame, are_returns: bool = True, base: int = 1, plot: bool = True,
                   return_data: bool = False) -> None | pd.DataFrame:
    '''
    Normalize price data to make it start at base (by default 1)

    parameters:
    are_returns: used when returns are imputed as the data parameter, otherwise data must be closing prices
    base: start value
    plot: if you want to plot the data
    return_data: if you want to return the data

    '''

    if are_returns:
        print('using RETURNS...')
        results = (data + base).cumprod()
        # add first date as base value
        first_date = pd.DataFrame(
            base, index=[data.index[0] - timedelta(days=1)], columns=data.columns)
        results = pd.concat([first_date, results])

    else:
        print('using CLOSING PRICES...')
        results = data/data.iloc[0] * base

    if plot:
        plt.plot(results)
        plt.legend(results.columns)
        plt.title('ETF Data')
    if return_data:
        return results


def relative_value(data: pd.DataFrame, sat_core: bool = True, plot: bool = True,
                   return_data: bool = False) -> None | pd.DataFrame:
    '''
    Computes the relative value of  a satellite/core 

    parameters:
    sat_core = True for satellite,core / False for core,satellite
    plot: if you want to plot the data
    return_data: if you want to return the data  

    '''

    if sat_core:
        z = data[data.columns[0]]/data[data.columns[1]]
    else:
        z = data[data.columns[1]]/data[data.columns[0]]
    if plot:
        plt.plot(z)
        plt.title('Satellite/Core Relative Value')
    if return_data:
        return z


def rolling_max_drawdown(data: pd.DataFrame, window: int = 252, min_periods: int = 1, plot: bool = True) -> None:
    '''
    Return the max drawdown in the past window days for each day in the series.


    Parameters:
    window: size of rolling window for mdd estimation
    min_periods: minimum periods to estimate the initial max drawdown. 
        Fix it to 1 with a daily DataFrame, if you want to let the first 'window' days data have an expanding window
    plot: if you want to plot the data

    '''

    print(f'Window: {window}')

    roll_max = data.rolling(window, min_periods=min_periods).max()
    dd = data/roll_max - 1
    mdd = dd.rolling(window, min_periods=min_periods).min()

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 15))
        ax1.plot(roll_max)
        ax1.set_title('Rolling Max')
        ax1.legend(roll_max.columns, loc="upper right")
        ax2.plot(mdd)
        ax2.set_title('MDD')
        ax2.legend(mdd.columns, loc="upper right")
    return mdd


def convert_rate(target: float, p: int) -> int:
    '''
    Convert an annualized rate into the required frequency

    Parameters:
    target = annualized target of return
    p: periods in a year

    '''

    target = (1+target)**(1/p)-1
    return target


def drawdown_series(returns: pd.DataFrame) -> pd.DataFrame:
    '''
    Return the drawdown series for a set of returns

    '''
    norm_data = normalize_data(returns, True, 1, False, True)
    drawdown = norm_data/norm_data.cummax() - 1

    return drawdown


def average_drawdown(returns: pd.DataFrame, **kwargs) -> pd.Series:
    '''
    Return the average drawdown for a set of returns
    '''
    drawdown = drawdown_series(returns)
    # drawdown[drawdown  == 0] = np.nan

    # include 0's in mean estimation
    average_drawdown = -drawdown.mean()
    return average_drawdown


def max_drawdown(returns: pd.DataFrame, **kwargs) -> pd.Series:
    '''
    Return the maximum drawdown for a set of returns
    '''
    drawdown = drawdown_series(returns)
    max_drawdown = drawdown.min()
    return -max_drawdown


def total_return(returns: pd.DataFrame) -> pd.Series:
    '''
    Return the total period return for a set of returns
    '''
    norm_data = normalize_data(returns, True, 1, False, True)
    metric = norm_data.iloc[-1] / norm_data.iloc[0] - 1
    return metric


def annualized_returns(returns: pd.DataFrame, p: int = 252) -> pd.Series:
    '''
    Return the annualized return for a set of returns
    '''
    cum_ret = (1+returns).prod()
    metric = cum_ret**(p/len(returns)) - 1
    return metric


def annualized_volatility(returns: pd.DataFrame, p: int = 252, **kwargs) -> pd.Series:
    '''
    Return the annualized volatility for a set of returns
    '''

    metric = returns.std()*(p**0.5)
    return metric


def annualized_tracking_error(returns: pd.DataFrame, benchmark: str, p: int = 252) -> pd.Series:
    """
    Return the annualized tracking error for a set of excess returns

    Parameters:
    benchmark: the column name in the dataframe that identifies the benchmark
    p: periods in a year
    """
    print(f'{annualized_tracking_error.__name__}: using {
          benchmark} as benchmark')
    excess_ret = excess_returns(returns, benchmark)
    te = excess_ret.std()
    te = te*(p**0.5)
    return te


def excess_returns(returns: pd.DataFrame, benchmark: str) -> pd.Series:
    '''
    Return the excess returns for a set of returns

    Parameters:
    benchmark: the column name in the dataframe that identifies the benchmark
    '''

    print(f'{excess_returns.__name__}: using {benchmark} as benchmark')
    returns_sub = returns.subtract(returns[benchmark], axis=0)
    returns_sub = returns_sub.drop(columns=benchmark)
    return returns_sub


def annualized_excess_returns(returns: pd.DataFrame, benchmark: str) -> pd.Series:
    '''
    Return the annualized excess returns for a set of returns

    Parameters:
    benchmark: the column name in the dataframe that identifies the benchmark
    '''
    excess_ret = excess_returns(returns, benchmark)
    ann_excess_ret = annualized_returns(excess_ret)
    return ann_excess_ret


def information_ratio(returns: pd.DataFrame, benchmark: str, p: int = 252) -> pd.Series:
    """
    Return the information ratio for a set of returns

    Parameters:
    benchmark: the column name in the dataframe that identifies the benchmark
    p: periods in a year
    """
    ann_excess_ret = annualized_excess_returns(returns, benchmark)
    ann_te = annualized_tracking_error(returns, benchmark, p)
    ir = ann_excess_ret/ann_te
    return ir


def beta_estimate(returns: pd.DataFrame, benchmark: str, **kwargs) -> pd.Series:
    '''
    Return the beta estimation for a set of returns

    Parameters:
    benchmark: the column name in the dataframe that identifies the benchmark
    '''

    betas = returns.cov()[benchmark]/returns[benchmark].std()**2
    betas = betas.drop(labels=[benchmark])

    return betas


def treynor_ratio(returns: pd.DataFrame, benchmark: str, target: float = 0.05, p: int = 252) -> pd.Series:
    '''
    Return the Treynor Ratio for a set of returns

    Parameters:
    benchmark: the column name in the dataframe that identifies the benchmark
    target = annualized target of return
    p: periods in a year
    '''
    print(f'{treynor_ratio.__name__}: using a target of: {
          target} and a benchmark of {benchmark} for computing beta')
    target = convert_rate(target, p)
    excess_return = annualized_returns(returns - target, p)
    excess_return = excess_return.drop(labels=[benchmark])

    betas = beta_estimate(returns, benchmark)
    tr = excess_return / betas
    return tr


def burke_drawdown_measure(returns: pd.DataFrame, **kwargs) -> pd.Series:
    '''
    Return the Burke's drawdown measure for a set of returns

    '''
    dd = drawdown_series(returns)
    dd = dd**2
    burke = dd.sum()**(1/2)
    return burke


def modified_burke_drawdown_measure(returns: pd.DataFrame, **kwargs) -> pd.Series:
    '''
    Return the Modified Burke's drawdown measure for a set of returns

    '''
    # divide across the len of the hole dataset
    mod_burke = burke_drawdown_measure(returns) * len(returns)**(-1/2)
    return mod_burke


def m2_measure(returns: pd.DataFrame, benchmark: str, target: float = 0.05, p: int = 252) -> pd.Series:
    '''
    Return the M2 measure for a set of returns

    Parameters:
    benchmark: the column name in the dataframe that identifies the benchmark
    target = annualized target of return
    p: periods in a year
    '''
    bench = returns[benchmark]
    returns = returns.drop(columns=[benchmark])
    sharpe_ratio = ratio_metric(
        returns, metric='sharpe_ratio', target=target, p=p)
    m2 = sharpe_ratio*annualized_volatility(bench) + target
    return m2


def partial_moments(returns: pd.DataFrame, target: float = 0.05, p: int = 252, order: int = 2, type='lower', **kwargs) -> pd.Series:
    '''
    Return the Lower/Higher Partial Moments for a set of returns

    Parameters:
    target = annualized target of return
    p: periods in a year
    order: order of the partial moment estimation
    type: either higher or lower
    '''

    if type in ['higher', 'lower']:
        print(f'computing {type}_{partial_moments.__name__} of order: {order}')
    else:
        print(f'type {type} not found')
        return None
    target = convert_rate(target, p)
    returns = returns - target
    if type == 'lower':
        returns *= -1
    # measure should include 0's
    returns = returns.clip(lower=0)
    returns = returns**(order)
    pm = returns.mean()**(1/order)
    return pm


def omega_ratio_sum_approx(returns: pd.DataFrame, target: float = 0.05, p: int = 252, **kwargs) -> pd.Series:
    '''
    Return the omega ratio for a set of returns

    Parameters:
    target = annualized target of return
    p: periods in a year
    '''
    target = convert_rate(target, p)
    returns = returns - target
    omega = returns[returns > 0].sum()/-returns[returns < 0].sum()

    return omega


def omega_ratio_put_option(returns: pd.DataFrame, target: float = 0.05, p: int = 252, **kwargs) -> pd.Series:
    '''
    Return the Omega Ratio for a set of returns

    Parameters:
    target = annualized target of return
    p: periods in a year
    '''
    target = convert_rate(target, p)
    exc_ret = returns.mean() - target
    lpm = partial_moments(returns, target, p, order=1, type='lower')
    omega = 1 + exc_ret/lpm

    return omega


def target_downside_deviation(returns: pd.DataFrame, target: float = 0.05, p: int = 252, **kwargs) -> pd.Series:
    """
    Return the target downside deviation for a set of returns

    Parameters:
    target = annualized target of return
    p: periods in a year
    """

    lpm = partial_moments(returns=returns, target=target,
                          p=p, order=2, type='lower')
    tdd = lpm**(1/2)
    return tdd


def sortino_ratio(returns: pd.DataFrame, target: float = 0.05, p: int = 252) -> pd.Series:
    """
    Return the Sortino Ratio for a set of returns

    Parameters:
    target = annualized target of return
    p: periods in a year
    """
    target = convert_rate(target, p)
    exc_ret = returns.mean() - target
    tdd = target_downside_deviation(returns, target, p)
    sortino = exc_ret/tdd

    return sortino


def kappa_risk_measure(returns: pd.DataFrame, target: float = 0.05, p: int = 252, order: int = 3,  **kwargs) -> pd.Series:
    """
    Return the kappa risk measure for a set of returns

    Parameters:
    target = annualized target of return
    p: periods in a year
    """
    lpm = partial_moments(returns=returns, target=target,
                          p=p, order=order, type='lower')
    tdd = lpm**(1/order)
    return tdd


def kappa_ratio(returns: pd.DataFrame, target: float = 0.05, p: int = 252, order: int = 3, **kwargs) -> pd.Series:
    '''
    Return the Kappa Ratio for a set of returns

    Parameters:
    target = annualized target of return
    p: periods in a year
    order: order of the lower partial moment
    '''
    target = convert_rate(target, p)
    exc_ret = returns.mean() - target
    lpm = kappa_risk_measure(returns, target, p, order=order, type='lower')
    kappa = exc_ret/lpm

    return kappa


def sharpe_ratio_on_avg(returns: pd.DataFrame, target: float = 0.05, p: int = 252, **kwargs) -> pd.Series:
    '''
    Return the Sharpe Ratio for a set of returns without annualizing the excess return but using the mean return

    Parameters:
    target = annualized target of return
    p: periods in a year
    order: order of the lower partial moment
    '''
    target = convert_rate(target, p)
    exc_ret = returns.mean() - target
    vol = returns.std()
    sharpe_ratio = exc_ret/vol

    return sharpe_ratio


def gain_loss_ratio(returns: pd.DataFrame, target: float = 0.05, p: int = 252) -> pd.Series:
    """
    Return the Gain-Loss Ratio for a set of returns

    Parameters:
    target = annualized target of return
    p: periods in a year
    """

    hpm = partial_moments(returns=returns, target=target,
                          p=p, order=1, type='higher')
    lpm = partial_moments(returns=returns, target=target,
                          p=p, order=1, type='lower')
    glr = hpm/lpm

    return glr


def upside_potential_ratio(returns: pd.DataFrame, target: float = 0.05, p: int = 252) -> pd.Series:
    """
    Return the Upside Potential Ratio for a set of returns

    Parameters:
    target = annualized target of return
    p: periods in a year
    """
    hpm = partial_moments(returns=returns, target=target,
                          p=p, order=1, type='higher')
    tdd = target_downside_deviation(returns, target, p)
    upr = hpm/tdd

    return upr


def value_at_risk(returns: pd.DataFrame, percentage: float = 0.05, estimation_type: str = 'sample', **kwargs) -> pd.Series:
    """
    Return the Value at Risk for a set of returns

    Parameters:
    percentage: the value at which the VaR will be estimated
    estimation_type: either 'sample' or 'cornish_fisher'. 'sample' gets the quantile value at the given percentage 
        'cornish_fisher' computes the VaR using a z Score, mean, std, skewness and kurtosis values

    """
    print(f'{value_at_risk.__name__}: VaR estimated using a {
          estimation_type} assumption and a confidence level of: {(1-percentage)*100}%')
    if estimation_type == 'sample':
        var = -returns.quantile(percentage)
    elif estimation_type == 'cornish_fisher':
        z = pd.Series(norm.ppf(percentage), index=returns.columns)
        s = population_value(returns, metric='skewness')
        k = population_value(returns, metric='kurtosis')
        z_cf = (z +
                (z**2 - 1)*s/6 +
                (z**3 - 3*z)*k/24 -
                (2*z**3 - 5*z)*(s**2)/36
                )

        var = -(returns.mean() + z_cf*returns.std(ddof=0))
    return var


def conditional_value_at_risk(returns: pd.DataFrame, percentage: float = 0.05, **kwargs) -> pd.Series:
    '''
    Return the Conditional Value at Risk for a set of returns

    Parameters:
    percentage: the value at which the CVaR will be estimated
    '''
    cvar = -returns[returns <= returns.quantile(percentage)].mean()
    return cvar


def ratio_metric(returns: pd.DataFrame, metric: str = 'sharpe_ratio', target: float = 0.0525,
                 p: int = 252, percentage=0.05) -> pd.Series:
    """
    Return the requested ratio for a set of returns. All of this ratios have in common that
      its numerator is the excess return, it only changes the way risk is estimated

    Parameters:
    metric: either sharpe_ratio, calmar_ratio, sortino_ratio, sterling_ratio, burke_ratio,
        modified_burke_ratio, sample_var_ratio, c_f_var_ratio, cvar_ratio, omega_ratio,
        kappa_ratio
    target = annualized target of return
    p: periods in a year 
    percentage: the value at which the Var or CVaR will be estimated, only used for certain ratio estimations
    """
    print(f'{ratio_metric.__name__}: estimating {
          metric} using target of {target}')
    metric_map = {'sharpe_ratio': annualized_volatility, 'calmar_ratio': max_drawdown,
                  'sortino_ratio': target_downside_deviation, 'sterling_ratio': average_drawdown,
                  'burke_ratio': burke_drawdown_measure, 'modified_burke_ratio': modified_burke_drawdown_measure,
                  'sample_var_ratio': value_at_risk, 'c_f_var_ratio': value_at_risk,
                  'cvar_ratio': conditional_value_at_risk,
                  'kappa_ratio': kappa_risk_measure}

    if metric == 'c_f_var_ratio':
        estimation_type = 'cornish_fisher'
    else:
        estimation_type = 'sample'

    target = convert_rate(target, p)
    excess_return = annualized_returns(returns - target, p)
    risk = metric_map[metric](returns=returns, p=p, target=target,
                              percentage=percentage, estimation_type=estimation_type)

    sharpe_ratio = excess_return/risk
    return sharpe_ratio


def scipy_metric(returns: pd.DataFrame, metric: str = 'skewness') -> pd.Series:
    '''
    Return a scipy metric for a set of returns

    metric: either skewness, kurtosis, normal_test

    From scipy documantation: 
        for normal_test: If the p-value is “small” - that is, if there is a low
            probability of sampling data from a normally distributed
            population that produces such an extreme value of the statistic
            - this may be taken as evidence against the null hypothesis in favor of the
            alternative: the weights were not drawn from a normal distribution.

    '''
    metric_map = {'skewness': skew,
                  'kurtosis': kurtosis, 'normal_test': normaltest}

    results = metric_map[metric](returns)
    if metric == 'normal_test':
        # obtaining p-values
        results = results[1]
    results = pd.Series(results, index=returns.columns)
    return results


def population_value(returns: pd.DataFrame, metric: str = 'kurtosis') -> pd.Series:
    '''
    Return the requested metric using population std

    Parameters:
    metric: either skewness or kurtosis
    '''

    if metric == 'skewness':
        a = 3
    elif metric == 'kurtosis':
        a = 4
    else:
        print(f'{population_value.__name__}: {metric} not found')

    centered_returns = returns - returns.mean()
    pob_std = returns.std(ddof=0)
    power_mean = (centered_returns**a).mean()
    pob_est = power_mean/pob_std**a
    return pob_est


def performance_metrics(returns: pd.DataFrame, metrics: list[(Callable, dict[str:str])] = [(annualized_returns, {'p': 252}),
                                                                                           (annualized_volatility, {
                                                                                               'p': 252}),
                                                                                           (ratio_metric, {
                                                                                               'metric': 'sharpe_ratio', 'target': 0.05}),
                                                                                           (max_drawdown, {})],
                        round_to=5, display_table: bool = True, return_table: bool = True) -> pd.DataFrame | None:
    '''
    Computes the requested performance metrics for a set of returns

    Parameters:
    metrics: a list with the tuple(function, parameters), where parameters is a dictionary
      where its key:value pairs are the parameters name and its desried values
    round_to: round performance metric to this amount of decimals
    display_table: if you want to display the performance metrics table
    return_table: if you want yo return the performance metrics table

    '''
    results = list()
    names = []
    for m in metrics:
        func = m[0]
        name = func.__name__
        parameters = func.__annotations__

        inserted_param_names = m[1]
        if name in ['scipy_metric', 'ratio_metric', 'population_value']:
            names.append(inserted_param_names['metric'])
        else:
            names.append(name)

        ordered_values = []
        if len(inserted_param_names) > 0:
            for i in inserted_param_names:
                for j in parameters:
                    if i == j:
                        ordered_values.append(inserted_param_names[i])
            # unpack needs python version 3.11 or greater
            results.append(func(returns, *ordered_values))

        else:
            results.append(func(returns))
    results = pd.DataFrame(results, index=names)
    results = results.round(round_to)
    results = results.transpose()
    if display_table:
        display(results)
    if return_table:
        return results


def data_results(etf_data: pd.DataFrame, start: str, end: str, freq: str = '1d',
                 etf_type: list = ['Regional', 'Sectorial', 'Factor', 'Bonds',
                                   'Alternative', 'Market', 'General', 'Commodities',
                                   'Currency', 'ESG', 'High Dividend'],
                 metrics: list[Callable, dict[str:str]] = [(annualized_returns, {'p': 252}),
                                                           (annualized_volatility, {
                                                               'p': 252}),
                                                           (ratio_metric, {
                                                               'metric': 'sharpe_ratio', 'target': 0.05}),
                                                           (max_drawdown, {})],
                 round_to: int = 5,
                 display_table: bool = False) -> pd.DataFrame | None:
    '''
    Return a report and returns of the requested assets

    Parameters:
    etf_data: a dataframe with the tickers data and additional information
      that wants to be shown in the report
    etf_type = ['Regional','Sectorial','Factor','Bonds','Alternative','Market',
      'General','Commodities','Currency','ESG','High Dividend']
    metrics: a list with the tuple(function, parameters), where parameters is a dictionary
      where its key:value pairs are the parameters name and its desried values
    round_to: round performance metric to this amount of decimals
    display_table: if you want to display the performance metrics table
    '''

    # Iterar por grupos (mutuamente excluyentes)
    etf_data[etf_data.type.isin(etf_type)]
    tickers = etf_data.symbol.tolist()

    df = data_yahoo(tickers, freq, start, end)
    if not isinstance(df, pd.DataFrame):
        return None

    df = fix_na_data_yahoo(df)

    # Convertir a retornos
    returns = df.pct_change().dropna()
    # Sacar descriptivas
    summary = performance_metrics(
        returns=returns, metrics=metrics, round_to=round_to, display_table=display_table)

    # Hacer el subset de etf_data
    etf_data = etf_data[etf_data['symbol'].isin(list(summary.index))]
    # Fijar como index la columna SYMBOL
    etf_data = etf_data.set_index('symbol')
    # Unir descriptivas con el subset de etf_data por index (SYMBOL)
    summary = pd.concat([etf_data, summary], axis=1)

    # Exportar a excel
    summary.to_excel(f'reports/report_{freq}_{start}-{end}.xlsx')
    returns.to_excel(f'reports/returns_{freq}_{start}-{end}.xlsx')
    return summary, returns


def cluster_assignation(corr_matrix: pd.DataFrame, threshold_scaler: float = 0.5, metric: str = 'euclidean', method: str = 'complete') -> np.array:
    """
    Return the hierachical cluster index for each stock given its returns DataFrame

    Parameters:
    corr_matrix: the correlation matrix of the asset returns
    threshold_scaler: scales the distance maximum used as a threshold for identifying the clusters
    metric: the distance metric
    method: the linkage criterion
    """

    distance = sch.distance.pdist(corr_matrix, metric=metric)
    linkage = sch.linkage(distance, method=method)
    limit = threshold_scaler * distance.max()
    idx = sch.fcluster(linkage, limit, 'distance')

    return idx


def cluster_corr(returns: pd.DataFrame, threshold_scaler: float = 0.5, metric: str = 'euclidean', method: str = 'complete') -> pd.DataFrame:
    """
    Return a correlation matrix sorted following hierarchical clusters

    Parameters:
    threshold_scaler: scales the distance maximum used as a threshold for identifying the clusters
    metric: the distance metric
    method: the linkage criterion
    """
    corr = returns.corr()
    idx = cluster_assignation(corr, threshold_scaler, metric, method)
    idx = np.argsort(idx)
    clust_corr = corr.iloc[idx, idx]

    return clust_corr


def compare_data(tickers: list, start: str, end: str,
                 freq: str = '1d',
                 metrics: list[Callable, dict[str:str]] = [(annualized_returns, {'p': 252}),
                                                           (annualized_volatility, {
                                                               'p': 252}),
                                                           (ratio_metric, {
                                                               'metric': 'sharpe_ratio', 'target': 0.05}),
                                                           (max_drawdown, {})],
                 round_to: int = 2,
                 sort_by: str | None = None,
                 clusters: bool = True,
                 threshold_scaler: float = 0.5,
                 metric: str = 'euclidean',
                 method: str = 'complete',
                 rolling_window: int = 30) -> pd.DataFrame:
    '''
    Attemps to retrieve the requested data from yfinance and gives a brief
      summary of the assets performance

    Parameters:
    tickers: a list with the tickers names
    start: format 'yyyy-mm-dd' 
    end: format 'yyyy-mm-dd' 
    freq: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    metrics: a list with the tuple(function, parameters), where parameters is a dictionary
      where its key:value pairs are the parameters name and its desried values
    round_to: round performance metric to this amount of decimals
    sort_by: name of the metric for which to sort the performance metrics
    clusters: if True, the correlation matrix will be sorted with hierarchical clusters
    threshold_scaler: scales the distance maximum used as a threshold for identifying the clusters
    metric: the distance metric
    method: the linkage criterion  
    rolling_window: size for the rolling window, affects variance estimation
    '''

    df = data_yahoo(tickers, freq, start, end)
    df.dropna(axis=1, inplace=True)
    rem_tick = set(tickers).difference(set(df.columns))
    if len(rem_tick) > 0:
        print(f'{len(rem_tick)} tickers omitted: {rem_tick}')

    df = normalize_data(df, False, 1, True, True)
    returns = df.pct_change().dropna()

    performance = performance_metrics(
        returns, metrics, round_to, display_table=False)
    if sort_by:
        display(performance.sort_values([sort_by], ascending=False))
    else:
        display(performance)

    if clusters:

        corr = cluster_corr(
            returns=returns, threshold_scaler=threshold_scaler, metric=metric, method=method)
    else:
        corr = returns.corr()
    display(corr.style.format(precision=2).background_gradient(cmap='coolwarm'))

    rolling_std = returns.rolling(rolling_window).std().dropna()
    rolling_variance = rolling_std**2

    config_subplot()

    fig, (ax1) = plt.subplots(1, 1, figsize=(25, 15))
    ax1.plot(rolling_variance)
    ax1.set_title('Rolling Variance', fontsize='xx-large')
    ax1.legend(rolling_variance.columns,
               loc="upper left", fontsize='xx-large')

    config_plt()

    return returns


TARGET = 0.05
PERCENTAGE = 0.05

ALL_ABS_METRICS = [(total_return, {}), (annualized_returns, {}), (annualized_volatility, {}),
                   (ratio_metric, {'metric': 'sharpe_ratio', 'target': TARGET}
                    ), (scipy_metric, {'metric': 'skewness'}),
                   (scipy_metric, {'metric': 'kurtosis'}), (scipy_metric, {'metric': 'normal_test'}), (
    value_at_risk, {'percentage': PERCENTAGE, 'estimation_type': 'cornish_fisher'}),
    (ratio_metric, {'metric': 'c_f_var_ratio',
                    'target': TARGET, 'percentage': PERCENTAGE}),
    (value_at_risk, {'percentage': PERCENTAGE,
                     'estimation_type': 'sample'}),
    (ratio_metric, {'metric': 'sample_var_ratio',
                    'target': TARGET, 'percentage': PERCENTAGE}),
    (conditional_value_at_risk, {'percentage': PERCENTAGE}),
    (ratio_metric, {'metric': 'cvar_ratio',
                    'target': TARGET, 'percentage': PERCENTAGE}),
    (max_drawdown, {}), (ratio_metric, {
        'metric': 'calmar_ratio', 'target': TARGET}),
    (average_drawdown, {}), (ratio_metric, {
        'metric': 'sterling_ratio', 'target': TARGET}),
    (burke_drawdown_measure, {}), (ratio_metric, {
        'metric': 'burke_ratio', 'target': TARGET}),
    (modified_burke_drawdown_measure, {}), (ratio_metric, {
        'metric': 'modified_burke_ratio', 'target': TARGET}),
    (target_downside_deviation, {'target': TARGET}),
    (ratio_metric, {'metric': 'sortino_ratio', 'target': TARGET}),
    (kappa_risk_measure, {'target': TARGET}),
    (ratio_metric, {
        'metric': 'kappa_ratio', 'target': TARGET}),
    (gain_loss_ratio, {'target': TARGET}),
    (upside_potential_ratio, {'target': TARGET}),
    (omega_ratio_sum_approx, {'target': TARGET}),
    (omega_ratio_put_option, {'target': TARGET}),
    (sortino_ratio, {'target': TARGET}),
    (kappa_ratio, {'target': TARGET}),
    (sharpe_ratio_on_avg, {'target': TARGET})
]

BENCHMARK = 'SPY'
ALL_REL_METRICS = [(annualized_excess_returns, {'benchmark': BENCHMARK}),
                   (annualized_tracking_error, {'benchmark': BENCHMARK}),
                   (information_ratio, {'benchmark': BENCHMARK}),
                   (beta_estimate,  {'benchmark': BENCHMARK}),
                   (treynor_ratio, {
                       'benchmark': BENCHMARK,  'target': TARGET}),
                   (m2_measure, {'benchmark': BENCHMARK, 'target': TARGET}),
                   ]
