
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def markowitz_opt(mu: np.array, sig: pd.DataFrame, w_init: np.array, gamma: float = 0.8, w_bounds: tuple[float] = (0.0, 1.0)) -> list[float]:

    def target(w):
        return 0.5*np.dot(w, np.dot(sig, w)) - gamma * np.dot(mu, w)

    # sum(w) = 1
    const = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    w = minimize(target, w_init, constraints=const,
                 bounds=w_bounds, tol=1e-11, method='SLSQP', )

    return w.x
