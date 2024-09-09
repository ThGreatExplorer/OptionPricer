import numpy as np
from scipy.stats import norm

def calculate_BS_OPM(curr_price :float, strike_price :float, time :float, vol :float, rf :float, call :bool) -> float:
    """Black Scholars Option Pricing Calculation for European Option.

    Args:
        curr_price (float): current price of stock
        strike_price (float): the strike price of the option
        time (float): time to maturity
        vol (float): volatility
        rf (float): risk free interest rate
        call (bool): where the stock is a call or put option

    Returns:
        float: the calculated price
    """
    d_1 = (np.log(curr_price / strike_price) + (rf + .5 * vol**2) * time)/ (vol * np.sqrt(time))
    d_2 = d_1 - vol * np.sqrt(time)
    if call:
        price = curr_price * norm.cdf(d_1) - strike_price * np.exp(-rf * time) * norm.cdf(d_2)
    else:
        price = strike_price * np.exp(-rf * time) * norm.cdf(-d_2) - curr_price * norm.cdf(-d_1)
    return price
