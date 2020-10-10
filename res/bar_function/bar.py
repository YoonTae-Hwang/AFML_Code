#import python scientific
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from numba import jit
import math
import pymc3 as pm
from theano import shared, theano as tt
import tqdm
# import visual tools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import missingno as msno #결측치


############### 1. Tick Bars ####################
def tick_bars(df, price_col, n):
    '''
    :param df: pd.Dataframe := Tick data set
    :param price_col: str := price columns
    :param n: int := number of pre determined Transactions

    :return: index
    '''

    t = df[price_col]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += 1
        if ts >= n:
            idx.append(i)
            ts = 0
            continue
    return idx
################################################


############### 2. Volume Bars ####################
def volume_bars(df, volum_col, n):
    '''
    :param df: pd.Dataframe := Tick data set
    :param volum_col: str := volum columns
    :param n: int := number of pre determined Transactions

    :return: index
    '''

    t = df[volum_col]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += x
        if ts >= n:
            idx.append(i)
            ts = 0
            continue
    return idx
################################################


############### 3. dollar Bars ####################
def dollar_bars(df, dollar_col, n):
    '''
    :param df: pd.Dataframe := Tick data set
    :param dollar_col: str := dollar columns
    :param n: int := number of pre determined Transactions

    :return: index
    '''

    t = df[dollar_col]
    ts = 0
    idx = []
    for i, x in enumerate(tqdm(t)):
        ts += x
        if ts >= n:
            idx.append(i)
            ts = 0
            continue
    return idx
################################################


############### 4. Select data set ####################
def select_sample_data(pre, after, price_col, date):
    '''

    :param pre: pd.Dataframe := origin tick data set
    :param after: pd.Dataframe := bar - data set
    :param price_col: str := price columns
    :param date: str := "yyyy-mm-dd", select date
    :return: x_df, xt_df := full_data & bar data
    '''
    x_df = pre[price_col].loc[date]
    xt_df = after[price_col].loc[date]
    return x_df, xt_df
################################################

############### 5. plot data ####################

def plot_sample_data(ref, sub, bar_type, *args, **kwds):
    '''

    :param ref: pd.Dataframe := x_df
    :param sub: pd.Dataframe := xt_df
    :param bar_type: str := Lebel {tick, vol, dollar}
    :param args: None
    :param kwds: None
    :return: price + bar plot, price plot, bar plot
    '''
    f, axes = plt.subplots(3, sharex=True, sharey=True, figsize=(10, 7))
    ref.plot(*args, **kwds, ax=axes[0], label='price')
    sub.plot(*args, **kwds, ax=axes[0], marker='X', ls='', label=bar_type)
    axes[0].legend();

    ref.plot(*args, **kwds, ax=axes[1], label='price', marker='o')
    sub.plot(*args, **kwds, ax=axes[2], ls='', marker='X',
             color='r', label=bar_type)

    for ax in axes[1:]: ax.legend()
    plt.tight_layout()

    return

################################################

############### 6. Tick Imbalance bar ####################
#Using jit
@jit(nopython = True)
def numba_isclose(a, b,
                  rel_tol : float = 1e-09,
                  abs_tol : float = 0.0):
    """
    :param a: price at time t
    :param b: price at time t-1
    :param rel_tol: rel_tol
    :param abs_tol: abs_tol
    :return: Bool 
    """
    return np.fabs(a-b) <= np.fmax(rel_tol * np.fmax(np.fabs(a), np.fabs(b)),
                                   abs_tol)

@jit(nopython = True)
def bt(p0, p1, b0):
    """
    bt is tick rule defines a sequence b_{t} 
    :param p0: price associated with tick time t-1
    :param p1: price associated with tick time t
    :param b0: tick rule defines a sequence b_{t}..
    :return: b := tick rule
    """
    
    if numba_isclose((p1-p0), 0.0, abs_tol= 0.001):
        b = b0 #same price p1 = p0
        return b
    else:
        b = np.abs(p1 - p0) / (p1 - p0) #p1 != p0
        return b
    
@jit(nopython= True)
def get_imbalance(t):
    """
    :param t: price data
    :return: imbalance dataset *remove last value
    """
    
    bs = np.zeros_like(t)
    for i in np.arange(1, bs.shape[0]):
        t_bt = bt(t[i-1], t[i], bs[i-1])
        bs[i-1] = t_bt
    return bs[:-1] 
    
@jit(nopython= True)
def test_t_abs(abs_Theta, t, E_bs):
    """
    (abs_Theta >= t * E_bs)
    :param abs_Theta: tick imbalance at time T 
    :param t:  time T
    :param E_bs:  expected value of theta T
    :return: bool
    """
    return (abs_Theta >= t * E_bs)

def agg_imbalance_bars_(df):
    """
    수정중
    :param df: 
    :return: 
    """
    
    return None
################################################
