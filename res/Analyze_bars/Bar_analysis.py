#import Standard lib
from pathlib import PurePath, Path
import sys
import time
from collections import OrderedDict as od
import re
import os
import json

#import python scientific
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from numba import jit
import math
import pymc3 as pm
from theano import shared, theano as tt


# import visual tools
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import missingno as msno #결측치

def count_bars_1W(df, price_col, scale = True):
    '''
    :param df: pd.Dataframe := origin tick data
    :param price_col: str := price col
    :param scale : Bool := using min-max scale
    :return: Grouper 1W price count dataset
    '''
    if scale == True:
        data = df.groupby(pd.Grouper(freq = "1W"))[price_col].count()
        return (data - data.min()) / (data.max() - data.min())
    else:
        return df.groupby(pd.Grouper(freq = "1W"))[price_col].count()

def count_bars_1M(df, price_col, scale = True):
    '''
    Count quantity of bars by each bar type

    :param df: pd.Dataframe := origin tick data
    :param price_col: str := price col
    :param scale : Bool := using min-max scale
    :return: Grouper 1W price count dataset
    '''
    if scale == True:
        data = df.groupby(pd.Grouper(freq = "1M"))[price_col].count()
        return (data - data.min()) / (data.max() - data.min())
    else:
        return df.groupby(pd.Grouper(freq = "1M"))[price_col].count()

def returns(s):
    '''
    :param s: array := price data set
    :return: pd.Series array
    '''
    array_data = np.diff(np.log(s))
    return pd.Series(array_data, index= array_data.index[1:])

def get_test(bar_type, bar_returns, test_fun, *args, **kwds):

    '''
    진행중....
    :param bar_type:
    :param bar_returns:
    :param test_fun:
    :param args:
    :param kwds:
    :return:
    '''

    return None