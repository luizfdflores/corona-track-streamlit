import requests
import pandas as pd
import numpy as np
from pandas import json_normalize
from scipy.optimize import curve_fit
from time import gmtime, strftime
import streamlit as st


base_url = 'http://corona-api.com/countries'


def getcountrylist():
    response = requests.get(base_url).json()
    countrylistcode = {}
    for country in response['data']:
        countrylistcode[country['name']] = country['code']
    return countrylistcode


def getcountrydata(countrycode):
    url = f'{base_url}/{countrycode}'
    response = requests.get(url).json()
    df = json_normalize(response['data'])
    countryname = df['name'].values[0]
    df_timeline = pd.concat(json_normalize(data) for data in df['timeline'].values.tolist()[0])
    df_timeline['name'] = countryname
    df_full = df.merge(df_timeline, on=['name']).drop('timeline', axis=1)
    return df_full


def getalldata(countrylistcode):
    for i, countryName in enumerate(list(countrylistcode.keys())):
        try:
            df_data = getcountrydata(countrylistcode[countryName])
            _, _, df = simplestatistics(df_data, False)
            df['country'] = countryName
            df['population'] = df_data['population']
            if i == 0:
                df_static = df
            else:
                df_static = pd.concat([df_static, df])
        except:
            pass
    return df_static


def logistic(t, a, b, c, d):
    return c + (d - c)/(1 + a * np.exp(- b * t))


def exponential(t, a, b, c):
    return a * np.exp(b * t) + c


def simplestatistics(data, print_stats=True):

    currdate = strftime("%Y-%m-%d", gmtime())
    data = data[data['date'] != currdate]
    co = pd.DataFrame(data.sort_values(by=['date']).set_index('date')['confirmed'])
    co.columns = ['Cases']
    co = co.loc[co['Cases'] > 0]
    recentdbltime = float('NaN')
    y = np.array(co['Cases'])
    x = np.arange(y.size)
    if len(y) >= 7:
        current = y[-1]
        lastweek = y[-8]
        ratio = current / lastweek
        recentdbltime = round(7 * np.log(2) / np.log(ratio), 1)
        dailypercentchange = round(100 * (pow(ratio, 1 / 7) - 1), 1)
        if print_stats:
            st.write('** Based on Most Recent Week of Data **')
            st.write('\tConfirmed cases on', co.index[-1], '\t', current)
            st.write('\tConfirmed cases on', co.index[-8], '\t', lastweek)
            st.write('\tRatio:', round(ratio, 2))
            st.write('\tWeekly increase:', round(100 * (ratio - 1), 1), '%')
            st.write('\tDaily increase:', dailypercentchange, '% per day')
            st.write('\tDoubling Time (represents recent growth):', recentdbltime, 'days')
            return x, y, None
        else:
            params = pd.DataFrame([[current, lastweek, ratio, recentdbltime, dailypercentchange]],
                                  columns=['current', 'lastweek', 'ratio', 'recentdbltime', 'dailypercentchange'])
            return x, y, params


def regressions(x, y):
    logisticworked = False
    exponentialworked = False
    logisticr2 = 0
    expr2 = 0
    ldoubletime = 0
    ldoubletimeerror = 0
    edoubletime = 0
    edoubletimeerror = 0
    epopt = 0
    lpopt = 0

    try:
        lpopt, lpcov = curve_fit(logistic, x, y, maxfev=10000)
        lerror = np.sqrt(np.diag(lpcov))

        # for logistic curve at half maximum, slope = growth rate/2. so doubling time = ln(2) / (growth rate/2)
        ldoubletime = np.log(2) / (lpopt[1] / 2)
        # standard error
        ldoubletimeerror = 1.96 * ldoubletime * np.abs(lerror[1] / lpopt[1])

        # calculate R^2
        residuals = y - logistic(x, *lpopt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        logisticr2 = 1 - (ss_res / ss_tot)

        if logisticr2 > 0.95:
            logisticworked = True

    except:
        pass

    try:
        epopt, epcov = curve_fit(exponential, x, y, bounds=([0, 0, -100], [100, 0.9, 100]), maxfev=10000)
        eerror = np.sqrt(np.diag(epcov))

        # for exponential curve, slope = growth rate. so doubling time = ln(2) / growth rate
        edoubletime = np.log(2) / epopt[1]
        # standard error
        edoubletimeerror = 1.96 * edoubletime * np.abs(eerror[1] / epopt[1])

        # calculate R^2
        residuals = y - exponential(x, *epopt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        expr2 = 1 - (ss_res / ss_tot)

        if expr2 > 0.95:
            exponentialworked = True

    except:
        pass

    if round(logisticr2, 2) > round(expr2, 2):
        return logisticr2, expr2, ldoubletime, ldoubletimeerror, edoubletime, edoubletimeerror, epopt, lpopt
    elif logisticworked:
        return logisticr2, 0, ldoubletime, ldoubletimeerror, 0, 0, epopt, lpopt
    elif exponentialworked:
        return 0, expr2, 0, 0, edoubletime, edoubletimeerror, epopt, lpopt
    else:
        return 0, 0, float('NaN'), float('NaN'), float('NaN'), float('NaN'), epopt, lpopt
