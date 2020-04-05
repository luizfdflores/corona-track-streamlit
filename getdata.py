import requests
import pandas as pd
import numpy as np
import json
from pandas import json_normalize
from scipy.optimize import curve_fit
import plotly.express as px
import plotly.graph_objects as go
from time import gmtime, strftime


base_url = 'http://corona-api.com/countries'


def getCountryList():
    response = requests.get(base_url).json()
    countryListCode = {}
    for country in response['data']:
        countryListCode[country['name']] = country['code']
    return countryListCode


def getCountryData(countryCode):
    url = f'{base_url}/{countryCode}'
    response = requests.get(url).json()
    df = json_normalize(response['data'])
    countryName = df['name'].values[0]
    df_timeline = pd.concat(json_normalize(data) for data in df['timeline'].values.tolist()[0])
    df_timeline['name'] = countryName
    df_full = df.merge(df_timeline, on=['name']).drop('timeline', axis=1)
    return df_full


def getAllData(countryCodeList):
    for i, countryCode in enumerate(countryCodeList):
        if i == 0:
            df = getCountryData(countryCode)
        else:
            try:
                df = pd.concat([df, getCountryData(countryCode)])
            except:
                print(f'Not able to get data to {countryCode}')
    return df


def logistic(t, a, b, c, d):
    return c + (d - c)/(1 + a * np.exp(- b * t))


def exponential(t, a, b, c):
    return a * np.exp(b * t) + c


def regressions(x, y):
    logisticworked = False
    exponentialworked = False

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
        else:
            return logisticr2, expr2, ldoubletime, ldoubletimeerror, edoubletime, edoubletimeerror, epopt, lpopt

    if logisticworked:
        return logisticr2, 0, ldoubletime, ldoubletimeerror, 0, 0, epopt, lpopt

    if exponentialworked:
        return 0, expr2, 0, 0, edoubletime, edoubletimeerror, epopt, lpopt

    else:
        return 0, 0, float('NaN'), float('NaN'), float('NaN'), float('NaN'), epopt, lpopt