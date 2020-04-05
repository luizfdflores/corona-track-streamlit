import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from getdata import getCountryData, getCountryList, regressions, logistic, exponential
from time import gmtime, strftime
import plotly.express as px
import plotly.graph_objects as go

countryDict = getCountryList()

@st.cache
def load_data(countryCode):

    df = getCountryData(countryCode)
    #combined_csv['Last Update'] =  pd.to_datetime(combined_csv['Last Update'])
    return df

st.title('Corona Virus')

country = st.sidebar.selectbox("Enter country", list(countryDict.keys()),)
st.write("Selected country: ", country)

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(countryDict[country])
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

#mask_countries = data['Country/Region'].isin(countries)

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Simple Analysis')
#st.line_chart(data)

mostrecentdate = data['date'].max()
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
    if current > lastweek:
        st.write('** Based on Most Recent Week of Data **')
        st.write('\tConfirmed cases on', co.index[-1], '\t', current)
        st.write('\tConfirmed cases on', co.index[-8], '\t', lastweek)
        ratio = current / lastweek
        st.write('\tRatio:', round(ratio, 2))
        st.write('\tWeekly increase:', round(100 * (ratio - 1), 1), '%')
        dailypercentchange = round(100 * (pow(ratio, 1 / 7) - 1), 1)
        st.write('\tDaily increase:', dailypercentchange, '% per day')
        recentdbltime = round(7 * np.log(2) / np.log(ratio), 1)
        st.write('\tDoubling Time (represents recent growth):', recentdbltime, 'days')

st.subheader('Chart')

logisticr2, expr2, ldoubletime, ldoubletimeerror, edoubletime, edoubletimeerror, epopt, lpopt = regressions(x, y)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Original Data', marker=dict(color='#000000')))

if logisticr2 > 0:
    fig.add_trace(go.Scatter(x=x, y=logistic(x, *lpopt), mode='lines', name='Logistic Curve Fit',
                                         line = dict(color='royalblue', width=2, dash='dot')))

if expr2 > 0:
    fig.add_trace(go.Scatter(x=x, y=exponential(x, *epopt), mode='lines', name='Exponential Curve Fit',
                             line=dict(color='firebrick', width=2, dash='dash')))

fig.update_layout(title= country + ' Cumulative COVID-19 Cases. (Updated on '+mostrecentdate+')',
                      xaxis_title='Days',
                      yaxis_title='Total Cases',
                      font=dict(size=14))

st.plotly_chart(fig, use_container_width=True)

if expr2 >0:
    st.write('\n** Based on Exponential Fit **\n')
    st.write('\tR^2:', expr2)
    st.write('\tDoubling Time (represents overall growth): ', round(edoubletime, 2), '(±', round(edoubletimeerror, 2),') days')

if logisticr2 >0:
    st.write('\n** Based on Logistic Fit**\n')
    st.write('\tR^2:', logisticr2)
    st.write('\tDoubling Time (during middle of growth): ', round(ldoubletime,2), '(±', round(ldoubletimeerror,2),') days')