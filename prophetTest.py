# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:39:06 2017

@author: j.klen
"""

# works with python 3.5 and with this Prophet installation instructions
# https://facebook.github.io/prophet/docs/installation.html

import pandas as pd
import numpy as np
from fbprophet import Prophet

# csv input should have 2 columns: "ds" - date and "y" - values
# works also automatically with missing values

df = pd.read_csv('daily-minimum-temperatures-in-meWithNA.csv')

m = Prophet(interval_width = 0.95)
m.fit(df)

future = m.make_future_dataframe(periods = 120)

forecast = m.predict(future)

m.plot(forecast)
m.plot_components(forecast)

df['yhat_lower'] = np.NaN
df['yhat_upper'] = np.NaN
df['actual'] = 'Y'


forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast['actual'] = 'N'
forecast.columns = ['ds', 'y', 'yhat_lower', 'yhat_upper', 'actual']
forecast = forecast[df.shape[0]:]

result = pd.concat((df, forecast))
result['ds'] = pd.DatetimeIndex(result.ds).normalize()
result # pandas dataframe with columns: df - date, y - values, yhat_lower - lower confidence interval bound,
# yhat_upper - upper confidence interval bound, actual - Y/actual N/forecasted

# result.values # if its better to have result as numpy array
