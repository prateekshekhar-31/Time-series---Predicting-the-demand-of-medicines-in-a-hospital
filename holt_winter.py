import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
#Importing data
df = pd.read_csv('Type-A-Medicine-3-Yrs-_2015-17_.csv')
#Printing head
df_2017 = df[df.Year == '2017']
df_2016 = df[df.Year == '2016']
df_2015 = df[df['Year'] == '2015']
df= df.drop([36])
df = df.drop([37])
df['Datetime'] = pd.to_datetime([f'{y}-{m}-01' for y , m in zip(df.Year,df.Month)])
df.head()

x = range(1,13)
plt.figure(figsize=(20,10))
plt.plot(df['Datetime'],df['Monthly Consumption of Type A Medicine'],label="2017")
#plt.plot(x,df_2016['Monthly Consumption of Type A Medicine'],label="2016")
#plt.plot(x,df_2015['Monthly Consumption of Type A Medicine'],label="2015")
#plt.legend()
plt.show()

train = df[:24]
test = df[24:35]

plt.plot(df['Monthly Consumption of Type A Medicine'],label="df")
plt.plot(train['Monthly Consumption of Type A Medicine'],label="train")
plt.plot(test['Monthly Consumption of Type A Medicine'],label="test")
plt.legend()
plt.show()

y_hat_avg = test.copy()
fit1 = ExponentialSmoothing(np.asarray(train['Monthly Consumption of Type A Medicine']) ,seasonal_periods=12 ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot( train['Monthly Consumption of Type A Medicine'], label='Train')
plt.plot(test['Monthly Consumption of Type A Medicine'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test['Monthly Consumption of Type A Medicine'], y_hat_avg.Holt_Winter))
print(rms)
p = range(35-45)
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.show()
pred = fit1.predict(start = 0, end = 40)
plt.plot(pred, label = "pred")
plt.plot(df['Monthly Consumption of Type A Medicine'], label="df")
plt.legend()
plt.show()

#ExponentialSmoothing.predict(np.asarray(train['Monthly Consumption of Type A Medicine']))

