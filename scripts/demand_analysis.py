#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ---------------------------
# Data Manipulation Libraries
# ---------------------------
import pandas as pd        # Handling tabular data, DataFrames, and CSV/Excel I/O
import numpy as np         # Numerical operations, arrays, linear algebra

# ---------------------------
# Data Visualization Libraries
# ---------------------------
import matplotlib.pyplot as plt   # Core plotting library
import seaborn as sns             # High-level statistical visualization

# ---------------------------
# Utilities & System Helpers
# ---------------------------
import warnings               # Manage warning messages in the notebook
from datetime import datetime # Work with dates and timestamps
import holidays               # Access country-specific holiday calendars

# ---------------------------
# Model Evaluation Metrics
# ---------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
# Used for regression evaluation and custom scoring in ML

# ---------------------------
# Predictive Modeling Libraries
# ---------------------------
import xgboost as xgb                       # Gradient boosting framework
from xgboost import XGBRegressor            # XGBoost regressor model
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_validate
# Splitting datasets, time-series-aware cross-validation, and model evaluation
from feature_engine.timeseries.forecasting import LagFeatures
# Create lag-based features for time series forecasting

# ---------------------------
# Classical Time Series Forecasting
# ---------------------------
from statsforecast import StatsForecast     # Unified interface for time series models
from statsforecast.models import AutoARIMA  # Automatic ARIMA modeling

# ---------------------------
# Prophet Time Series Forecasting
# ---------------------------
from prophet import Prophet                                 # Additive regression model for forecasting
from prophet.diagnostics import performance_metrics         # Evaluate model performance
from prophet.diagnostics import cross_validation            # Time-based cross-validation


warnings.filterwarnings("ignore")
color_pal=sns.color_palette()


# In[2]:


df = pd.read_csv('demand_df.csv')
df['Local time'] = pd.to_datetime(df['Local time'])
df.set_index('Local time', inplace=True)
df.sort_index(inplace=True, ascending=True)
df = df.drop_duplicates()
print(df.index.duplicated().sum())


# In[3]:


az_population = pd.read_excel('Arizona_Population.xlsx')
az_population = az_population[~az_population['Year'].isin([2000, 2001, 2002, 
                                                            2003, 2004, 2005,
                                                            2006, 2007, 2008,
                                                            2009, 2010, 2011,
                                                            2012, 2013, 2014,
                                                            2015, 2025])]


# In[4]:


# Extract as Series
actual = df['Demand (MWh)']
predicted = df['Demand Forecast (MWh)']

fig, ax = plt.subplots(figsize=(15,5))
df['Demand (MWh)'].plot(ax=ax)
df['Demand Forecast (MWh)'].plot(ax=ax, style='.')
plt.legend(['Real Data', 'Predictions'])
ax.set_title('Electricity Demand: Actual vs Predictions')
ax.set_ylabel('Demand (MWh)')
ax.set_xlabel('Date')
plt.show()

# Metrics
mse_value = mean_squared_error(actual, predicted)
rmse_value = np.sqrt(mse_value)
mae = mean_absolute_error(actual, predicted)
mape = (abs(actual - predicted) / actual).mean() * 100
bias = (predicted - actual).mean()
r2 = r2_score(actual, predicted) * 100
demand_mean = actual.mean()
percentage_rmse = (rmse_value / demand_mean) * 100

# Print
print(f"MSE: {mse_value:.2f}")
print(f"RMSE: {rmse_value:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Bias: {bias:.2f}")
print(f"Mean: {demand_mean:.2f}")
print(f"RMSE % of Mean: {percentage_rmse:.2f}%")
print(f"R2 Score: {r2:.2f}%")


# In[5]:


def create_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df


# In[6]:


df = create_features(df) 


# In[7]:


# This converts all month numbers to a specific season
conditions = [
    df['month'].isin([12, 1, 2]),
    df['month'].isin([3, 4, 5]),
    df['month'].isin([6, 7, 8]),
    df['month'].isin([9, 10, 11])
]
choices = ['Winter', 'Spring', 'Summer', 'Fall']

# Apply mapping
df['Season'] = np.select(conditions, choices, default='Unknown')


# In[8]:


df = df.drop_duplicates()
print(df.index.duplicated().sum())
# df.to_csv('full_df.csv')


# In[9]:


# Commented out because I don't want duplicates being generated
#merged_df.to_csv("demand_df.csv")


# In[10]:


df_season = df.groupby('Season').agg('sum').sort_values(by='Demand (MWh)', ascending=False).round()
df_season = df_season[['Demand (MWh)', 'Net Generation (MWh)',	'Demand Forecast (MWh)']]
df_season


# In[11]:


sns.color_palette("flare", as_cmap=True)
ax = sns.barplot(
    data=df_season,
    x=df_season.index,
    y="Demand (MWh)",
    errorbar=None,
    palette="Spectral"
)
    

# Set values to millions and add commas
ax.ticklabel_format(style='plain', axis='y')
ax.get_yaxis().set_major_formatter(
    plt.FuncFormatter(lambda x, loc: f'{int(x):,}')
)

ax.set_title("Total seasonal demand of electricity (past 10 years)")
ax.set_ylabel("MWh")

plt.show()


# In[12]:


color_pal = sns.color_palette()
df.plot(style='.', y='Demand (MWh)', figsize=(15,5), color=color_pal[0], title='Demand for energy (MWh) from SRP customers...')


# In[13]:


df.loc[(df.index >='01-01-2024 00:00:00') & (df.index <= '12-31-2024 23:00:00')].plot(y='Demand (MWh)')


# In[14]:


subset = df.loc['2024-01-01 00:00:00':'2024-12-31 23:00:00']

ax = subset['Demand (MWh)'].plot(figsize=(12,6), label='Demand (MWh)', color='green')
subset['Temperature (F)'].plot(ax=ax, secondary_y=True, label='Temperature (F)', color='orange')
ax.set_title('Demand vs Temperature — Jan 1, 2024')
ax.legend(loc='upper left')
ax.right_ax.legend(loc='upper right')


# In[15]:


sns.scatterplot(data=df, x='Temperature (F)', y='Demand (MWh)', hue=df['month'])


# In[16]:


us_holidays = holidays.country_holidays('US')


# In[17]:


# 1. Holiday (1 = holiday, 0 = not holiday)
df['is_holiday'] = df.index.to_series().apply(lambda x: 1 if x.date() in us_holidays else 0)

# 2. Weekend flag (1 = Saturday/Sunday)
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

# 3. Daytime flag (1 = 6 AM–7 PM)
df['is_day'] = ((df.index.hour >= 6) & (df.index.hour < 20)).astype(int)


# In[18]:


df[df['is_holiday'] == 1]


# In[19]:


plt.figure(figsize=(4,3))
sns.heatmap(df[['Temperature (F)', 'Demand (MWh)']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Temperature and Demand')
plt.show()


# In[20]:


fig, ax = plt.subplots(2, 1, figsize=(10, 10))

sns.boxplot(data=df, x='hour', y='Demand (MWh)', ax=ax[0], palette='rocket')
ax[0].set_title('Electrical Demand by Hour')
ax[0].set_ylabel('Demand (MWh)')

sns.boxplot(data=df, x='hour', y='Temperature (F)', ax=ax[1], palette='rocket')
ax[1].set_title('Temperature by Hour')
ax[1].set_ylabel('Temperature (F)')
ax[1].set_xlabel('hour')

plt.tight_layout()
plt.show()


# In[21]:


fig, ax = plt.subplots(2, 1, figsize=(10, 10))

sns.boxplot(data=df, x='month', y='Demand (MWh)', ax=ax[0], palette='rocket')
ax[0].set_title('Electrical Demand by Month')
ax[0].set_ylabel('Demand (MWh)')
ax[0].set_xlabel('Month')

sns.boxplot(data=df, x='month', y='Temperature (F)', ax=ax[1], palette='rocket')
ax[1].set_title('Temperature by Month')
ax[1].set_ylabel('Temperature (F)')
ax[1].set_xlabel('Month')


# In[22]:


ax = sns.boxplot(data=df, x='Season', y='Demand (MWh)', palette="Spectral")
ax.set_title('Electrical Demand by Season')
ax.set_ylabel('Demand (MWh)')
ax.set_xlabel('')


# In[23]:


ax = sns.boxplot(data=df, x='Season', y='Temperature (F)', palette="Spectral")
ax.set_title('Phoenix Temperatures by Season')
ax.set_ylabel('Temperature (F)')
ax.set_xlabel('')


# In[24]:


df_year = df.groupby('year').agg('sum').sort_values(by='Demand (MWh)', ascending=False).round()
df_year = df_year[['Demand (MWh)']]
df_year = df_year.sort_index(ascending=False)
df_year = df_year.reset_index()

# Filtered out 2015 and 2025 since data isn't complete
df_year = df_year[~df_year['year'].isin([2015, 2025])]


# In[25]:


az_population = az_population.rename(columns={'Year': 'year'})

merged_df = pd.merge(az_population , df_year , on='year')

merged_df

# merged_df.to_csv("population_and_demand.csv")


# In[26]:


plt.figure(figsize=(12, 6))   # <-- ONLY place you put figsize

ax = sns.lineplot(data=df_year, x='year', y='Demand (MWh)')

# Format y-axis labels with commas
ax.ticklabel_format(style='plain', axis='y')
ax.get_yaxis().set_major_formatter(
    plt.FuncFormatter(lambda x, loc: f'{int(x):,}')
)

plt.tight_layout()
plt.show()


# In[27]:


plt.figure(figsize=(12, 6))

ax = sns.scatterplot(data=merged_df, x='Population', y='Demand (MWh)')

ax.ticklabel_format(style='plain', axis='y')
ax.get_yaxis().set_major_formatter(
    plt.FuncFormatter(lambda x, loc: f'{int(x):,}')
)
ax.get_xaxis().set_major_formatter(
    plt.FuncFormatter(lambda x, loc: f'{int(x):,}')
)

plt.show()


# In[28]:


plt.figure(figsize=(12, 6))

ax = sns.scatterplot(data=df, x='Temperature (F)', y='Demand (MWh)')

ax.ticklabel_format(style='plain', axis='y')
ax.get_yaxis().set_major_formatter(
    plt.FuncFormatter(lambda x, loc: f'{int(x):,}')
)
ax.get_xaxis().set_major_formatter(
    plt.FuncFormatter(lambda x, loc: f'{int(x):,}')
)

plt.show()


# In[29]:


df


# In[30]:


df.index.duplicated().sum()
df = df[~df.index.duplicated(keep='first')]


# In[31]:


lag_transformer = LagFeatures(
    variables=['Demand (MWh)'],
    periods=[1, 2, 3, 6, 24, 48, 168]
)

df_lagged = lag_transformer.fit_transform(df)
df = df_lagged.dropna()


# # Creating Predictive Model

# In[32]:


ml_df = df.copy()
ml_df = pd.get_dummies(ml_df, columns=['Season'], prefix='Season')

train = ml_df.loc[:'2023-12-31']
test = ml_df.loc['2024-01-01':]

fig, ax = plt.subplots(figsize=(15,5))

# Plot only Demand (MWh)
train['Demand (MWh)'].resample('D').mean().plot(ax=ax, label='Train')
test['Demand (MWh)'].resample('D').mean().plot(ax=ax, label='Test')

ax.set_title('Electricity Demand (MWh) Over Time')
ax.set_ylabel('Demand (MWh)')
ax.set_xlabel('Date')
ax.axvline(pd.Timestamp('2024-01-01'), color='red', linestyle='--', label='Train/Test Split')
ax.legend()

plt.show()


# In[33]:


ml_df.columns


# In[34]:


lag_cols = [col for col in df_lagged.columns if 'Demand (MWh)_lag' in col]


"""
    'Demand (MWh)_lag_1',
    'Demand (MWh)_lag_2',
    'Demand (MWh)_lag_3',
    'Demand (MWh)_lag_6',
    'Demand (MWh)_lag_48',
    'Demand (MWh)_lag_168'
    """
FEATURES = [
    'Temperature (F)',
    'hour',
    'dayofweek',
    'month',
    'year',
    'dayofyear',
    'is_day',
    'Demand (MWh)_lag_24',
    'Season_Spring',
    'Season_Winter',
]
"""
# Removing most features made bias worse, going to use all features above ^

FEATURES = [
    'Temperature (F)',
    'hour',
    'dayofweek',
    'month',
    'year',
    'dayofyear',
    'is_day'         
]

"""
TARGET = ['Demand (MWh)']


# In[35]:


lag_cols


# In[36]:


X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# In[37]:


reg = xgb.XGBRegressor(
    n_estimators=1000,
    early_stopping_rounds=50,
    random_state=42,
    learning_rate=0.01,
    max_depth = 3,
    min_child_weight=10,
    gamma=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=1
)

eval_set = [(X_train, y_train), (X_test, y_test)]

reg.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    verbose=100  # prints every 10 rounds; or use True/False
)


# In[38]:


test


# # Feature Importance

# In[39]:


cmap = sns.cubehelix_palette(
    start=2, rot=0, dark=0.75, light=.25,
    reverse=False, as_cmap=True
)


feature_importance_df = pd.DataFrame(
    data=reg.feature_importances_,
    index=reg.feature_names_in_,
    columns=['importance']
).sort_values(by='importance', ascending=False)

colors = [cmap(i / (len(feature_importance_df) - 1)) 
          for i in range(len(feature_importance_df))]

sns.barplot(
    data=feature_importance_df,
    x='importance',
    y=feature_importance_df.index,
    palette=colors
)


# In[40]:


xgb.plot_importance(reg, max_num_features=20, importance_type='gain', height=0.6)
plt.show()


# In[41]:


feature_importance_df


# # Forecast on Test

# In[42]:


test['ml_prediction'] = reg.predict(X_test)

ml_df = ml_df.merge(
    test[['ml_prediction']],  # ensure it's a DataFrame
    how='left',
    left_index=True,
    right_index=True
)


# In[43]:


predicted_df = ml_df.loc['2024-01-01':]
fig, ax = plt.subplots(figsize=(15,5))
predicted_df['Demand (MWh)'].plot(ax=ax)
predicted_df['ml_prediction'].plot(ax=ax, style='.')
plt.legend(['Real Data', 'Predictions'])
ax.set_title('Electricity Demand: Actual vs Predictions')
ax.set_ylabel('Demand (MWh)')
ax.set_xlabel('Date')
plt.show()


# In[44]:


import matplotlib.pyplot as plt

# Select only ml_prediction and Demand for the week
week_df = ml_df.loc['2025-10-25':'2025-10-31', ['Demand (MWh)', 'ml_prediction']]

# Plot
ax = week_df.plot(figsize=(15,5))
ax.set_title('Electricity Demand vs Predictions (October 25–31, 2024)')
ax.set_ylabel('Demand (MWh)')
ax.set_xlabel('Date')
plt.show()


# In[45]:


# ml_df.to_csv("xgbboost_reaL_versus_predicted.csv")


# # XGBRegressor:

# In[46]:


actual = test['Demand (MWh)']
predicted = test['ml_prediction']

mse_value = mean_squared_error(actual, predicted)
rmse_value = np.sqrt(mse_value)
mae = mean_absolute_error(actual, predicted)
mape = (abs(actual - predicted) / actual).mean() * 100
bias = (predicted - actual).mean()
r2 = r2_score(actual, predicted) * 100
demand_mean = actual.mean()
percentage_rmse = (rmse_value / demand_mean) * 100

print(f"MSE: {mse_value:.2f}")
print(f"RMSE: {rmse_value:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Bias: {bias:.2f}")
print(f"Mean: {demand_mean:.2f}")
print(f"RMSE % of Mean: {percentage_rmse:.2f}%")
print(f"R2 Score: {r2:.2f}%")


# # Baseline:

# In[47]:


# Extract as Series
actual = df['Demand (MWh)']
predicted = df['Demand Forecast (MWh)']

# Metrics
mse_value = mean_squared_error(actual, predicted)
rmse_value = np.sqrt(mse_value)
mae = mean_absolute_error(actual, predicted)
mape = (abs(actual - predicted) / actual).mean() * 100
bias = (predicted - actual).mean()
r2 = r2_score(actual, predicted) * 100
demand_mean = actual.mean()
percentage_rmse = (rmse_value / demand_mean) * 100

# Print
print("Baseline predictions:")
print(f"MSE: {mse_value:.2f}")
print(f"RMSE: {rmse_value:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Bias: {bias:.2f}")
print(f"Mean: {demand_mean:.2f}")
print(f"RMSE % of Mean: {percentage_rmse:.2f}%")
print(f"R2 Score: {r2:.2f}%")


# In[48]:


test['error'] = np.abs(test['Demand (MWh)'] - test['ml_prediction'])


# In[49]:


test['date'] = test.index.date


# In[50]:


test.groupby('date')['error'].mean().sort_values(ascending=True).head(5)


# In[51]:


test.groupby('date')['error'].mean().sort_values(ascending=True).tail(5)


# In[52]:


df_fc = df.reset_index().rename(columns={
    'Local time': 'ds',
    'Demand (MWh)': 'y'
})

df_fc['unique_id'] = 'demand'

df_fc = df_fc[['unique_id', 'ds', 'y']]


# In[53]:


df_train = df_fc.iloc[:-24*7]
df_test  = df_fc.iloc[-24*7:]

models = [
    AutoARIMA(seasonal=False, alias="ARIMA"),
]


# In[54]:


df_fc


# In[55]:


sf = StatsForecast(models=models, freq='H')
sf = sf.fit(df_train)

# Forecast 642 future hours
fh = 640
pred_df = sf.predict(fh)
print(pred_df)


# In[56]:


pred_df['DateTime'] = pd.to_datetime(pred_df['ds'])


# In[57]:


pred_df = pred_df.set_index('DateTime')


# In[58]:


pred_df.tail(5)


# In[59]:


df_fc = df_fc[['ds', 'y']]
df_fc


# In[60]:


m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True
)
m.fit(df_fc)


# In[61]:


future = m.make_future_dataframe(periods=365)
future.tail()


# In[62]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[63]:


fig1 = m.plot(forecast)


# In[64]:


fig2 = m.plot_components(forecast)


# In[65]:


# Commented this out because it takes a while, I instead saved the CSV and import it.
"""
df_cv = cross_validation(
    model=m,
    initial='365',   
    period='30 days',       
    horizon='90 days',
)

df_metrics.to_csv("prophet_metrics.csv")
"""



# In[66]:


df_cv = pd.read_csv("prophet_metrics.csv")


# In[67]:


df_cv


# In[68]:


future = m.make_future_dataframe(periods=216, freq='H')
future.tail()


# In[69]:


forecast = m.predict(future)
forecast.head()


# In[70]:


fig1 = m.plot(forecast)


# In[71]:


fig2 = m.plot_components(forecast)


# In[72]:


forecast = forecast[['ds', 'yhat']]


# In[73]:


forecast = forecast.rename(columns={'ds': 'Local time', 'yhat':'Predicted Demand (MWh)'})
forecast["Local time"] = pd.to_datetime(forecast['Local time'])
forecast = forecast.set_index("Local time")


# In[74]:


forecast


# In[75]:


real_df = df.copy()
real_df = real_df[["Demand (MWh)"]]
real_df


# In[76]:


forecast_loc = forecast.loc[:'2025-10-31 23:00:00']
forecast_loc


# In[77]:


merged_df = pd.merge(real_df, forecast_loc, on='Local time')
merged_df.to_csv("prophet_real_versus_predicted.csv")


# In[78]:


fig, ax = plt.subplots(figsize=(15,5))
merged_df['Demand (MWh)'].plot(ax=ax)
merged_df['Predicted Demand (MWh)'].plot(ax=ax, style='.')
plt.legend(['Real Data', 'Predictions'])
ax.set_title('Electricity Demand: Actual vs Predictions')
ax.set_ylabel('Demand (MWh)')
ax.set_xlabel('Date')
plt.show()


# In[79]:


week_df = merged_df.loc['2025-10-25':'2025-10-31', ['Demand (MWh)', 'Predicted Demand (MWh)']]

# Plot
ax = week_df.plot(figsize=(15,5))
ax.set_title('Electricity Demand vs Predictions (October 25–31, 2025)')
ax.set_ylabel('Demand (MWh)')
ax.set_xlabel('Date')
plt.show()


# In[80]:


one_df = merged_df.loc['2025-10-31 00:00:00':'2025-10-31 23:00:00', ['Demand (MWh)', 'Predicted Demand (MWh)']]

# Plot
ax = one_df.plot(figsize=(15,5))
ax.set_title('Electricity Demand vs Predictions (October 31, 2025)')
ax.set_ylabel('Demand (MWh)')
ax.set_xlabel('Date')
plt.show()


# # This is comparing only the historical data's performance, and Prophet isn't designed for that:

# In[81]:


actual = merged_df['Demand (MWh)']
predicted = merged_df['Predicted Demand (MWh)']

mse_value = mean_squared_error(actual, predicted)
rmse_value = np.sqrt(mse_value)
mae = mean_absolute_error(actual, predicted)
mape = (abs(actual - predicted) / actual).mean() * 100
bias = (predicted - actual).mean()
r2 = r2_score(actual, predicted) * 100
demand_mean = actual.mean()
percentage_rmse = (rmse_value / demand_mean) * 100

print(f"MSE: {mse_value:.2f}")
print(f"RMSE: {rmse_value:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Bias: {bias:.2f}")
print(f"Mean: {demand_mean:.2f}")
print(f"RMSE % of Mean: {percentage_rmse:.2f}%")
print(f"R2 Score: {r2:.2f}%")


# # I will show the metrics for the prediction of 7 days into the future, compared to the real data

# In[82]:


forecast = forecast.loc['2025-11-01 00:00:00':]

# 1-day (24 hours)
forecast_1d = forecast.head(24)

# 7-day (168 hours)
forecast_7d = forecast.head(168)

# 9-day (216 hours)
forecast_9d = forecast.head(216)


# In[83]:


forecast_1d.head(5)


# In[84]:


future_demand_real = pd.read_excel("srp_demand_october_november.xlsx")


# In[85]:


future_demand_real = future_demand_real.set_index('Local time')


# In[86]:


future_demand_real = future_demand_real.rename(columns={'Adjusted D':'Demand (MWh)'})
future_demand_real = future_demand_real[['Demand (MWh)']]


# In[87]:


future_demand_real.loc['2025-11-01 00:00:00':'2025-11-09 23:00:00']


# In[88]:


future_demand_compared = pd.merge(future_demand_real, forecast_9d, on='Local time')


# In[89]:


future_demand_compared


# In[90]:


one_week = future_demand_compared.loc['2025-11-01 00:00:00':'2025-11-07 23:00:00', ['Demand (MWh)', 'Predicted Demand (MWh)']]

# Plot
ax = one_week.plot(figsize=(15,5))
ax.set_title('Electricity Demand vs Predictions (November 1st-November 7th)')
ax.set_ylabel('Demand (MWh)')
ax.set_xlabel('Date')
plt.show()

actual = one_week['Demand (MWh)']
predicted = one_week['Predicted Demand (MWh)']

mse_value = mean_squared_error(actual, predicted)
rmse_value = np.sqrt(mse_value)
mae = mean_absolute_error(actual, predicted)
mape = (abs(actual - predicted) / actual).mean() * 100
bias = (predicted - actual).mean()
r2 = r2_score(actual, predicted) * 100
demand_mean = actual.mean()
percentage_rmse = (rmse_value / demand_mean) * 100

print(f"MSE: {mse_value:.2f}")
print(f"RMSE: {rmse_value:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"Bias: {bias:.2f}")
print(f"Mean: {demand_mean:.2f}")
print(f"RMSE % of Mean: {percentage_rmse:.2f}%")
print(f"R2 Score: {r2:.2f}%")


# In[91]:


# Function to compute metrics
def compute_metrics(actual, predicted, model_name):
    mse_value = mean_squared_error(actual, predicted)
    rmse_value = np.sqrt(mse_value)
    mae = mean_absolute_error(actual, predicted)
    mape = (abs(actual - predicted) / actual).mean() * 100
    bias = (predicted - actual).mean()
    demand_mean = actual.mean()
    percentage_rmse = (rmse_value / demand_mean) * 100
    r2 = r2_score(actual, predicted) * 100
    
    return {
        "Model": model_name,
        "MSE": mse_value,
        "RMSE": rmse_value,
        "MAE": mae,
        "MAPE (%)": mape,
        "Bias": bias,
        "Mean": demand_mean,
        "RMSE % of Mean": percentage_rmse,
        "R2 (%)": r2
    }

# --- Define the period for evaluation ---
start_date = '2025-10-25 00:00:00'
end_date   = '2025-10-31 23:00:00'

# --- Baseline ---
baseline_df = df.loc[start_date:end_date, ['Demand (MWh)', 'Demand Forecast (MWh)']]
baseline_metrics = compute_metrics(
    baseline_df['Demand (MWh)'], 
    baseline_df['Demand Forecast (MWh)'], 
    "Baseline (7-day forecast)"
)

# --- XGBoost ---
xgb_df = test.loc[start_date:end_date, ['Demand (MWh)', 'ml_prediction']]
xgb_metrics = compute_metrics(
    xgb_df['Demand (MWh)'], 
    xgb_df['ml_prediction'], 
    "XGBoost (7-day forecast)"
)

# --- Prophet ---
prophet_df = merged_df.loc[start_date:end_date, ['Demand (MWh)', 'Predicted Demand (MWh)']]
prophet_metrics = compute_metrics(
    prophet_df['Demand (MWh)'], 
    prophet_df['Predicted Demand (MWh)'], 
    "Prophet (7-day forecast)"
)

# --- Combine all metrics into a DataFrame ---
df_metrics = pd.DataFrame([baseline_metrics, xgb_metrics, prophet_metrics])

# Display with 2 decimal places and sort by RMSE
pd.set_option("display.float_format", "{:.2f}".format)
df_metrics_sorted = df_metrics.sort_values(by="RMSE", ascending=True)
df_metrics_sorted


# In[92]:


# df_metrics_sorted.to_csv("model_metric_comparison.csv")


# In[93]:


df


# # Correlation testing

# In[116]:


from scipy.stats import pearsonr

r, p = pearsonr(df["Temperature (F)"], df["Demand (MWh)"])
print("Pearson r:", r)
print("p-value:", p)


# In[97]:


population_and_demand = pd.DataFrame({
    "Year": [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016],
    "Population": [7583692.40, 7475644.80, 7367597.20, 7259549.60, 7151502.00, 7268175.00, 7164230.00, 7109843.75, 7055457.50],
    "Demand_MWh": [35683775.0, 33912809.0, 32464874.0, 31793878.0, 32208503.0, 30291298.0, 30381482.0, 29804595.0, 29470736.0]
})

r, p = pearsonr(population_and_demand["Population"], population_and_demand["Demand_MWh"])
print("Pearson r:", r)
print("p-value:", p)


# In[108]:


import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(endog=df["Demand (MWh)"],
                          groups=df["Season"],
                          alpha=0.05)
print(tukey)


# In[ ]:




