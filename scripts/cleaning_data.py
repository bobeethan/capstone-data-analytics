#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ---------------------------
# Data Manipulation Libraries
# ---------------------------
import pandas as pd        # Handle tabular data (DataFrames), CSV/Excel I/O
import numpy as np         # Numerical operations, arrays, linear algebra

# ---------------------------
# Data Visualization Libraries
# ---------------------------
import matplotlib.pyplot as plt   # Core plotting library
import seaborn as sns             # High-level statistical visualization

# ---------------------------
# Utilities & System Helpers
# ---------------------------
import warnings               # Manage warning messages
from datetime import datetime # Work with dates and timestamps

# ---------------------------
# Machine Learning for Imputation
# ---------------------------
from sklearn.ensemble import GradientBoostingRegressor
# Gradient boosting model, can be used to predict missing values in datasets

# ---------------------------
# Time Series Analysis Tools
# ---------------------------
from statsmodels.tsa.seasonal import seasonal_decompose
# Decompose time series into trend, seasonal, and residual components
from statsmodels.graphics.tsaplots import plot_acf
# Plot autocorrelation function to understand time series patterns


# # Importing data and cleaning

# In[2]:


warnings.filterwarnings(
    "ignore",
    message="Conditional Formatting extension is not supported"
)

df = pd.read_excel("SRP_Electricity_Demand.xlsx", sheet_name='Known Data Issues')

# Select relevant columns
df = df[["Local date", "Hour", "Local time", "Adjusted D", "Adjusted NG", "DF", "Adjusted TI"]]

df = df.rename(columns={'Adjusted D': 'Demand (MWh)', 'Adjusted NG': 'Net Generation (MWh)', 'Adjusted TI': 'Total Interchange (MWh)', 'DF':'Demand Forecast (MWh)'})

# Convert to datetime and set index
df["Local time"] = pd.to_datetime(df["Local time"])
df = df.set_index("Local time")

# Filter rows up to 2025-10-31
df = df[(df.index >= "2015-01-01") & (df.index <= "2025-10-31 23:00:00")]

df


# # There are missing values and we need to impute them:

# In[3]:


for col in ['Demand (MWh)', 'Demand Forecast (MWh)', 'Net Generation (MWh)']:
    df[col] = df[col].mask(df[col] < 10, np.nan)

# Optional: check how many NaNs were created
for col in ['Demand (MWh)', 'Demand Forecast (MWh)', 'Net Generation (MWh)']:
    print(f"{col}: {df[col].isna().sum()} missing values")


# In[4]:


df.plot(y='Demand (MWh)', figsize= (15,5))


# In[5]:


# Check how many null values
(df.isnull().sum().sort_values(ascending=False))


# In[6]:


((df.isnull().sum()/len(df))*100).sort_values(ascending=False)


# # Filling null values using ML

# In[7]:


df.reset_index(inplace=True)

# List of target columns to process
target_columns = [
    'Net Generation (MWh)',
    'Total Interchange (MWh)',
    'Demand Forecast (MWh)',
    'Demand (MWh)'
]

# Time range for plotting example month
start_month = pd.to_datetime('2015-01-01')
end_month = pd.to_datetime('2025-10-31')

# Make a single copy to update progressively
filled_data = df.copy()

for target_col in target_columns:
    print(f"\n--- Processing column: {target_col} ---")
    
    # ---------------------------
    # 1. Separate non-missing and missing data
    # ---------------------------
    non_missing_data = filled_data.dropna(subset=[target_col]).copy()
    missing_data = filled_data[filled_data[target_col].isna()].copy()

    if missing_data.empty:
        print(f"No missing values for {target_col}, skipping GBR.")
        continue

    # ---------------------------
    # 2. Extract time-based features
    # ---------------------------
    for dataset in [non_missing_data, missing_data]:
        dataset['hour'] = dataset['Local time'].dt.hour
        dataset['dayofweek'] = dataset['Local time'].dt.dayofweek
        dataset['month'] = dataset['Local time'].dt.month

    # ---------------------------
    # 3. Train Gradient Boosting Regressor
    # ---------------------------
    features = ['hour', 'dayofweek', 'month']
    X_train = non_missing_data[features]
    y_train = non_missing_data[target_col]
    X_missing = missing_data[features]

    gb_regressor = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    gb_regressor.fit(X_train, y_train)

    # ---------------------------
    # 4. Predict missing values and fill in-place
    # ---------------------------
    predicted_values = gb_regressor.predict(X_missing)
    filled_data.loc[missing_data.index, target_col] = predicted_values

    # ---------------------------
    # 5. Time-based interpolation
    # ---------------------------
    filled_data = filled_data.set_index('Local time')
    filled_data[target_col] = filled_data[target_col].interpolate(method='time')
    filled_data = filled_data.reset_index()

    # ---------------------------
    # 6. STL Decomposition
    # ---------------------------
    original_series = df.set_index('Local time')[target_col]
    original_decompose = seasonal_decompose(original_series.interpolate(), model='additive', period=144)

    imputed_series = filled_data.set_index('Local time')[target_col]
    imputed_decompose = seasonal_decompose(imputed_series, model='additive', period=144)

    # ---------------------------
    # 7. Plot example month
    # ---------------------------
    plot_data = filled_data[(filled_data['Local time'] >= start_month) &
                            (filled_data['Local time'] <= end_month)]
    original_month_data = df[(df['Local time'] >= start_month) &
                             (df['Local time'] <= end_month)]

    plt.figure(figsize=(14, 6))
    plt.plot(plot_data['Local time'], plot_data[target_col], label=f'Imputed Data ({target_col})', color='green', alpha=0.8)
    plt.plot(original_month_data['Local time'], original_month_data[target_col], label='Original Data (with Missing)', color='red', alpha=0.9)
    plt.title(f'Original vs. Gradient Boosting Imputed {target_col}')
    plt.xlabel('Datetime')
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------------------
    # 8. Autocorrelation comparison
    # ---------------------------
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plot_acf(df[target_col].dropna(), lags=50, ax=plt.gca(), title=f'Original Data ACF ({target_col})')
    plt.grid(True)
    plt.subplot(1,2,2)
    plot_acf(filled_data[target_col], lags=50, ax=plt.gca(), title=f'Imputed Data ACF ({target_col})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # 9. STL Decomposition plots
    # ---------------------------
    plt.figure(figsize=(14,5))
    plt.plot(original_decompose.trend, label='Original Trend', color='blue')
    plt.plot(imputed_decompose.trend, label='Imputed Trend (GBR + Smoothing)', color='green', linestyle='--')
    plt.title(f'Trend Comparison: {target_col}')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(14,5))
    plt.plot(original_decompose.seasonal, label='Original Seasonality', color='blue', alpha=0.7)
    plt.plot(imputed_decompose.seasonal, label='Imputed Seasonality (GBR + Smoothing)', color='green', linestyle='--', alpha=0.7)
    plt.title(f'Seasonality Comparison: {target_col}')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[8]:


# Check how many null values
(filled_data.isnull().sum().sort_values(ascending=False))


# # Import weather data and clean

# In[9]:


Phoenix_Weather_Data_2015_2025 = pd.read_csv("Phoenix_Weather_Data_2015_2025.csv")
Phoenix_Weather_Data_2015_2025['Local time'] = (
    pd.to_datetime(Phoenix_Weather_Data_2015_2025['Year'].astype(str), format='%Y')
    + pd.to_timedelta(Phoenix_Weather_Data_2015_2025['Day of Year'] - 1, unit='D')
    + pd.to_timedelta(Phoenix_Weather_Data_2015_2025['Hour of Day'] - 1, unit='h')
)
Phoenix_Weather_Data_2015_2025['Temperature (F)'] = (Phoenix_Weather_Data_2015_2025['Air Temperature'] * 9/5) + 32
Phoenix_Weather_Data_2015_2025['Temperature (F)'] = Phoenix_Weather_Data_2015_2025['Temperature (F)'].round(decimals=0).astype(int)
Phoenix_Weather_Data_2015_2025 = Phoenix_Weather_Data_2015_2025[["Year", "Hour of Day", "Temperature (F)", "Local time"]]
Phoenix_Weather_Data_2015_2025 = Phoenix_Weather_Data_2015_2025.rename(columns={'Hour of Day': 'Hour'})
Phoenix_Weather_Data_2015_2025


# In[10]:


# Check how many null values
(Phoenix_Weather_Data_2015_2025.isnull().sum().sort_values(ascending=False))


# # Merge the datasets

# In[11]:


merged_df = pd.merge(Phoenix_Weather_Data_2015_2025, filled_data, on='Local time')
merged_df


# In[12]:


merged_df.set_index('Local time', inplace=True)
merged_df.sort_index(ascending=False, inplace=True)


# In[13]:


(merged_df.isnull().sum().sort_values(ascending=False))


# In[14]:


merged_df


# In[15]:


merged_df = merged_df[['Temperature (F)', 'Demand (MWh)', 'Net Generation (MWh)', 'Demand Forecast (MWh)', 'Total Interchange (MWh)']]
merged_df


# In[16]:


# I commented it out because the saved dataset is in the cleaned folder
# merged_df.to_csv("demand_df.csv")

