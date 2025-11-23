# capstone-data-analytics
Final Capstone Project for Data Analytics Course

This analysis goes over Salt River Project's customer Electricity Demand (MWh) from a decade ago up to October 31st 2025.
I perform EDA on the dataset and utilize ML and predictive models to predict future electricity Demand.
This is a time series dataset with high granuality down to the hour. 

I recommend installing Anaconda at https://anaconda.org/ and using Jupyter Notebook to open the .ipynb files instead of executing a script, however I included the required .py scripts for users utilizing IDEs such as VS code and PyCharm.

The dataset folders follow the same naming conventions as the associated Python scripts or notebooks.
This makes it easy to place all datasets into the same directory as the script/notebook you are running.

A more in depth explanation of this analysis is located in my report under the documentation folder.

I utilize read_csv and read_excel in pandas, which assumes the datafile is located in the same folder as the notebook or python script.

All the libraries I utilize for my analysis:

Included in an Anaconda Python Env:
- pandas            (DataFrames)
- numpy             (numerical computing)
- matplotlib        (plotting)
- seaborn           (statistical visualization)
- scikit-learn      (ML models + metrics + train/test split)
- statsmodels       (seasonal_decompose, plot_acf)

Not included in Anaconda:
- holidays          (country holiday calendars)
- xgboost           (gradient boosting models)
- feature-engine    (Lag features for time series)
- statsforecast     (classical time series forecasting)
- prophet           (additive regression forecasting model)

The actual folders contained are datasets, documentation, scripts, and visualizations.

Datasets contain my raw and cleaned data used for analysis with each .ipnyb notebook file in each.
The actual documenation includes my overall project report and Power BI instructions.
The scripts contain the .ipnyb notebook files as well as python .py scripts.
Lastly, the visualizations contain my Power BI project and images of any graphs generated through Seaborn and MatPlotLib
