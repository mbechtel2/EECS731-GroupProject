################################################################################
# 
# Command line parameters:
#   - Country name as listed in OWID dataset
#   - 2 letter country code
#   - Offset for number of days since first cases reported in OWID dataset
#        (ex: for US -> 1/21 to 2/14 is 25 days
#   - (Optional) ARIMA order value, defaults to 8
#
# Example uses:
#   - US => python project "United States" US 25
#   - DE => python project "Germany" DE 18 6
#
################################################################################

# Python imports
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.arima_model import ARIMA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

# Read daily COVID data
covid_dataset = pd.read_csv("../data/raw/owid-covid-data.csv")
covid_cases = covid_dataset[covid_dataset["location"]=="{}".format(sys.argv[1])]
covid_cases = covid_cases.loc[:,("date", "total_cases", "new_cases", "new_cases_smoothed")].dropna()
covid_cases = covid_cases[int(sys.argv[3]):]

# Read daily transportation data
daily_transits = pd.read_csv("../data/raw/mobility_reports/2020_{}_Region_Mobility_Report.csv".format(sys.argv[2]))[:263]
daily_transits = daily_transits.loc[:,("date", "transit_stations_percent_change_from_baseline")].dropna()

# Rename dataset columns (improve readability)
covid_cases = covid_cases.rename(columns={"date":"Date", "total_cases":"Total", 
                                 "new_cases":"New", "new_cases_smoothed":"New_smoothed"})
daily_transits = daily_transits.rename(columns={"date":"Date",  
                                       "transit_stations_percent_change_from_baseline":"Transits"})

# Check dataset lengths
assert len(covid_cases) == len(daily_transits), "Cases={}, Transits={}".format(len(covid_cases), len(daily_transits))

# Save all data to new dataset
df = pd.DataFrame(data={"Date":covid_cases["Date"],"Total_cases":covid_cases["Total"],
                    "New_cases":covid_cases["New"],"New_smoothed":covid_cases["New_smoothed"], 
                    "Transits":daily_transits["Transits"].values}).reset_index()
df.to_csv("../data/processed/{}.csv".format(sys.argv[2]))      


# Make new folder for visualizations (if one doesn't exist)
foldername = "../visualizations/{}".format(sys.argv[2])
if not os.path.exists(foldername):
    os.makedirs(foldername)

################################################################################
# Regression
################################################################################

# Get train and test sets
feature_set = covid_cases.loc[:,("Total","New")]
output_set = np.asarray(daily_transits["Transits"], dtype=np.int32)
cases_train,cases_test,transport_train,transport_test = train_test_split(feature_set, output_set, test_size=0.50, random_state = 0, shuffle=False)
x_range = range(len(cases_test))

# Perform Random Forest Regression
randomForest = RandomForestClassifier()
rf_pred = randomForest.fit(cases_train, transport_train).predict(cases_test)
plt.plot(x_range, transport_test, label="Actual Transit Uses")
plt.plot(x_range, rf_pred, label="Predicted Transit Uses")
plt.xlabel("Day")
plt.ylabel("Relative Transit Uses")
plt.legend()
plt.gcf().set_size_inches((16.0,8.0), forward=False)
plt.savefig("{}/Regression_RandomForest.png".format(foldername), bbox_inches='tight', dpi=100)

################################################################################
# Time series
################################################################################

# Perform ARIMA forecasting
if len(sys.argv) >= 5:
    val = int(sys.argv[4])
    arima = ARIMA(daily_transits["Transits"], order=(val,0,val))
else:
    arima = ARIMA(daily_transits["Transits"], order=(8,0,8))
arima_pred = arima.fit(disp=0)
arima_pred.plot_predict(start=1,end=len(daily_transits["Transits"])+100)
plt.xlabel("Day")
plt.ylabel("Relative Transit Uses")
plt.gcf().set_size_inches((16.0,8.0), forward=False)
plt.savefig("{}/TimeSeries_ARIMA.png".format(foldername), bbox_inches='tight', dpi=100)

################################################################################
# Anomaly Detection
################################################################################

# Perform Local Outlier detection
plt.clf()
localOutlier = LocalOutlierFactor()
local_pred = localOutlier.fit_predict(daily_transits["Transits"].values.reshape(-1,1))
x_range = range(len(daily_transits["Transits"]))
plt.scatter(x_range, daily_transits["Transits"], c=local_pred)
plt.xlabel("Day")
plt.ylabel("Relative Transit Uses")
plt.gcf().set_size_inches((16.0,8.0), forward=False)
plt.savefig("{}/AnomalyDetection_LocalOutlier.png".format(foldername), bbox_inches='tight', dpi=100)