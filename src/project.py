# Python imports
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima_model import ARIMA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_squared_error

# Suppress warnings that may appear (improve notebook readability)
import warnings
from sklearn.exceptions import ConvergenceWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning as CW2
from statsmodels.tools.sm_exceptions import HessianInversionWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=CW2)
warnings.simplefilter(action='ignore', category=HessianInversionWarning)

# List of countries tested. The following data is used:
#   - Country name
#   - Two letter code
#   - Number of days between first reported cases and 2/15/20
#   - ARIMA order values that work for each country
countries = ["United States", "Germany", "United Kingdom", "France", "Japan", 
             "Canada", "Spain", "South Korea", "Italy", "Mexico"]
codes = ["US", "DE", "GB", "FR", "JP", "CA", "ES", "KR", "IT", "MX"]
diff_days = [25, 18, 14, 21, 31, 20, 13, 26, 15, 32]
arima_vals = [8, 6, 8, 8, 8, 4, 8, 4, 8, 4]

# Ensure all of the above lists have equal length
assert len(countries) == len(codes) == len(diff_days) == len(arima_vals)

# Iterate through each country
for index,country in enumerate(countries):
    # Get the list values for the current country
    print()
    print("{}".format(country))
    code = codes[index]
    diff = diff_days[index]
    arima_val = arima_vals[index]

    # Read daily COVID data
    covid_dataset = pd.read_csv("../data/raw/owid-covid-data.csv")
    covid_cases = covid_dataset[covid_dataset["location"]=="{}".format(country)]
    covid_cases = covid_cases.loc[:,("date", "total_cases", "new_cases", "new_cases_smoothed")].dropna()
    covid_cases = covid_cases[diff:]

    # Read daily transportation data
    daily_transits = pd.read_csv("../data/raw/mobility_reports/2020_{}_Region_Mobility_Report.csv".format(code))[:263]
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
    df.to_csv("../data/processed/{}.csv".format(code))      


    # Make new folder for visualizations (if one doesn't exist)
    foldername = "../visualizations/{}".format(code)
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    
    # Plot daily transit usage
    plt.clf()
    plt.plot(range(len(daily_transits)), daily_transits["Transits"])
    plt.xlabel("Day")
    plt.ylabel("Relative Transit Usage")
    plt.gcf().set_size_inches((16.0,8.0), forward=False)
    plt.savefig("{}/transit.png".format(foldername), bbox_inches='tight', dpi=100)

    ################################################################################
    # Regression
    ################################################################################

    # Get train and test sets
    minMax = MinMaxScaler()
    feature_set = covid_cases.loc[:,("Total","New")]
    #feature_set = minMax.fit_transform(feature_set)
    output_set = np.asarray(daily_transits["Transits"], dtype=np.int32).reshape(-1,1)
    #output_set = minMax.fit_transform(output_set)
    cases_train,cases_test,transport_train,transport_test = train_test_split(feature_set, output_set, test_size=0.5, random_state = 0) #, shuffle=False)
    x_range = range(len(cases_test))

    # Perform Random Forest Regression
    randomForest = RandomForestClassifier(n_estimators = 10)
    rf_pred = randomForest.fit(cases_train, transport_train.ravel()).predict(cases_test)
    plt.clf()
    plt.plot(x_range, transport_test, label="Actual Transit Uses")
    plt.plot(x_range, rf_pred, label="Predicted Transit Uses")
    plt.xlabel("Day")
    plt.ylabel("Relative Transit Uses")
    plt.legend()
    plt.gcf().set_size_inches((16.0,8.0), forward=False)
    plt.savefig("{}/Regression_RandomForest.png".format(foldername), bbox_inches='tight', dpi=100)

    # Perform Gradient Boosting Regression
    gradBoost = GradientBoostingClassifier()
    gb_pred = gradBoost.fit(cases_train, transport_train.ravel()).predict(cases_test)
    plt.clf()
    plt.plot(x_range, transport_test, label="Actual Transit Uses")
    plt.plot(x_range, gb_pred, label="Predicted Transit Uses")
    plt.xlabel("Day")
    plt.ylabel("Relative Transit Uses")
    plt.legend()
    plt.gcf().set_size_inches((16.0,8.0), forward=False)
    plt.savefig("{}/Regression_GradientBoost.png".format(foldername), bbox_inches='tight', dpi=100)

    # Perform MinMax preprocessing for the MLP tests
    feature_set = minMax.fit_transform(feature_set)
    cases_train,cases_test,transport_train,transport_test = train_test_split(feature_set, output_set, test_size=0.5, random_state = 0) #, shuffle=False)
    plt.clf()
    plt.plot(range(len(feature_set)), feature_set[:,0], label="Total COVID cases")
    plt.plot(range(len(feature_set)), feature_set[:,1], label="New COVID cases")
    plt.xlabel("Day")
    plt.ylabel("Normalized COVID Cases")
    plt.legend()
    plt.gcf().set_size_inches((16.0,8.0), forward=False)
    plt.savefig("{}/Cases_Normalized.png".format(foldername), bbox_inches='tight', dpi=100)

    # Perform Neural Network (MLP) Regression
    neuralNetwork = MLPRegressor(max_iter=10000)
    nn_pred = neuralNetwork.fit(cases_train, transport_train.ravel()).predict(cases_test)
    plt.clf()
    plt.plot(x_range, transport_test, label="Actual Transit Uses")
    plt.plot(x_range, nn_pred, label="Predicted Transit Uses")
    plt.xlabel("Day")
    plt.ylabel("Relative Transit Uses")
    plt.legend()
    plt.gcf().set_size_inches((16.0,8.0), forward=False)
    plt.savefig("{}/Regression_NeuralNetwork.png".format(foldername), bbox_inches='tight', dpi=100)

    print("\tRF stdev = {:.2f}".format(sqrt(mean_squared_error(rf_pred, transport_test.ravel()))))
    print("\tGB stdev = {:.2f}".format(sqrt(mean_squared_error(gb_pred, transport_test.ravel()))))
    print("\tNN stdev = {:.2f}".format(sqrt(mean_squared_error(nn_pred, transport_test.ravel()))))

    ################################################################################
    # Time series
    ################################################################################

    # Perform ARIMA forecasting
    arima = ARIMA(daily_transits["Transits"], order=(arima_val,0,arima_val))
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