# EECS731-GroupProject

Team name: Traveling with Confidence?

Group members: Dana Almansour, Michael Bechtel, and Brandon Wheat

In this project, we perform Regression, Time Series Forecasting and Anomaly Detection on daily COVID case and transit hub usage datasets for different countries around the world. In particular, we look at the number of new cases per day and the running total cases in each country and how they relate to the public transportation usage in each country. In the end, we make the following observations and findings:

- We find that COVID cases do have a positive correlation to transit hub usage, but that they can't be used for predicting future public transportation trends.
- We find that all countries are expected to continue their current trends for the near future.
- We find that all countries do have outlying days, but that some countries produce results that would have more potential for helping to address and improve public transportation trends.

## Data Sources

Daily Transit: [Google Community Mobility Reports](https://www.google.com/covid19/mobility/)

COVID Cases: [Our-World-in-Data (OWID)](https://ourworldindata.org/coronavirus-source-data)

## Usage

	cd src/
	python project.py 
	
All numerical results are output to the console, while all figures and graphs are saved to the visualizations/ directory. Namely, each country will have their own subdirectory named after they 2-letter code.

## Acknowledgements

Google LLC "Google COVID-19 Community Mobility Reports".
https://www.google.com/covid19/mobility/ Accessed: November 8, 2020.
