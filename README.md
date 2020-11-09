# EECS731-GroupProject

## Data Sources

Daily Transit: [Google Community Mobility Reports](https://www.google.com/covid19/mobility/)

COVID Cases: [Our-World-in-Data (OWID)](https://ourworldindata.org/coronavirus-source-data)

## Usage

	cd src/
	python project.py <name> <code> <numdays> <order>
	
The project.py file takes three to four command line parameters:

- name: The name of the country as listed in the OWID dataset (e.g. "United States")
- code: The two digit code associated with the country (e.g. US)
- numdays: The number of days from when the first cases reported in the country to 2/15 (when the community mobility reports begin their data). For example, the US would be 1/21 to 2/14, which would be 25 days.
- order: Optional parameter. In some cases, the default order values used for the ARIMA model don't work. As such, another order value can be passed. Otherwise, this can be left empty.

Example uses:

	python project "United States" US 25
	python project "Germany" DE 18 6

## Acknowledgements

Google LLC "Google COVID-19 Community Mobility Reports".
https://www.google.com/covid19/mobility/ Accessed: November 8, 2020.