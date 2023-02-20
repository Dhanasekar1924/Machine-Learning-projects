# TAXI-FARE-PREDICTION
This project aims to predict taxi fares based on features such as pickup and dropoff locations, passenger count, and pickup time.

Getting Started
Prerequisites
The project requires the following packages:

mlcp
numpy
pandas
datetime
math
matplotlib
scipy
Installing

To install the required packages, run the following command:
pip install mlcp numpy pandas datetime math matplotlib scipy

Running the code
To run the code, open the Jupyter Notebook file "TaxiFare_Prediction.ipynb" and run each cell sequentially.

Code structure
The code is organized into several sections, each with a specific purpose:

Data reading: Reads the input data file and sets up parameters for data analysis.
Primary analysis: Performs basic analysis on the data, such as identifying unwanted features, checking for missing values, and visualizing the target variable.
Visual analysis: Creates scatter plots of the input variables against the target variable to identify correlations.
Feature engineering: Performs transformations on the input variables to improve predictive power.
Model training: Trains models on the transformed data to predict the target variable.
Model evaluation: Evaluates the performance of the trained models using various metrics.
Model tuning: Uses grid search to find the best hyperparameters for the models.
Data
The input data file is "data/RFC_4/TaxiFare.csv". The file contains the following features:

unique_id
date_time_of_pickup
longitude_of_pickup
latitude_of_pickup
longitude_of_dropoff
latitude_of_dropoff
passenger_count
amount
The target variable is "amount", which represents the taxi fare.

Results

After performing feature engineering and model training, the best model achieved an R^2 score of 0.76 on the test set. This suggests that the model explains 76% of the variance in the target variable. Further improvements to the model could be made by collecting more data, engineering additional features, or using more advanced modeling techniques.

Acknowledgments
This project was created as part of the Machine Learning Career Path on Codecademy. The input data file was provided by Codecademy.
