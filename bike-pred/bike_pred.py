import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the data and prepare for use
data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)
# rides[:24*10].plot(x='dteday', y='cnt')

# Convert categorical variables like season, weather, month to dummy variables
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

# Remove unused columns
fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(labels=fields_to_drop, axis=1)

# To make training the network easier, we'll standardize each of the continuous variables.
# That is, we'll shift and scale the variables such that they have zero mean and a standard deviation of 1.

# The scaling factors are saved so we can go backwards when we use the network for predictions.
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std

# We'll save the data for the last approximately 21 days to use as a test set after we've trained the network.
# We'll use this set to make predictions and compare them with the actual number of riders.

# Save data for approximately the last 21 days
test_data = data[-21*24:]

# Now remove the test data from the data set
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# We'll split the data into two sets, one for training and one for validating as the network is being trained.
# Since this is time series data, we'll train on historical data, then try to predict on future data (the validation set).

# Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]