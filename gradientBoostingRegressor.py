
import numpy as np
import time
from sklearn import ensemble
from sklearn.utils import shuffle
# import sys
# from sklearn import datasets
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt


def precipitation_round(x):
    if x <= 70:
        return np.floor(x/2)*2+1
    else:
        return np.floor((x-70)/100)*100+570


def categorize(precipitation):
    categories = list()
    for i in range(len(precipitation)):
        precipitation[i] = precipitation_round(precipitation[i])
        if categories.count(precipitation[i]) == 0:
            categories.append(precipitation[i])
    for i in range(len(precipitation)):
        precipitation[i] = categories.index(precipitation[i])
    return categories


def zero_fill(data, label):
    fill_count = 0
    for i in range(len(data)):
        if data[i] == label:
            data[i] = 0.0
            fill_count += 1
    return fill_count


# prediction is numpy.ndarray
def encode_category(prediction, categories):
    result = [0]*len(prediction)
    for i in range(len(prediction)):
        result[i] = categories[prediction[i].argmax()]
    return result


# set zero values for empty observation in test
def test_zero_fill(data, prediction, label, zero_value):
    fill_count = 0
    empty_range = range(4, 32)  # id already removed, so from 4th column, 28 columns in total
    for i in range(len(data)):
        for j in empty_range:
            if data[i, j] != label:
                break
        else:
            prediction[i] = zero_value
            fill_count += 1
    return fill_count


# make negative values zero
def predict_zero_negative(data):
    fill_count = 0
    for i in range(len(data)):
        if data[i] < 0:
            data[i] = 0.0
            fill_count += 1
    return fill_count

###############################################################################

# Column Meaning:
# 0 id
# 1 observation line count
# 2 valid data count
# 3 valid data percentage
# 4 distance
# 5-12 reflectivity->precipitation, 5 is the baseline
# 13-32 all radar data including reflectivity, time averaged
# 33-39/40(depends on whether or not remove the last precipitation type)
# precipitation type likelihoods
# last column(40/41): expected

# Load data
t0 = time.clock()
data_train = np.loadtxt('features.csv', dtype='float32', delimiter=',')
sample_count, feature_count = data_train.shape
print("Finished fetching data with {:g} samples in {:.0f} seconds.".format(sample_count, time.clock()-t0))

# define all parameters
is_quiet = False
is_test = False
empty_prediction_value = 0.75
random_seed = 1
missing_label = -999.0
gbr_param = {'n_estimators': 600, 'max_depth': 6, 'min_samples_split': 100,
             'learning_rate': 0.1, 'loss': 'lad', 'max_features': 10}
# gbr_param = {'n_estimators': 400, 'max_depth': 6, 'min_samples_split': 1,
#              'learning_rate': 0.1, 'loss': 'lad', 'max_features': 5}

# preprocessing data
X = data_train[:, 1:-1]
y = data_train[:, -1]
y_test = list()
data_test = list()
mae_base = 0
if is_test:
    X_train, y_train = X, y
    t0 = time.clock()
    data_test = np.loadtxt('features_test.csv', dtype='float32', delimiter=',')
    X_test = data_test[:, 1:]
    print("Finished fetching test data with {:g} samples in {:.0f} seconds.".format(len(data_test), time.clock()-t0))
else:
    y_base = data_train[:, 5]
    zero_fill_count = zero_fill(y_base, missing_label)
    if not is_quiet:
        print("Finished zero filling {:d} missing numbers.".format(zero_fill_count))

    X, y = shuffle(X, y, random_state=random_seed)
    offset = np.floor(X.shape[0] * 0.8)
    X_train, y_train, y_base_train = X[:offset], y[:offset], y_base[:offset]
    X_test, y_test, y_base_test = X[offset:], y[offset:], y_base[offset:]
    mae_base = abs(y_test - y_base_test).mean()

#################################################################################
# Fit regression model
clf = ensemble.GradientBoostingRegressor(**gbr_param)

t0 = time.clock()
clf.fit(X_train, y_train)
if not is_quiet:
    print("Finished training Gradient Boosting Regressor with {:g} features in {:.1f} minutes.".
          format(feature_count, (time.clock()-t0)/60.0))

if is_test:
    y_test = clf.predict(X_test)
    zero_fill_count = predict_zero_negative(y_test)
    if not is_quiet:
        print("Replaced {:d} negative predictions with zeors out of {:d} test predictions."
              .format(zero_fill_count, len(y_test)))
    zero_fill_count = test_zero_fill(X_test, y_test, missing_label, empty_prediction_value)
    if not is_quiet:
        print("Filled {:d} empty observations out of {:d} test observations.".format(zero_fill_count, len(y_test)))
    output_file = open('test_prediction_gbr.csv', 'w')
    output_file.write("Id,Expected\n")
    for ii in range(len(data_test)):
        out_str = str(int(data_test[ii, 0]))+","+str(y_test[ii])
        output_file.write(out_str+'\n')
    output_file.close()
else:
    y_predict_gbr = clf.predict(X_test)
    zero_fill_count = predict_zero_negative(y_predict_gbr)
    if not is_quiet:
        print("Replaced {:d} negative predictions with zeors out of {:d} predictions."
              .format(zero_fill_count, len(y_predict_gbr)))
    mae = abs(y_test - y_predict_gbr).mean()
    print("GradientBoostingRegressor MAE = {:.4f} while base MAE = {:4f}".format(mae, mae_base))
