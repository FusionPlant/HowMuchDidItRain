import numpy as np
import time
from sklearn import ensemble
import cPickle as cP
# import sys
# from sklearn import datasets
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt


def zero_fill(data, label):
    fill_count = 0
    for i in range(len(data)):
        if data[i] == label:
            data[i] = 0.0
            fill_count += 1
    return fill_count


# set zero values for empty observation in test
def test_zero_fill(data, prediction, label, zero_value):
    fill_count = 0
    empty_range = zero_fill_range
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

# data_list Columns:
# 0 id
# 1 observation line count
# 2 valid data count
# 3 valid data percentage
# 4 distance
# 5-20 reflectivity->precipitation, 5 is time-averaged baseline, 6 is simple averaged baseline
# 21-60 all radar data including reflectivity, time averaged followed by simple averaged
# 61-92 precipitation type likelihoods
# 93(does not exist in test file) expected precipitation

# define all parameters
is_test = True
empty_prediction_value = 0.762
random_seed = 13
missing_label = -999.0
gbr_param = {'n_estimators': 2000, 'max_depth': 12, 'min_samples_split': 100,
             'min_samples_leaf': 200, 'min_weight_fraction_leaf': 0, 'subsample': 0.7,
             'learning_rate': 0.4, 'loss': 'lad', 'max_features': 15}
zero_fill_range = range(4, 32)  # id already removed, so from 4th column, 28 columns in total
data_folder = 'data'

# preprocessing data
y_test = np.array([])
data_test = np.array([])
y_base_test = np.array([])
y_base_simple_test = np.array([])
if is_test:
    t0 = time.clock()
    file_handle = open(data_folder+'/training_data_4test', 'r')
    [X_train, y_train] = cP.load(file_handle)
    file_handle.close()
    file_handle = open(data_folder+'/testing_data', 'r')
    [data_test] = cP.load(file_handle)
    file_handle.close()
    X_test = data_test[:, 1:]
    print("Finished fetching {:d} training samples and {:d} testing observations in {:.0f} seconds."
          .format(len(X_train), len(data_test), time.clock()-t0))
else:
    t0 = time.clock()
    file_handle = open(data_folder+'/training_data_4cv' + str(random_seed), 'r')
    [X_train, y_train, X_test, y_test, y_base_test, y_base_simple_test] = cP.load(file_handle)
    file_handle.close()
    print("Finished fetching {:d} training samples in {:.0f} seconds."
          .format(len(y_train)+len(y_test), time.clock()-t0))

#################################################################################
# Fit regression model

t0 = time.clock()
clf = ensemble.GradientBoostingRegressor(**gbr_param)
print("Start training GBR model, please wait...")
clf.fit(X_train, y_train)
print("Finished training GBR model with {:d} features in {:.1f} minutes."
      .format(len(X_train[0]), (time.clock()-t0)/60.0))

if is_test:
    y_test = clf.predict(X_test)
    zero_fill_count = predict_zero_negative(y_test)
    print("Replaced {:d} negative predictions with zeors out of {:d} GBR predictions."
          .format(zero_fill_count, len(y_test)))
    zero_fill_count = test_zero_fill(X_test, y_test, missing_label, empty_prediction_value)
    print("Filled {:d} empty observations out of {:d} test observations.".format(zero_fill_count, len(y_test)))
    output_file = open(data_folder+'/test_prediction_gbr.csv', 'w')
    output_file.write("Id,Expected\n")
    for ii in range(len(data_test)):
        out_str = str(int(data_test[ii, 0]))+","+str(y_test[ii])
        output_file.write(out_str+'\n')
    output_file.close()
else:
    y_predict_gbr = clf.predict(X_test)
    y_train_predict_gbr = clf.predict(X_train)
    zero_fill_count = predict_zero_negative(y_predict_gbr)
    predict_zero_negative(y_train_predict_gbr)
    print("Replaced {:d} negative predictions with zeors out of {:d} GBR predictions."
          .format(zero_fill_count, len(y_predict_gbr)))
    print("\n******************* Feature Importance ****************************\n")
    print(clf.feature_importances_)
    print("\n*******************************************************************\n")
    mae_train = abs(y_train - y_train_predict_gbr).mean()
    mae = abs(y_test - y_predict_gbr).mean()
    mae_base = abs(y_test - y_base_test).mean()
    mae_base_simple = abs(y_test - y_base_simple_test).mean()
    msg = "GBR training MAE = {:.4f}, GBR testing MAE = {:.4f}, " +\
          "baseline testing MAE = {:.4f}, simple averaged baseline testing MAE = {:.4f}."
    print(msg.format(mae_train, mae, mae_base, mae_base_simple))

    mae_mix = list()
    mae_simple_mix = list()
    for ii in range(9):
        ratio = ii*0.1+0.1
        mae_mix.append(abs(ratio*y_base_test + (1.0-ratio)*y_predict_gbr - y_test).mean())
        mae_simple_mix.append(abs(ratio*y_base_simple_test + (1.0-ratio)*y_predict_gbr - y_test).mean())
    print("GBR and baseline mixed MAE = {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}."
          .format(*mae_mix))
    print("GBR and simple averaged baseline mixed MAE = {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}."
          .format(*mae_simple_mix))
    print("{:d} out of {:d} GBR predictions is greater than baseline prediction."
          .format(np.sum(np.greater(y_predict_gbr, y_base_test)), len(y_predict_gbr)))
    print("{:d} out of {:d} GBR predictions is greater than simple averaged baseline prediction."
          .format(np.sum(np.greater(y_predict_gbr, y_base_simple_test)), len(y_predict_gbr)))
