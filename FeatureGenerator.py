import math
import time
import numpy as np
import pickle
from sklearn.utils import shuffle


# Output an float list from input string, with None indicating missing data
def csv_line_reader(line_str):
    line_data = line_str.rstrip().split(',')
    for i in range(len(line_data)):
        if line_data[i].rstrip() == '':
            line_data[i] = None
        else:
            line_data[i] = float(line_data[i])
    return line_data


# Output a generator, whose each item is an array, which contains all data from a same id
def by_id_reader(input_file):
    data_id = None
    id_block = None

    for line_num, line in enumerate(input_file):
        if line_num % 100000 == 0:
            print "Reading input line {:d}".format(line_num)

        line_data = csv_line_reader(line)
        if line_data[0] is None:  # should never happen #debug#
            print "Error: Missing id in line {:d}".format(line_num)
        if line_data[0] != data_id:
            data_id = line_data[0]
            if id_block is not None:
                yield id_block
            id_block = [line_data]
        else:
            id_block.append(line_data)

    if id_block is not None:
        yield id_block


# count number of valid data in each data block from a same id
def count_data(id_block):
    line_count = len(id_block)
    data_count = 0
    for i in range(line_count):
        for j in to_count_data_col:
            if id_block[i][j] is not None:
                data_count += 1
    return data_count


# use reflectivity array to predict precipitation in one hour in mm
# See: https://en.wikipedia.org/wiki/DBZ_(meteorology)
def predict_with_reflectivity(reflectivity, time_min, is_time_average):
    mm_per_hr = None
    last_valid_minute = 0.0
    precipitation_mm = 0.0
    if is_time_average:
        for dbz, minute in zip(reflectivity, time_min):
            if dbz is not None:
                if mm_per_hr is None:  # first valid entry
                    mm_per_hr = pow(pow(10.0, dbz/10.0)/200, 0.625)
                    precipitation_mm += mm_per_hr * minute
                    last_valid_minute = minute
                else:  # after first valid entry
                    precipitation_mm += mm_per_hr * (minute-last_valid_minute)/2
                    mm_per_hr = pow(pow(10.0, dbz/10.0)/200, 0.625)
                    precipitation_mm += mm_per_hr * (minute-last_valid_minute)/2
                    last_valid_minute = minute
        if mm_per_hr is None:
            return missing_label
        else:
            precipitation_mm += mm_per_hr * (60.0-last_valid_minute)
            return precipitation_mm / 60.0
    else:
        value_count = 0
        for dbz in reflectivity:
            if dbz is not None:
                precipitation_mm += pow(pow(10.0, dbz/10.0)/200, 0.625)
                value_count += 1
        if value_count == 0:
            return missing_label
        else:
            return precipitation_mm / value_count


# average data from value array
def average_values(value_all, time_min, is_time_average):
    last_valid_value = None
    last_valid_minute = 0.0
    value_sum = 0.0
    if is_time_average:
        for value, minute in zip(value_all, time_min):
            if value is not None:
                if last_valid_value is None:  # first valid entry
                    last_valid_value = value
                    value_sum += last_valid_value * minute
                    last_valid_minute = minute
                else:  # after first valid entry
                    value_sum += last_valid_value * (minute-last_valid_minute)/2
                    last_valid_value = value
                    value_sum += last_valid_value * (minute-last_valid_minute)/2
                    last_valid_minute = minute
        if last_valid_value is None:
            return missing_label
        else:
            value_sum += last_valid_value * (60.0-last_valid_minute)
            return value_sum / 60.0
    else:
        value_count = 0
        for value in value_all:
            if value is not None:
                value_sum += value
                value_count += 1
        if value_count == 0:
            return missing_label
        else:
            return value_sum / value_count


# determine if the values in list x is in the range specified in list y
# to be updated
def type_likelihood_cal(radar_data, type_range, type_gaussian):
    likelihood = 1.0
    for i in range(len(radar_data)):
        if radar_data[i] is None:
            pass
        elif radar_data[i] < type_range[i][0] or radar_data[i] > type_range[i][1]:
            return 0
        else:
            likelihood *= math.exp(-(radar_data[i]-type_gaussian[i][0])**2 / (2*type_gaussian[i][1]**2)) / \
                          type_gaussian[i][1]
    return likelihood


# use radar data to predict precipitation type
# Ref in dBz, RhoHV is unit-less, Zdr in dB, Kdp in deg/km
def precipitation_types_likelihood_cal(ref, rho_hv, zdr, kdp):
    likelihoods = [0]*type_count
    for i in range(type_count):
        likelihoods[i] = type_likelihood_cal([ref, rho_hv, zdr, kdp], type_list[i], gaussian_list[i])

    sum_likelihood = sum(likelihoods)
    if sum_likelihood == 0:
        return [1.0/type_count]*type_count

    for i in range(type_count):
        likelihoods[i] /= sum_likelihood
    return likelihoods


def list_addition(x_list, y_list):
    for i in range(len(x_list)):
        x_list[i] += y_list[i]


# for each data block from a same id, process it and return one line of data in list form
def id_block_process(id_block):
    # Throw away >70mm data
    if len(id_block[0]) == 24 and id_block[0][23] > max_precipitation:
            return None

    line_count = len(id_block)
    features = list()

    # Add id as feature
    features.append(id_block[0][id_col])

    # Add number of columns as feature
    features.append(line_count)

    # Add number and percentage of non-missing data as feature
    valid_data_count = count_data(id_block)
    features.append(valid_data_count)
    features.append(valid_data_count / float(line_count*len(to_count_data_col)))

    # Add distance as feature
    features.append(id_block[0][dist_col])

    radar_begin_col = len(features)

    # Time interval calculation
    time_min = [0]*line_count
    for i in range(line_count):
        time_min[i] = int(id_block[i][time_min_col])  # time data should never be missing

    # Add precipitation predictions as features
    for j in reflectivity_col:
        reflectivity = list()
        for i in range(line_count):
            reflectivity.append(id_block[i][j])
        features.append(predict_with_reflectivity(reflectivity, time_min, True))
        features.append(predict_with_reflectivity(reflectivity, time_min, False))

    # Add averaged values as features
    for j in to_average_col:
        value = list()
        for i in range(line_count):
            value.append(id_block[i][j])
        features.append(average_values(value, time_min, True))
        features.append(average_values(value, time_min, False))

    # Remove empty lines
    if drop_empty_block:
        for i in range(radar_begin_col, len(features)):
            if features[i] != missing_label:
                break
        else:
            return None

    # Add precipitation type
    # to be updated
    for j in range(len(ref_col)):
        type_likelihood = [0]*type_count
        for i in range(line_count):
            list_addition(type_likelihood,
                          precipitation_types_likelihood_cal(id_block[i][ref_col[j]], id_block[i][rho_hv_col[j]],
                                                             id_block[i][zdr_col[j]], id_block[i][kdp_col[j]]))
        type_likelihood = [type_likelihood[i]/line_count for i in range(type_count)]
        if is_remove_last_type:
            features.extend(type_likelihood[:-1])
        else:
            features.extend(type_likelihood)

    # Add expected values in the last column
    if len(id_block[0]) == 24:  # with expected column
        features.append(id_block[0][23])

    return features


def file_reader(input_file, output_list):
    dropped_line_count = 0
    for id_block in by_id_reader(input_file):
        id_features = id_block_process(id_block)
        if id_features is not None:
            output_list.append(id_features)
        else:
            dropped_line_count += 1
    return dropped_line_count


def f_mean(x):
    return (x[0]+x[1])/2.0


def f_std(x):
    return (x[1]-x[0])/2.0


def zero_fill(data, label):
    fill_count = 0
    for i in range(len(data)):
        if data[i] == label:
            data[i] = 0.0
            fill_count += 1
    return fill_count


type_list = [[[5, 50], [0.97, 1.01], [0, 6], [0, 1]],
             [[40, 60], [0.95, 1], [0.5, 8], [1, 5]],
             [[10, 50], [0.92, 1.01], [2.5, 7], [0, 2]],
             [[45, 80], [0.75, 1], [-0.3, 4.5], [-2, 10]],
             [[30, 55], [0.92, 1.01], [-0.3, 2.2], [-2, 2]],
             [[5, 35], [0.97, 1.01], [-0.3, 1.5], [-1, 0.5]],
             [[25, 50], [0.88, 0.985], [-0.5, 3], [-1, 0.5]],
             [[0, 25], [0.95, 1.01], [-1, 5], [-1, 0.5]]]
gaussian_list = [[[f_mean(type_list[ii][jj]), f_std(type_list[ii][jj])] for jj in range(4)] for ii in range(8)]
type_count = len(type_list)

####################################################################################################
# parameters
id_col = 0
time_min_col = 1
dist_col = 2
reflectivity_col = range(3, 11)
to_average_col = range(3, 23)
# range(3, 23): include reflectivity in addition to precipitation calculated with it
# range(11, 23): does not include reflectivity again
to_count_data_col = range(3, 23)
max_precipitation = 70.0
missing_label = -999.0
drop_empty_block = True
is_add_precipitation_type = True
is_simple_precipitation_type = False
is_remove_last_type = False  # Make sure columns are linearly independent
is_test = False  # Generating test features or training features
is_csv = False
random_seeds = [1, 13]
output_folder = 'data'

###############################################################################
# data_list Columns:
# 0 id
# 1 observation line count
# 2 valid data count
# 3 valid data percentage
# 4 distance
# 5-20 reflectivity->precipitation, 5 is time-averaged baseline, 6 is simple averaged baseline
# 21-60 all radar data including reflectivity, time averaged followed by simple averaged
# 60-91 precipitation type likelihoods
# 92(does not exist in test file) expected precipitation

# To do: remove dist
# To do: change max_precipitation
# fill empty field but not for completely empty observations

if is_test:
    in_data_file = open('data/test.csv', 'r')
    out_data_file = open('data/testing_features.csv', 'w')
    drop_empty_block = False
else:
    in_data_file = open('data/train.csv', 'r')
    out_data_file = open('data/training_features.csv', 'w')
in_data_file.readline()

if is_add_precipitation_type:
    if is_simple_precipitation_type:
        ref_col = [3]
        rho_hv_col = [11]
        zdr_col = [15]
        kdp_col = [19]
    else:
        ref_col = [3, 5, 7, 9]
        rho_hv_col = [11, 13]*2
        zdr_col = [15, 17]*2
        kdp_col = [19, 21]*2
else:
    ref_col = list()
    rho_hv_col = list()
    zdr_col = list()
    kdp_col = list()

t0 = time.clock()
data_list = list()
dropped_line = file_reader(in_data_file, data_list)
print("Feature generation completed in {:.1f} minutes. {:d} observations dropped."
      .format((time.clock()-t0)/60.0, dropped_line))
in_data_file.close()

if is_csv:
    t0 = time.clock()
    for sample in enumerate(data_list):
        out_str = ",".join([str(item) for item in sample])
        out_data_file.write(out_str+'\n')
    out_data_file.close()
    print("CSV file generated in {:.1f} minutes.".format((time.clock()-t0)/60.0))
else:
    if is_test:
        t0 = time.clock()
        data_test = np.array(data_list)
        file_handle = open(output_folder+'/testing_data', 'w')
        pickle.dump([data_test], file_handle)
        file_handle.close()
        print("Testing data files generated in {:.1f} minutes.".format((time.clock()-t0)/60.0))
    else:
        t0 = time.clock()
        data_train = np.array(data_list)
        X0 = data_train[:, 1:-1]
        y0 = data_train[:, -1]
        y_base0 = data_train[:, 5]
        y_base_simple0 = data_train[:, 6]
        zero_fill_count = zero_fill(y_base0, missing_label)
        zero_fill(y_base_simple0, missing_label)
        print("Finished zero filling {:d} missing numbers.".format(zero_fill_count))

        for random_seed in random_seeds:
            X, y, y_base, y_base_simple = shuffle(X0, y0, y_base0, y_base_simple0, random_state=random_seed)
            offset = np.floor(X.shape[0] * 0.8)
            X_train, y_train = X[:offset], y[:offset]
            X_test, y_test, y_base_test, y_base_simple_test = \
                X[offset:], y[offset:], y_base[offset:], y_base_simple[offset:]

            file_handle = open(output_folder+'/training_data_4cv' + str(random_seed), 'w')
            pickle.dump([X_train, y_train, X_test, y_test, y_base_test, y_base_simple_test], file_handle)
            file_handle.close()
            file_handle = open(output_folder+'/training_data_4test' + str(random_seed), 'w')
            pickle.dump([X, y], file_handle)
            file_handle.close()

        print("Training data files generated in {:.1f} minutes.".format((time.clock()-t0)/60.0))
