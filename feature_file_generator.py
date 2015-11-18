import numpy as np


# Output an float64 array from input string, with np.nan indicating missing data
def csv_line_reader(line_str):
    line_array = np.fromstring(line_str, sep=',')
    for i in range(len(line_array)):
        if line_array[i] == -1:
            line_array[i] = np.nan
    return line_array


# Output a generator, whose each item is an array, which contains all data from a same id
def by_id_reader(input_file):
    data_id = None
    id_block = None

    for line_num, line in enumerate(input_file):
        if line_num % 20000 == 0:
            print "Reading input line {:d}".format(line_num)

        line_array = csv_line_reader(line)
        if np.isnan(line_array[0]):  # should never happen #debug#
            print "Error: Missing id in line {:d}".format(line_num)
        if line_array[0] != data_id:
            data_id = line_array[0]
            if id_block is not None:
                yield id_block
            id_block = np.array([line_array])
        else:
            id_block = np.append(id_block, [line_array], 0)

    if id_block is not None:
        yield id_block


# use reflectivity array to predict precipitation in one hour in mm
def predict_with_reflectivity(reflectivity, time_interval_min):
    precipitation_mm = 0.0
    minutes_past = 0
    last_mm_per_hr = 0
    for dbz, minute in np.array([reflectivity, time_interval_min]).T:
        # See: https://en.wikipedia.org/wiki/DBZ_(meteorology)
        # to be updated #
        minutes_past += minute
        if not np.isnan(dbz):
            mm_per_hr = pow(pow(10.0, dbz/10.0)/200, 0.625)
            precipitation_mm += mm_per_hr * minutes_past / 60.0
            minutes_past = 0
            last_mm_per_hr = mm_per_hr
    return precipitation_mm + last_mm_per_hr * minutes_past / 60.0


# average data from value array
# to be updated #
def average_values(value_all, time_interval_min):
    averaged_value = 0.0
    value_count = 0
    for value in value_all:
        if not np.isnan(value):
            averaged_value += value
            value_count += 1
    if value_count > 0:
        averaged_value = averaged_value / value_count
    return averaged_value


# for each data block from a same id, process it and return one line of data in array form
def id_block_process(id_block):
    id_col = 0
    time_min_col = 1
    dist_col = 2
    to_average_col = range(11, 23)  # range(3, 23)
    reflectivity_col = range(3, 11)
    line_count = len(id_block)
    if line_count <= 0:
        print "Error: Empty block"  # should not happen #debug#
        return None

    features = np.array([])

    # Add id as feature #debug#
    features = np.append(features, id_block[0, id_col])

    # Add distance as feature # to be updated #
    features = np.append(features, id_block[0, dist_col])

    # Add number of columns as feature # to be updated #
    features = np.append(features, [line_count])

    # Time interval calculation
    time_min = [0]*line_count
    for i in range(line_count):
        time_min[i] = int(id_block[i, time_min_col])  # time data should never be missing

    time_interval_min = [0]*line_count
    # to be updated #
    time_interval_min[0] = time_min[0]
    for i in range(1, line_count-1):
        time_interval_min[i] = time_min[i]-time_min[i-1]
    time_interval_min[-1] = 60-sum(time_interval_min[:-1])

    # Add precipitation predictions as features
    for j in reflectivity_col:
        reflectivity = np.array([])
        for i in range(line_count):
            reflectivity = np.append(reflectivity, id_block[i, j])
        features = np.append(features,
                             predict_with_reflectivity(reflectivity, time_interval_min))

    # Add averaged values as features
    for j in to_average_col:
        value = np.array([])
        for i in range(line_count):
            value = np.append(value, id_block[i, j])
        features = np.append(features,
                             average_values(value, time_interval_min))

    # Add expected values in the last column
    if len(id_block[0]) == 24:  # with expected column
        features = np.append(features, id_block[0, 23])

    return features


def file_reader(input_file, output_file):
    id_block_all = by_id_reader(input_file)
    for block_num, id_block in enumerate(id_block_all):
        # if block_num % 5000 == 0:
        #     print "Writing output line {:d}".format(block_num)

        id_features = id_block_process(id_block)
        out_str = np.array2string(id_features, separator=',', max_line_width=999,
                                  formatter={'float':lambda x: '{:.8f}'.format(x)})[1:-1]+'\n'
        output_file.write(out_str)


in_data_file = open('data/train.csv')
out_data_file = open('features.csv', 'w')

# Remove the header
in_data_file.readline()

file_reader(in_data_file, out_data_file)

in_data_file.close()
out_data_file.close()
