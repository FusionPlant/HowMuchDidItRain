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


# convert time in minutes into time interval in minutes
def time_to_time_interval(time_min):
    line_count = len(time_min)
    time_interval_min = [0]*line_count
    for i in range(line_count-1):
        time_interval_min[i] = (time_min[i]+time_min[i+1])/2.0
    time_interval_min[-1] = 60.0
    for i in range(line_count-1, 0, -1):
        time_interval_min[i] -= time_interval_min[i-1]
    return time_interval_min


# use reflectivity array to predict precipitation in one hour in mm
# See: https://en.wikipedia.org/wiki/DBZ_(meteorology)
def predict_with_reflectivity(reflectivity, time_interval_min, missing_label):
    for dbz in reflectivity:
        if dbz is not None:
            mm_per_hr = pow(pow(10.0, dbz/10.0)/200, 0.625)  # first not None value
            break
    else:
        return missing_label

    precipitation_mm = 0.0
    for dbz, minute in zip(reflectivity, time_interval_min):
        if dbz is not None:
            mm_per_hr = pow(pow(10.0, dbz/10.0)/200, 0.625)
        precipitation_mm += mm_per_hr * minute / 60.0
    return precipitation_mm


# average data from value array
def average_values(value_all, time_interval_min, missing_label):
    averaged_value = 0.0
    for value in value_all:
        if value is not None:
            valid_value = value  # first not None value
            break
    else:
        return missing_label

    for value, minute in zip(value_all, time_interval_min):
        if value is not None:
            valid_value = value
        averaged_value += valid_value * minute
    return averaged_value / 60.0


# count number of valid data in each data block from a same id
def count_data(id_block, to_count_data_col):
    line_count = len(id_block)
    data_count = 0
    for i in range(line_count):
        for j in to_count_data_col:
            if id_block[i][j] is not None:
                data_count += 1
    return data_count


# for each data block from a same id, process it and return one line of data in list form
def id_block_process(id_block, max_mm, missing_label):
    # id_col = 0
    time_min_col = 1
    dist_col = 2
    reflectivity_col = range(3, 11)
    to_average_col = range(11, 23)
    # range(3, 23): include reflectivity in addition to precipitation calculated with it
    # range(11, 23): does not include reflectivity again
    to_count_data_col = range(3, 23)
    line_count = len(id_block)

    features = list()

    # # Add id as feature
    # features.append(id_block[0][id_col])

    # Add number of columns as feature
    features.append(line_count)

    # Add number of non-missing data as feature
    features.append(count_data(id_block, to_count_data_col))

    # Add distance as feature
    features.append(id_block[0][dist_col])

    # Time interval calculation
    time_min = [0]*line_count
    for i in range(line_count):
        time_min[i] = int(id_block[i][time_min_col])  # time data should never be missing
    time_interval_min = time_to_time_interval(time_min)

    # Add precipitation predictions as features
    for j in reflectivity_col:
        reflectivity = list()
        for i in range(line_count):
            reflectivity.append(id_block[i][j])
        features.append(predict_with_reflectivity(reflectivity, time_interval_min, missing_label))

    # Add averaged values as features
    for j in to_average_col:
        value = list()
        for i in range(line_count):
            value.append(id_block[i][j])
        features.append(average_values(value, time_interval_min, missing_label))

    # Add expected values in the last column
    if len(id_block[0]) == 24:  # with expected column
        # Throw away >70mm data
        if id_block[0][23] > max_mm:
            return None
        features.append(id_block[0][23])

    return features


def file_reader(input_file, output_file, max_mm, missing_label):
    id_block_all = by_id_reader(input_file)
    for block_num, id_block in enumerate(id_block_all):
        # if block_num % 5000 == 0:
        #     print "Writing output line {:d}".format(block_num)

        id_features = id_block_process(id_block, max_mm, missing_label)
        if id_features is not None:
            out_str = ",".join([str(item) for item in id_features])
            output_file.write(out_str+'\n')


in_data_file = open('data/train.csv')
out_data_file = open('features.csv', 'w')
max_precipitation = 70.0
missing_data_label = -999.0  # used to be 0.0

# Remove the header
in_data_file.readline()

file_reader(in_data_file, out_data_file, max_precipitation, missing_data_label)

in_data_file.close()
out_data_file.close()
