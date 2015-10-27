* data directory

All input & output data directory/files are hardcoded. Please adjust accordingly.

* run order

split_empty.py - split empty and non-empty rows
merge_ob_set.py - merge observation with the same ID
norm.py - normalize data, replace empty with 0
train.py - train and estimate error on training data(better on validation data)
