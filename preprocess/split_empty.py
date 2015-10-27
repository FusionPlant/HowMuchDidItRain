import os
import sys

train_path = 'data/train.csv'
#train_path = 'data/train.sample.csv'
in_file = open(train_path)
in_file.readline()

buf = 1000000
empty_file = open('train_empty.csv', 'w', buf)
train_file = open('train_non_empty.csv', 'w', buf)

is_empty = True
obs = []
for i, line in enumerate(in_file):
    data = line.split(',')
    is_empty = True
    for i in range(3, 23):
        if data[i] != '':
            is_empty = False
            break

    if is_empty:
        empty_file.write(line)
    else:
        train_file.write(line)


in_file.close()
empty_file.close()
train_file.close()
