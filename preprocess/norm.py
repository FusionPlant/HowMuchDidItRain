import os
import sys


in_file = open('train_merge.csv')
out_file = open('train_norm.csv', 'w')

su = [0] * 21
cnt = [0] * 21
for line_num, line in enumerate(in_file):
    data = line.split(',')
    for i in range(1, len(data) - 1):
        if data[i] != '':
            cnt[i-1] += 1
            su[i-1] += float(data[i])

# reset file pointer
in_file.seek(0)

avg = [0] * 21
for i in range(len(su)):
    avg[i] = su[i] / float(cnt[i])
print avg

sig = [0] * 21
for line_num, line in enumerate(in_file):
    data = line.split(',')
    for i in range(1, len(data) - 1):
        if data[i] != '':
            sig[i-1] += (float(data[i]) - avg[i-1]) ** 2

for i in range(len(su)):
    sig[i] = sig[i] / float(cnt[i])
print sig


# normlize
in_file.seek(0)

for line_num, line in enumerate(in_file):
    data = line.split(',')
    for i in range(1, len(data) - 1):
        if data[i] != '':
            # normalize
            data[i] = str((float(data[i]) - avg[i-1]) / sig[i-1])
        else:
            # fill with 0
            data[i] = '0'
    out_file.write(','.join(data))


in_file.close()
out_file.close()
