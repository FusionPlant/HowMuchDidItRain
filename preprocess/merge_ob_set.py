import os
import sys

in_file = open('train_non_empty.csv')
#in_file = open('train.sample.csv')
out_file = open('train_merge.csv', 'w')

id = None
num = 20
for line_num, line in enumerate(in_file):
    data = line.split(',')
    if id != data[0] or id is None:
        if id is not None:
            # clear old
            out_line = [data[0], data[2]]
            for i in range(num):
                if cnt[i] == 0:
                    out_line.append('')
                else:
                    out_line.append(str(su[i] / float(cnt[i])))
            out_line.append(data[23])
            out_file.write(",".join(out_line))

        # start new
        id = data[0]
        cnt = [0] * num
        su = [0] * num

    for i in range(3, 23):
        if data[i] != '':
            j = i - 3
            cnt[j] += 1
            su[j] += float(data[i])

in_file.close()
out_file.close()
    
