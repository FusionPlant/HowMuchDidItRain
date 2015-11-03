import sys, os
import csv

#in_file = open('train_merge.csv', 'rb')
#out_file = open('train_prediction.csv', 'wb')

in_file = open('test_merge.csv', 'rb')
out_file = open('test_prediction.csv', 'wb')

reader = csv.reader(in_file)

writer = csv.writer(out_file)
writer.writerow(['Id', 'Expected'])

for row in reader:
    if row[3] > 0:
        expected = row[3]
    else:
        # Ref
        for i in [6, 5, 4]:
            if row[i] > 0:
                expected = row[i]
                break

        if row[i] > 0:
            continue

        # RefComposite
        for i in [7, 10, 9, 8]:
            if row[i] > 0:
                expected = row[i]
                break

    rs = [row[0], expected]
    writer.writerow(rs)


in_file.close()
out_file.close()
