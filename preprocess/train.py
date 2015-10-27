import os
import sys
import random
import json
import numpy as np
import numpy.linalg as linalg

in_file = open('train_norm.csv')

# prepre for gradient descent
# a1 = X' * X
# a2 = X' * y
data = in_file.readline().split(',')
FEA_LEN = len(data) - 1
y_idx = len(data) - 1

in_file.seek(0)
mpath = 'gd.json'
if not os.path.isfile(mpath):
    a1 = [[0] * FEA_LEN for i in range(FEA_LEN)]
    a2 = [0] * FEA_LEN

    for line_num, line in enumerate(in_file):
        data = [float(z) for z in line.split(',')]
        data[0] = 1

        for i in range(FEA_LEN):
            for j in range(i, FEA_LEN):
                a1[i][j] += data[i] * data[j]
            a2[i] += data[i] * data[y_idx]
        #if line_num == 100000:
        #    print a2
        #    sys.exit()

    # a1 is symmetric
    for i in range(FEA_LEN):
        for j in range(i-1):
            a1[i][j] = a1[j][i];

    mfile = open(mpath, 'w')
    json.dump([a1, a2], mfile)
    mfile.close()
else:
    [a1, a2] = json.load(open(mpath))

#print "a2:"
#print a2
#sys.exit()

#lam = 100
#for i in range(FEA_LEN):
#    a1[i][i] += lam

a1_inv = linalg.pinv(np.matrix(a1))

w = a1_inv * np.matrix.transpose(np.matrix(a2))
w = np.array(w).reshape(-1,).tolist()

print "w:"
print w
#sys.exit()

# MAE
in_file.seek(0)
err = 0
cnt = 0
for line_num, line in enumerate(in_file):
    data = [float(z) for z in line.split(',')]
    cnt += 1

    pre = w[0]
    for i in range(1, FEA_LEN):
        pre += w[i] * data[i]
    err += abs(pre - data[y_idx]) 
    #if line_num == 100:
    #    print float(err / cnt)
    #    break

print("NAE: {0}".format(float(err) / cnt))

in_file.close()
