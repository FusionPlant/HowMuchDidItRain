import os
import sys
import random
import json
import numpy as np
import numpy.linalg as linalg

in_file = open('train_norm.csv')

# init w
FEA_LEN = 21 + 1
w = [0] * FEA_LEN
for i in range(FEA_LEN):
    w[i] = random.random()

#print w

# prepre for gradient descent
# a1 = X' * X
# a2 = X' * y
in_file.seek(0)
sample_num = 0

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
            a2[i] += data[i] * data[22]
        #if line_num == 100000:
        #    print a2
        #    sys.exit()

    # a1 is symmetric
    for i in range(FEA_LEN):
        for j in range(i, FEA_LEN):
            a1[j][i] = a1[i][j];

    mfile = open(mpath, 'w')
    json.dump([a1, a2], mfile)
    mfile.close()
else:
    [a1, a2] = json.load(open(mpath))

#print "a2:"
#print a2
#sys.exit()

eta = 0.01
lam = 100
for i in range(FEA_LEN):
    a1[i][i] += lam

a1_inv = linalg.pinv(np.matrix(a1))

w = a1 * np.matrix.transpose(np.matrix(a2))
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
    err += abs(pre - data[22]) 
    if line_num == 100:
        print float(err / cnt)
        break

print("NAE: {0}".format(float(err) / cnt))

in_file.close()
