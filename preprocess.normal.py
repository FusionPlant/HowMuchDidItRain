import sys, os

in_file = open('data/train.csv')
#in_file = open('data/train.sample.csv')
out_file = open('train_merge.csv', 'w')

#in_file = open('data/test.csv')
#out_file = open('data/dask_solution.csv', 'w')

#in_file = open('data/test.csv')
##in_file = open('train.sample.csv')
#out_file = open('test_merge.csv', 'w')

header = in_file.readline().rstrip().split(',')
is_test = False if header[-1] == 'Expected' else True

#is_test = True
#is_test = False

def transfer_ref(ref, minutes_past):
    #print "Estimating rainfall from {0} observations".format(len(minutes_past))
    # how long is each observation valid?
    valid_time = [0] * len(minutes_past)
    valid_time[0] = minutes_past[0]
    for n in range(1, len(minutes_past)):
        valid_time[n] = minutes_past[n] - minutes_past[n-1]
    valid_time[-1] = valid_time[-1] + 60 - sum(valid_time)
    #valid_time = valid_time / 60.0

    for i in range(len(valid_time)):
        valid_time[i] = valid_time[i] / 60.0

    # rs up rainrate * validtime
    rs = 0
    for dbz, hours in zip(ref, valid_time):
        # See: https://en.wikipedia.org/wiki/DBZ_(meteorology)
        #if np.isfinite(dbz):
        if dbz is not None:
            mmperhr = pow(pow(10, dbz/10)/200, 0.625)
            rs = rs + mmperhr * hours
    return rs

id = None
dis = None
cnt = None
su = None
num = 20
ob_list = None
for line_num, line in enumerate(in_file):
    if line_num % 100000 == 0:
        print line_num
        #break

    data = line.rstrip().split(',')
    if id != data[0] or id is None:
        if id is not None:
            # clear old
            out_line = [id, dis, len(ob_list)]
            if len(ob_list) > 0:
                # minutes
                mins = []
                for i in range(len(ob_list)):
                    mins.append(int(ob_list[i][1]))

                # refs
                ref_num = 8
                for ref_idx in range(ref_num):
                    ref = []
                    for i in range(len(ob_list)):
                        if ob_list[i][ref_idx + 3] != '':
                            ref.append(float(ob_list[i][ref_idx + 3]))
                        else:
                            ref.append(None)
                    out_line.append(transfer_ref(ref, mins))
            else:
                # empty data
                for i in range(num):
                    out_line.append(0)

            #for i in range(num):
            #    if cnt[i] == 0:
            #        out_line.append('')
            #    else:
            #        out_line.append(str(su[i] / float(cnt[i])))
            if not is_test:
                out_line.append(expected)

            #out_file.write(",".join([str(z) for z in out_line]))
            #out_file.write('\n')
            #out_line = [out_line[0], out_line[3]]
            out_file.write(",".join([str(z) for z in out_line]))
            out_file.write('\n')

        # start new
        id = data[0]
        dis = data[2]
        #cnt = [0] * num
        ob_list = []
        if not is_test:
            expected = data[23]

    #for i in range(3, 3 + num):
    #    if data[i].strip() != '':
    #        j = i - 3
    #        cnt[j] += 1
    #        #su[j] += float(data[i])
    #    else:
    #        emp_cnt += 1
    emp_cnt = 0
    for i in range(3, 3 + num):
        if data[i].strip() == '':
            emp_cnt += 1

    ob_list.append(data)
    #if emp_cnt != num:
    #    ob_list.append(data)

#for i in range(len(ob_list)):
#    print ob_list[i]
# clear old
out_line = [id, dis, len(ob_list)]
ref_num = 8
if len(ob_list) > 0:
    # minutes
    mins = []
    for i in range(len(ob_list)):
        mins.append(int(ob_list[i][1]))

    # refs
    for ref_idx in range(ref_num):
        ref = []
        for i in range(len(ob_list)):
            if ob_list[i][ref_idx + 3] != '':
                ref.append(float(ob_list[i][ref_idx + 3]))
            else:
                ref.append(None)
        out_line.append(transfer_ref(ref, mins))
else:
    # empty data
    for i in range(ref_num):
        out_line.append(0)

if not is_test:
    out_line.append(expected)

#out_file.write(",".join([str(z) for z in out_line]))
#out_file.write('\n')
#out_line = [out_line[0], out_line[3]]
out_file.write(",".join([str(z) for z in out_line]))
out_file.write('\n')

in_file.close()
out_file.close()
