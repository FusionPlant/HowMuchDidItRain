import dask.dataframe as dd
import pandas as pd
import numpy as np
import sys, time

in_file="data/train.csv"
#in_file="data/train.sample.csv"

#out_file="data/sample_solution.csv"
#out_file="dask.csv"
out_file="train_dask.csv"

start_ts = time.time()

data = dd.read_csv(in_file, chunkbytes=100000000)
data = data.set_index('Id')

print "Finish reading file - {}".format(time.time() - start_ts)

def marshall_palmer(ref, minutes_past):
    #print "Estimating rainfall from {0} observations".format(len(minutes_past))
    # how long is each observation valid?
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in xrange(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    sum = 0
    for dbz, hours in zip(ref, valid_time):
        # See: https://en.wikipedia.org/wiki/DBZ_(meteorology)
        if np.isfinite(dbz):
            mmperhr = pow(pow(10, dbz/10)/200, 0.625)
            sum = sum + mmperhr * hours
    return sum

row_cnt = 0

def myfunc(hour):
    global row_cnt
    row_cnt += 1
    if row_cnt % 10000 == 0:
        print row_cnt

    #rowid = hour['Id'].iloc[0]
    # sort hour by minutes_past
    #hour = hour.sort('minutes_past', ascending=True)
    hour = hour.sort_values('minutes_past', ascending=True)
    #est = marshall_palmer(hour['Ref'], hour['minutes_past'])

    cols = hour.columns.values
    rs = []
    for i in range(2, 10):
        rs.append(marshall_palmer(hour[cols[i]], hour['minutes_past']))
    rs.append(hour['Expected'].iloc[0])
    #return est
    return pd.Series(np.array(rs))

all_cols = data.columns
cols = list(all_cols[2:10])
cols.append(all_cols[-1])

estimates = data.groupby(data.index).apply(myfunc, columns= cols)
print "Finish computing - {}".format(time.time() - start_ts)

estimates.to_csv(out_file, header=True)
print "Finish output - {}".format(time.time() - start_ts)
