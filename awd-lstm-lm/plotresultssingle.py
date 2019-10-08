import numpy as np
import matplotlib.pyplot as plt
import glob


fns = glob.glob('./HDFS/ptb/results_*.txt')

plt.figure()

numcurve = 0
for (ii, fn) in enumerate(fns):
    #if 'B_' not in fn and 'MYLSTM' not in fn:
    #    continue 
    if 'rngseed' in fn:
        if 'seed0' not in fn:
            continue
    if 'agdiv10'  in fn:
        continue
    #if '44' not in fn:
    #    continue
    print(fn)
    #if 'perneuron'  in fn:
    #    continue
    numcurve += 1
    if numcurve > 20:
        ls = ':'
    elif numcurve > 10:
        ls = '--'
    else:
        ls = '-'
    #z = np.loadtxt(fn)
    z = np.exp(np.loadtxt(fn))
    plt.plot(z, label=fn,ls=ls)

plt.legend(loc='upper right')
plt.show()


