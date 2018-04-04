# Code for making a figure
#
# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy
from scipy import stats

colorz = ['r', 'b', 'g', 'c', 'm', 'y', 'orange', 'k']

groupnames = glob.glob('./tmp/loss_*new*eplen_251*rngseed_0.txt')  
#groupnames = glob.glob('./tmp/loss_*new*eplen_250*rngseed_0.txt')  

plt.rc('font', size=14)



def mavg(x, N):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[N:] - cumsum[:-N]) / N

plt.ion()
#plt.figure(figsize=(5,4))  # Smaller figure = relative larger fonts
plt.figure()

allmedianls = []
alllosses = []
poscol = 0
minminlen = 999999
for numgroup, groupname in enumerate(groupnames):
    if "lstm" in groupname:
        continue
    g = groupname[:-6]+"*"
    print("====", groupname)
    fnames = glob.glob(g)
    fulllosses=[]
    losses=[]
    lgts=[]
    for fn in fnames:

        z = np.loadtxt(fn)
        
        z = mavg(z, 10)  # For each run, we average the losses over K successive episodes - otherwise figure is unreadable due to noise!

        z = z[::10] # Decimation - speed things up!

        z = z[:2001]
        
        if len(z) < 1000:
            print(fn)
            continue
        #z = z[:90]
        lgts.append(len(z))
        fulllosses.append(z)
    minlen = min(lgts)
    if minlen < minminlen:
        minminlen = minlen
    print(minlen)
    #if minlen < 1000:
    #    continue
    for z in fulllosses:
        losses.append(z[:minlen])

    losses = np.array(losses)
    alllosses.append(losses)
    
    meanl = np.mean(losses, axis=0)
    stdl = np.std(losses, axis=0)
    #cil = stdl / np.sqrt(losses.shape[0]) * 1.96  # 95% confidence interval - assuming normality
    cil = stdl / np.sqrt(losses.shape[0]) * 2.5  # 95% confidence interval - approximated with the t-distribution for 7 d.f. (?)

    medianl = np.median(losses, axis=0)
    allmedianls.append(medianl)
    q1l = np.percentile(losses, 25, axis=0)
    q3l = np.percentile(losses, 75, axis=0)
    
    highl = np.max(losses, axis=0)
    lowl = np.min(losses, axis=0)
    #highl = meanl+stdl
    #lowl = meanl-stdl

    xx = range(len(meanl))

    # xticks and labels
    xt = range(0, len(meanl), 500)
    xtl = [str(10 * 10 * i) for i in xt]   # Because of decimation above, and only every 10th loss is recorded in the files

    if "plastic" in groupname:
        lbl = "Plastic"
    elif "rnn" in groupname:
        lbl = "Non-plastic"

    #plt.plot(mavg(meanl, 100), label=g) #, color='blue')
    #plt.fill_between(xx, lowl, highl,  alpha=.2)
    #plt.fill_between(xx, q1l, q3l,  alpha=.1)
    #plt.plot(meanl) #, color='blue')
    ####plt.plot(mavg(medianl, 100), label=g) #, color='blue')  # mavg changes the number of points !
    #plt.plot(mavg(q1l, 100), label=g, alpha=.3) #, color='blue')
    #plt.plot(mavg(q3l, 100), label=g, alpha=.3) #, color='blue')
    #plt.fill_between(xx, q1l, q3l,  alpha=.2)
    #plt.plot(medianl, label=g) #, color='blue')
   
    AVGSIZE = 1
    
    xlen = len(mavg(q1l, AVGSIZE))
    plt.plot(mavg(medianl, AVGSIZE), color=colorz[poscol % len(colorz)], label=lbl)  # mavg changes the number of points !
    plt.fill_between( range(xlen), mavg(q1l, AVGSIZE), mavg(q3l, AVGSIZE),  alpha=.2, color=colorz[poscol % len(colorz)])
    
    #xlen = len(mavg(meanl, AVGSIZE))
    #plt.plot(mavg(meanl, AVGSIZE), label=g, color=colorz[poscol % len(colorz)])  # mavg changes the number of points !
    #plt.fill_between( range(xlen), mavg(meanl - cil, AVGSIZE), mavg(meanl + cil, AVGSIZE),  alpha=.2, color=colorz[poscol % len(colorz)])
    
    poscol += 1
    
    #plt.fill_between( range(xlen), mavg(lowl, 100), mavg(highl, 100),  alpha=.2, color=colorz[numgroup % len(colorz)])

    #plt.plot(mavg(losses[0], 1000), label=g, color=colorz[numgroup % len(colorz)])
    #for curve in losses[1:]:
    #    plt.plot(mavg(curve, 1000), color=colorz[numgroup % len(colorz)])

ps = []
# Adapt for varying lengths across groups
#for n in range(0, alllosses[0].shape[1], 3):
for n in range(0, minminlen):
    ps.append(scipy.stats.ranksums(alllosses[0][:,n], alllosses[1][:,n]).pvalue)
ps = np.array(ps)
np.mean(ps[-500:] < .05)
np.mean(ps[-500:] < .01)

plt.legend(loc='best', fontsize=14)
#plt.xlabel('Loss (sum square diff. b/w final output and target)')
plt.xlabel('Number of Episodes')
plt.ylabel('Reward')
plt.xticks(xt, xtl)
#plt.tight_layout()



