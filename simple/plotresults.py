# Code to plot learning curves
#
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

fnames = glob.glob('./tmp/loss_simple_*.txt')
#fnames = glob.glob('./tmp/loss_api_*.txt')
#fnames = glob.glob('./tmp/loss_fixed_*.txt')


plt.ion()
plt.rc('font', size=12)
plt.figure(figsize=(5,4))  # Smaller figure = relative larger fonts


fulllosses=[]
losses=[]
lgts=[]
for fn in fnames:
    z = np.loadtxt(fn)
    lgts.append(len(z))
    fulllosses.append(z)
minlen = min(lgts)
for z in fulllosses:
    losses.append(z[:minlen])

losses = np.array(losses)
meanl = np.mean(losses, axis=0)
stdl = np.std(losses, axis=0)

highl = np.max(losses, axis=0)
lowl = np.min(losses, axis=0)
#highl = meanl+stdl
#lowl = meanl-stdl

xx = range(len(meanl))

# xticks and labels
xt = range(0, len(meanl), 50)
xtl = [str(10*i) for i in xt]

plt.fill_between(xx, lowl, highl, color='blue', alpha=.5)
plt.plot(meanl, color='blue')
#plt.xlabel('Loss (sum square diff. b/w final output and target)')
plt.xlabel('Number of Episodes')
plt.ylabel('Loss')
plt.xticks(xt, xtl)
plt.tight_layout()


