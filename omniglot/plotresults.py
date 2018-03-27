import numpy as np
import glob
import matplotlib.pyplot as plt

groupnames = glob.glob('./tmp/loss_*lr_0.0001*nbclasses_5*rngseed_0.txt')
#fnames = glob.glob('./tmp/loss_api_*.txt')
#fnames = glob.glob('./tmp/loss_fixed_*.txt')

def mavg(x, N):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  return (cumsum[N:] - cumsum[:-N]) / N

plt.ion()
#plt.figure(figsize=(5,4))  # Smaller figure = relative larger fonts
plt.figure()

for groupname in groupnames:
    g = groupname[:-6]+"*"
    fnames = glob.glob(g)
    fulllosses=[]
    losses=[]
    lgts=[]
    for fn in fnames:
        if "COPY" in fn:
            continue
        if "00.tx" in fn:
            continue
        z = np.loadtxt(fn)
        z = z[::10] # Decimation
        lgts.append(len(z))
        fulllosses.append(z)
    minlen = min(lgts)
    for z in fulllosses:
        losses.append(z[:minlen])

    losses = np.array(losses)
    
    meanl = np.mean(losses, axis=0)
    stdl = np.std(losses, axis=0)

    medianl = np.median(losses, axis=0)
    q1l = np.percentile(losses, 25, axis=0)
    q3l = np.percentile(losses, 75, axis=0)
    
    highl = np.max(losses, axis=0)
    lowl = np.min(losses, axis=0)
    #highl = meanl+stdl
    #lowl = meanl-stdl

    xx = range(len(meanl))

    # xticks and labels
    xt = range(0, len(meanl), 2000)
    #xtl = [str(200*i) for i in xt]  #200 = 20 episode per loss saving, plus the decimation above
    xtl = [str(1000*i) for i in xt]  #1000 = 100 episode per loss saving, plus the decimation above

    #plt.plot(mavg(meanl, 100), label=g) #, color='blue')
    #plt.fill_between(xx, lowl, highl,  alpha=.2)
    #plt.fill_between(xx, q1l, q3l,  alpha=.3)
    #plt.plot(meanl) #, color='blue')
    plt.plot(mavg(medianl, 100), label=g) #, color='blue')  # mavg changes the number of points !
    #plt.plot(mavg(q1l, 100), label=g, alpha=.3) #, color='blue')
    #plt.plot(mavg(q3l, 100), label=g, alpha=.3) #, color='blue')
    #plt.fill_between(xx, q1l, q3l,  alpha=.2)
    #plt.plot(medianl, label=g) #, color='blue')

plt.legend()
#plt.xlabel('Loss (sum square diff. b/w final output and target)')
plt.xlabel('Number of Episodes')
plt.ylabel('Loss')
plt.xticks(xt, xtl)
plt.tight_layout()


