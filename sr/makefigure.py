import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy
from scipy import stats

colorz = ['g', 'orange', 'r', 'b', 'c', 'm', 'y', 'k']



groupnames = glob.glob('./tmp/loss_SRB_addpw_2_alg_A3C_bent_0.1_blossv_0.1_bs_30_bv_0.1_clamp_0_cs_20_da_tanh_eplen_120_eps_1e-06_fm_1_gc_2.0_gr_0.9_hs_200_is_0_l2_0.0_lr_0.0001_nbiter_200000_ni_4_nu_0.1_pe_500_pf_0.0_rew_1.0_rule_hebb_type_*_wp_0.0_rngseed_0.txt')

#Previous:
#groupnames = glob.glob('./tmp8/loss_*eplen_251*densize_200*absize_11_*ndstart_1*rngseed_1.txt')  
#groupnames = glob.glob('./tmp8/loss_*eplen_251*densize_200*absize_11_*ndstart_1*rngseed_1.txt')  


#groupnames = glob.glob('./tmp/loss_*new*eplen_251*rngseed_0.txt')  
#groupnames = glob.glob('./tmp/loss_*new*eplen_250*rngseed_0.txt')  

plt.rc('font', size=14)


def my_mavg(x, N):
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
        if True:
            if "seed_11" in fn:
                continue
            if "seed_12" in fn:
                continue
            if "seed_13" in fn:
                continue
            if "seed_14" in fn:
                continue
            if "seed_15" in fn:
                continue


        z = np.loadtxt(fn)
        
        z = z[::10] # Decimation - speed things up!
        #z = my_mavg(z, 20)  # For each run, we average the losses over K successive (decimated) episodes - otherwise figure is unreadable due to noise!


        z = z[:1801]
        
        #if len(z) < 9000:
        #    print(fn)
        #    continue
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
    #xt = range(0, len(meanl), 2000)
    xt = range(0, 1801, 500)
    xtl = [str(10 * 10 * i) for i in xt]   # Because of decimation above, and only every 10th loss is recorded in the files

    if "plastic" in groupname:
        lbl = "Non-modulated plastic"
    elif "modplast" in groupname:
        lbl = "Simple modulation"
    elif "modul" in groupname:
        lbl = "Retroactive modulation"
    elif "rnn" in groupname:
        lbl = "Non-plastic"
    else:
        raise ValueError("Which type?")

    #plt.plot(my_mavg(meanl, 100), label=g) #, color='blue')
    #plt.fill_between(xx, lowl, highl,  alpha=.2)
    #plt.fill_between(xx, q1l, q3l,  alpha=.1)
    #plt.plot(meanl) #, color='blue')
    ####plt.plot(my_mavg(medianl, 100), label=g) #, color='blue')  # my_mavg changes the number of points !
    #plt.plot(my_mavg(q1l, 100), label=g, alpha=.3) #, color='blue')
    #plt.plot(my_mavg(q3l, 100), label=g, alpha=.3) #, color='blue')
    #plt.fill_between(xx, q1l, q3l,  alpha=.2)
    #plt.plot(medianl, label=g) #, color='blue')
   
    AVGSIZE = 20
    
    xlen = len(my_mavg(q1l, AVGSIZE))
    plt.fill_between( range(xlen), my_mavg(q1l, AVGSIZE), my_mavg(q3l, AVGSIZE),  alpha=.2, color=colorz[poscol % len(colorz)])
    plt.plot(my_mavg(medianl, AVGSIZE), color=colorz[poscol % len(colorz)], label=lbl)  # my_mavg changes the number of points !
    
    #xlen = len(my_mavg(meanl, AVGSIZE))
    #plt.plot(my_mavg(meanl, AVGSIZE), label=g, color=colorz[poscol % len(colorz)])  # my_mavg changes the number of points !
    #plt.fill_between( range(xlen), my_mavg(meanl - cil, AVGSIZE), my_mavg(meanl + cil, AVGSIZE),  alpha=.2, color=colorz[poscol % len(colorz)])
    
    poscol += 1
    
    #plt.fill_between( range(xlen), my_mavg(lowl, 100), my_mavg(highl, 100),  alpha=.2, color=colorz[numgroup % len(colorz)])

    #plt.plot(my_mavg(losses[0], 1000), label=g, color=colorz[numgroup % len(colorz)])
    #for curve in losses[1:]:
    #    plt.plot(my_mavg(curve, 1000), color=colorz[numgroup % len(colorz)])

ps = []
# Adapt for varying lengths across groups
#for n in range(0, alllosses[0].shape[1], 3):
for n in range(0, minminlen):
    ps.append(scipy.stats.ranksums(alllosses[0][:,n], alllosses[1][:,n]).pvalue)
ps = np.array(ps)
print(np.mean(ps[-500:] < .05), np.mean(ps[-500:] < .01))

plt.legend(loc='best', fontsize=14)
#plt.xlabel('Loss (sum square diff. b/w final output and target)')
plt.xlabel('Number of Episodes')
plt.ylabel('Reward')
plt.xticks(xt, xtl)
#plt.tight_layout()



