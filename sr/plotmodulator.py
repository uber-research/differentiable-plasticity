import numpy as np; import matplotlib.pyplot as plt          

c = np.load('cueshown0.dat.npy'); r = np.load('rewardsprevstep0.dat.npy') ; m = np.load('modulator0.dat.npy')

params = {'legend.fontsize': 'x-large',
               'axes.labelsize': 'x-large',
                'axes.titlesize':'x-large',
                'xtick.labelsize':'x-large',
                'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

fig = plt.figure(figsize=(13,10))

for numgraph in range(c.shape[0]):
    finalgraph=0
    if numgraph == c.shape[0] - 1:
        finalgraph=1
    ax1 = plt.subplot(c.shape[0]+1, 1, numgraph+1)
    if numgraph == 0:
        ax1.set_title('Retroactive neuromodulation')
    z = np.zeros((6, c[numgraph].size))

    for nn in range(c[numgraph].size):
        z[np.int(c[numgraph][nn]+1), nn]=1
    if finalgraph:
        ax1.set_xlabel('Timestep')
    ax1.set_xlim(-.5,120.5)
    ax1.set_ylim(-.5,5.5)

    ax1.imshow(1-z, cmap='gray',clim=(-1,1), aspect='auto')
    ax1.set_yticks([0,1,2,3,4,5])
    ax1.set_yticklabels(labels=["No cue", "Cue 1", "Cue 2", "Cue 3", "Cue 4", "Response cue"])
    
    ax2 = ax1.twinx()
    ax2.set_ylim(-1,1)
    ax2.plot(m[numgraph], label="Modulator", lw=2)
    ax2.plot(r[numgraph], label="Reward", lw=2)
    ax2.plot(np.zeros_like(r[numgraph]), 'k:')
    if finalgraph:
        ax2.legend(loc='upper left', bbox_to_anchor=(0, -.2))


plt.tight_layout()  # Too tight!
#fig.subplots_adjust(hspace=0.5)

plt.show()
