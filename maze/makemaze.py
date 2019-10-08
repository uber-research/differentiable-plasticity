# Not used for the current version.

import numpy as np

def genmaze(size, nblines):
    nbiter = 0
    N = size
    m = np.zeros((N,N))
    m[0,:] = 1
    m[-1,:] = 1
    m[:,0] = 1
    m[:, -1]= 1

    MAXLINES = nblines
    mynblines = 0
    while True:
        nbiter += 1
        if nbiter == 10000:
            #print("Inf. loop in maze gen, resetting map & retrying") # If that happens too often parameters are probably not good
            #print("IL") # If that happens too often parameters are probably not good
            m.fill(0)
            m[0,:] = 1;   m[-1,:] = 1;  m[:,0] = 1;  m[:, -1]= 1;
            nbiter = 0
            mynblines = 0
        rcol = 1 + np.random.randint(N-1)
        rrow = 1 + np.random.randint(N-1)
        if m[rrow, rcol] == 1:
            continue
        ori = np.random.randint(2)
        if ori == 0: # horizontal
            start = rcol
            while m[rrow, start] == 0:
                start -= 1
            end = rcol
            while m[rrow, end] == 0:
                end += 1
            end -= 1
            start += 1
            if end-start < 4:
                continue
            if np.sum(m[rrow-1, start:end+1]) > 0 or np.sum(m[rrow+1, start:end+1]) > 0:
                continue
            if np.sum(m[rrow-2, start:end+1]) > 0 or np.sum(m[rrow+2, start:end+1]) > 0:
                continue
            m[rrow, start:end+1] = 1
            opening = np.random.randint(start+1, end-1)
            m[rrow, opening] = 0
            m[rrow, opening+1] = 0
            mynblines += 1
        elif ori == 1: # vertical
            start = rrow
            while m[start, rcol] == 0:
                start -= 1
            end = rrow
            while m[end, rcol] == 0:
                end += 1
            end -= 1
            start += 1
            if end-start < 5:
                continue
            if np.sum(m[start:end+1, rcol-1]) > 0 or np.sum(m[start:end+1, rcol+1]) > 0:
                continue
            if np.sum(m[start:end+1, rcol-2]) > 0 or np.sum(m[start:end+1, rcol+2]) > 0:
                continue
            m[start:end+1, rcol] = 1
            opening = np.random.randint(start+1, end-1)
            m[opening, rcol] = 0
            m[opening+1, rcol] = 0
            mynblines += 1
        if mynblines >= MAXLINES:
            break
    return m



if __name__ == '__main__':
    
    #M = genmaze(size=50, nblines=8)
    M = genmaze(size=15, nblines=4)
    #M = genmaze(size=19, nblines=4)
    print(M)


