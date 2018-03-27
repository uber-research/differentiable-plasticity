import pdb 
import torch
import torch.nn as nn
from torch.autograd import Variable
import click
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
import random
import sys
import pickle
import pdb
import time
import skimage
from skimage import transform
import os
import OpusHdfsCopy
from OpusHdfsCopy import transferFileToHdfsDir, checkHdfs
import platform


import numpy as np
import matplotlib.pyplot as plt
import glob

import omniglot

from omniglot import Network



np.set_printoptions(precision=4)


defaultParams = {
    'nbclasses': 3,
    'nbshots': 1,  # Number of 'shots' in the few-shots learning
    'prestime': 5,
    'nbfeatures' : 64    ,
    'prestimetest': 3,
    'interpresdelay': 2,
    'imagesize': 31,    # 28*28
    'nbiter': 10000000,
    'learningrate': 1e-5,
    'print_every': 10,
    'rngseed':0
}
NBTESTCLASSES = 100




#ttype = torch.FloatTensor;
ttype = torch.cuda.FloatTensor;


# Generate the full list of inputs, labels, and the target label for an episode
def generateInputsLabelsAndTarget(params, imagedata, test=False):
    #print(("Input Boost:", params['inputboost']))
    #params['nbsteps'] = params['nbshots'] * ((params['prestime'] + params['interpresdelay']) * params['nbclasses']) + params['prestimetest'] 
    inputT = np.zeros((params['nbsteps'], 1, 1, params['imagesize'], params['imagesize']))    #inputTensor, initially in numpy format... Note dimensions: number of steps x batchsize (always 1) x NbChannels (also 1) x h x w 
    labelT = np.zeros((params['nbsteps'], 1, params['nbclasses']))      #labelTensor, initially in numpy format...

    patterns=[]
    if test:
        cats = np.random.permutation(np.arange(len(imagedata) - NBTESTCLASSES, len(imagedata)))[:params['nbclasses']]  # Which categories to use for this *testing* episode?
    else:
        cats = np.random.permutation(np.arange(len(imagedata) - NBTESTCLASSES))[:params['nbclasses']]  # Which categories to use for this *training* episode?
    #print("Test is", test, ", cats are", cats)
    #cats = np.array(range(params['nbclasses'])) + 10

    cats = np.random.permutation(cats)
    #print(cats)

    # We show one picture of each category, with labels, then one picture of one of these categories as a test, without label
    # But each of the categories may undergo rotation by 0, 90, 180 or 270deg
    # NOTE: We randomly assign one rotation to all the possible categories, not just the ones selected for the episode - it makes the coding simpler
    rots = np.random.randint(4, size=len(imagedata))

    #rots.fill(0)

    testcat = random.choice(cats) # select the class on which we'll test in this episode
    unpermcats = cats.copy()

    # Inserting the character images and labels in the input tensor at the proper places
    location = 0
    for nc in range(params['nbshots']):
        np.random.shuffle(cats)   # Presentations occur in random order
        for ii, catnum in enumerate(cats):
            #print(catnum)
            p = random.choice(imagedata[catnum])
            for nr in range(rots[catnum]):
                p = np.rot90(p)
            p = skimage.transform.resize(p, (31, 31))
            for nn in range(params['prestime']):
                #numi =nc * (params['nbclasses'] * (params['prestime']+params['interpresdelay'])) + ii * (params['prestime']+params['interpresdelay']) + nn

                inputT[location][0][0][:][:] = p[:][:]
                labelT[location][0][np.where(unpermcats == catnum)] = 1
                #if nn == 0:
                #    print(labelT[location][0])
                location += 1
            location += params['interpresdelay']

    # Inserting the test character
    p = random.choice(imagedata[testcat])
    for nr in range(rots[testcat]):
        p = np.rot90(p)
    p = skimage.transform.resize(p, (31, 31))
    for nn in range(params['prestimetest']):
        inputT[location][0][0][:][:] = p[:][:]
        location += 1
        
    # Generating the test label
    testlabel = np.zeros(params['nbclasses'])
    testlabel[np.where(unpermcats == testcat)] = 1
    #print(testcat, testlabel)

    #pdb.set_trace()
        
    
    assert(location == params['nbsteps'])

    inputT = torch.from_numpy(inputT).type(ttype)  # Convert from numpy to Tensor
    labelT = torch.from_numpy(labelT).type(ttype)
    targetL = torch.from_numpy(testlabel).type(ttype)

    return inputT, labelT, targetL





def train(paramdict=None):
    print("Setting random seed to", 0)
    np.random.seed(0); random.seed(0); torch.manual_seed(0)
    print("Starting testing...")
    params = {}

    params.update(defaultParams)
    if paramdict:
        params.update(paramdict)
    
    #pdb.set_trace()

    print("Loading Omniglot data...")
    imagedata = []
    imagefilenames=[]
    for basedir in ('./omniglot-master/python/images_background/', 
                    './omniglot-master/python/images_evaluation/'):
        alphabetdirs = glob.glob(basedir+'*')
        print(alphabetdirs[:4])
        for alphabetdir in alphabetdirs:
            chardirs = glob.glob(alphabetdir+"/*")
            for chardir in chardirs:
                chardata = []
                charfiles = glob.glob(chardir+'/*')
                for fn in charfiles:
                    filedata = plt.imread(fn)
                    chardata.append(filedata)
                imagedata.append(chardata)
                imagefilenames.append(fn)
    # imagedata is now a list of lists of numpy arrays 
    # imagedata[CharactertNumber][FileNumber] -> numpy(105,105)
    np.random.shuffle(imagedata)  # Randomize order of characters 
    print(len(imagedata))
    print(imagedata[1][2].shape)
    print("Data loaded!")


    successrates = []
    totaliter = 0 
    totalmistakes = 0

    for myseed in range(10):

        # Opus just didn't save some files for the 4000000 iter, for some reason. If you're looking specifically at iter 4000000, uncomment one of these:
        #if myseed == 3  or myseed == 4:  # for tanh
        #if myseed == 0  or myseed == 5:  # for selu
        #    continue
        
        
        suffix = "_activation_tanh_alphatype_yoked_flare_0_gamma_0.3_imgsize_31_interpresdelay_0_lr_0.0001_nbclasses_5_nbfeatures_64_nbiter_5000000_nbshots_1_prestime_1_prestimetest_1_rule_oja_stepsizelr_1000000.0_rngseed_"+str(myseed)
        #suffix = "_activation_tanh_alphatype_yoked_flare_0_gamma_0.3_imgsize_31_interpresdelay_0_lr_0.0001_nbclasses_5_nbfeatures_64_nbiter_5000000_nbshots_1_plastsize_500_prestime_1_prestimetest_1_rule_oja_stepsizelr_1000000.0_rngseed_"+str(myseed)
        #suffix = "_activation_tanh_alphatype_yoked_flare_0_gamma_0.3_imgsize_31_interpresdelay_0_lr_0.0001_nbclasses_5_nbfeatures_64_nbiter_5000000_nbshots_1_plastsize_500_prestime_1_prestimetest_1_rule_oja_stepsizelr_1000000.0_rngseed_"+str(myseed)#+"_4000000"
        #suffix = "_activation_selu_alphatype_free_flare_0_gamma_0.3_imgsize_31_interpresdelay_0_lr_0.0001_nbclasses_5_nbfeatures_64_nbiter_5000000_nbshots_1_plastsize_500_prestime_1_prestimetest_1_rule_oja_stepsizelr_1000000.0_rngseed_"+str(myseed)
        #suffix = "_activation_selu_alphatype_free_flare_0_gamma_0.3_imgsize_31_interpresdelay_0_lr_0.0001_nbclasses_5_nbfeatures_64_nbiter_5000000_nbshots_1_plastsize_500_prestime_1_prestimetest_1_rule_oja_stepsizelr_1000000.0_rngseed_"+str(myseed)
        with open('./tmp/results'+suffix+'.dat', 'rb') as fo:
            tmpw = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(ttype))
            tmpalpha = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(ttype))
            tmpeta = torch.nn.Parameter(torch.from_numpy(pickle.load(fo)).type(ttype))
            tmplss = pickle.load(fo)
            paramdictLoadedFromFile = pickle.load(fo)
        
        params.update(paramdictLoadedFromFile)

        
        print("Initializing network")
        net = Network(params)
        #net.cuda()
        
        print ("Size of all optimized parameters:", [x.size() for x in net.parameters()])
        allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
        print ("Size (numel) of all optimized elements:", allsizes)
        print ("Total size (numel) of all optimized elements:", sum(allsizes))

        
        print("Passed params: ", params)
        print(platform.uname())
        sys.stdout.flush()
        params['nbsteps'] = params['nbshots'] * ((params['prestime'] + params['interpresdelay']) * params['nbclasses']) + params['prestimetest']  # Total number of steps per episode
        
        net.load_state_dict(torch.load('./tmp/torchmodel'+suffix + '.txt'))
        
        
        
        params['nbiter'] = 100



        # Initialize random seeds ; not sure if really useful here...
        #print("Setting random seed to", params['rngseed'])
        #np.random.seed(params['rngseed']); random.seed(params['rngseed']); torch.manual_seed(params['rngseed'])
        #print(click.get_current_context().params)

        total_loss = 0.0
        #print("Initializing optimizer")
        ##optimizer = torch.optim.Adam([net.w, net.alpha, net.eta], lr=params['learningrate'])
        #optimizer = torch.optim.Adam(net.parameters(), lr=params['learningrate'])
        all_losses = []
        #print_every = 20
        nowtime = time.time()
        print("Starting episodes...")
        sys.stdout.flush()

        nbmistakes = 0

        for numiter in range(params['nbiter']):
            
            hebb = net.initialZeroHebb()
            #optimizer.zero_grad()


            is_test_step = 1

            inputs, labels, target = generateInputsLabelsAndTarget(params, imagedata, test=is_test_step)


            for numstep in range(params['nbsteps']):
                y, hebb = net(Variable(inputs[numstep], requires_grad=False), Variable(labels[numstep], requires_grad=False), hebb)

            #loss = (y[0] - Variable(target, requires_grad=False)).pow(2).sum()
            criterion = torch.nn.BCELoss()
            loss = criterion(y[0], Variable(target, requires_grad=False))

            #if is_test_step == False:
            #    loss.backward()
            #    optimizer.step()

            lossnum = loss.data[0]
            #total_loss  += lossnum
            if is_test_step:
                total_loss = lossnum

            if is_test_step: # (numiter+1) % params['print_every'] == 0:

                print(numiter, "====")
                td = target.cpu().numpy()
                yd = y.data.cpu().numpy()[0]
                #print("y: ", yd[:10])
                #print("target: ", td[:10])
                if np.argmax(td) != np.argmax(yd):
                    print("Mistake!")
                    nbmistakes += 1
                #print("target: ", target.unsqueeze(0)[0][:10])
                absdiff = np.abs(td-yd)
                #print("Mean / median / max abs diff:", np.mean(absdiff), np.median(absdiff), np.max(absdiff))
                #print("Correlation (full / sign): ", np.corrcoef(td, yd)[0][1], np.corrcoef(np.sign(td), np.sign(yd))[0][1])
                #print inputs[numstep]
                previoustime = nowtime
                nowtime = time.time()
                #print("Time spent on last", params['print_every'], "iters: ", nowtime - previoustime)
                #total_loss /= params['print_every']
                #print("Mean loss over last", params['print_every'], "iters:", total_loss)
                #print("Loss on single withheld-data episode:", lossnum)
                all_losses.append(total_loss)
                #print ("Eta: ", net.eta.data.cpu().numpy())
                sys.stdout.flush()
                sys.stderr.flush()

                total_loss = 0

        all_losses = np.array(all_losses)
        print("Mean / std all losses :", np.mean(all_losses), np.std(all_losses))
        print("1st Quartile / median / 3rd Quartile all losses :", np.percentile(all_losses, 25), np.percentile(all_losses, 50), np.percentile(all_losses, 75))
        print("Max of all losses :", np.max(all_losses))
        print("Nb of mistakes :", nbmistakes, "over", numiter+1, "trials - (", 100.0 - 100.0 * nbmistakes / (numiter+1), " % correct )")
        successrates.append(100.0 - 100.0 * nbmistakes / (numiter+1))
        totalmistakes += nbmistakes
        totaliter += params['nbiter']

    print ("Mean / stdev success rate across runs: ", np.mean(successrates), np.std(successrates))
    print ("Success rate across all trials:", 100.0 - 100.0 * totalmistakes / totaliter)
    print (totalmistakes, "mistakes out of ", totaliter, "trials")
    print ("Median success rate across runs: ", np.median(successrates))


@click.command()
@click.option('--nbclasses', default=defaultParams['nbclasses'])
@click.option('--nbshots', default=defaultParams['nbshots'])
@click.option('--prestime', default=defaultParams['prestime'])
@click.option('--prestimetest', default=defaultParams['prestimetest'])
@click.option('--interpresdelay', default=defaultParams['interpresdelay'])
@click.option('--nbiter', default=defaultParams['nbiter'])
@click.option('--learningrate', default=defaultParams['learningrate'])
@click.option('--print_every', default=defaultParams['print_every'])
@click.option('--rngseed', default=defaultParams['rngseed'])
def main(nbclasses, nbshots, prestime, prestimetest, interpresdelay, nbiter, learningrate, print_every, rngseed):
    train(paramdict=dict(click.get_current_context().params))
    #print(dict(click.get_current_context().params))

if __name__ == "__main__":
    #train()
    main()

