# Plastic LSTMs, with neuromodulation (backpropamine), 
# as described in Miconi et al. ICLR 2019,
# by Thomas Miconi and Aditya Rawal.
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 



import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


import pdb



# SimplePlasticLSTM is a full-fledged implementation of Plastic LSTMs that uses
# default settings and is not parametrizable beyond input size and hidden size.
# This allows for simpler code and easier understanding. See "PlasticLSTM"
# below for a more customizable version.

class SimplePlasticLSTM(nn.Module):             
    def __init__(self, isize, hsize, params):   # Note that 'params' is ignored for this class; we keep it to preserve the constructor's signature
        super(SimplePlasticLSTM, self).__init__()
        self.softmax= torch.nn.functional.softmax
        self.activ = F.tanh

        # Plastic connection trainable parameters, i.e. w and alpha:
        self.w =  torch.nn.Parameter(.02 * torch.rand(hsize, hsize) - .01)
        self.alpha = torch.nn.Parameter(.0001 * torch.rand(1,1,hsize))      # One alpha per neuron (all incoming connections to a neuron share same alpha)
        #self.alpha = torch.nn.Parameter(.0001 * torch.ones(1))             # One alpha for the whole network
        #self.alpha =  torch.nn.Parameter(.0001 * torch.rand(hsize, hsize)) # One alpha per connection
        
        self.h2f = torch.nn.Linear(hsize, hsize)
        self.h2i = torch.nn.Linear(hsize, hsize)
        self.h2opt = torch.nn.Linear(hsize, hsize)
        #self.h2c = torch.nn.Linear(hsize, hsize)  # This (equivalent to Whg in PyTorch LSTM docs / Uc in Wikipedia description of LSTM) is replaced by the plastic connection
        self.x2f = torch.nn.Linear(isize, hsize)
        self.x2opt = torch.nn.Linear(isize, hsize)
        self.x2i = torch.nn.Linear(isize, hsize)
        self.x2c = torch.nn.Linear(isize, hsize)  
       
        # Modulator output (M(t))
        self.h2mod = torch.nn.Linear(hsize, 1)      # Takes input from the h-state, computes the neuromodulator output
        self.modfanout = torch.nn.Linear(1, hsize)  # Projects the network's common neuromodulator output onto each neuron
        
        self.isize = isize
        self.hsize = hsize


    def forward(self, inputs, hidden): #, hebb, et, pw):  # hidden is a tuple of h, c and hebb
        hebb = hidden[2]
        fgt = F.sigmoid(self.x2f(inputs) + self.h2f(hidden[0]))
        ipt = F.sigmoid(self.x2i(inputs) + self.h2i(hidden[0]))
        opt = F.sigmoid(self.x2opt(inputs) + self.h2opt(hidden[0]))
        
        # To implement plasticity, we replace h2c / Whg / Uc with a plastic connection composed of w, alpha and hebb
        # Note that h2c / Whg / Uc is the matrix of weights that takes in the
        # previous time-step h, and whose output (after adding the current input 
        # and passing through tanh) is multiplied by the input gates before being 
        # added to the cell state
        # Note: Each *column* in w, hebb and alpha constitutes the inputs to a single cell
        # For w and alpha, columns are 2nd dimension (i.e. dim 1); for hebb, it's dimension 2 (dimension 0 is batch)
        
        # This is probably not the most elegant way to do it, but it works (remember that there is one alpha per neuron, applied to all input connections of this neuron)
        h2coutput = hidden[0].unsqueeze(1).bmm(self.w + torch.mul(self.alpha, hebb)).squeeze(1)  

        x2coutput = self.x2c(inputs)
        inputstocell =  F.tanh(self.x2c(inputs) + h2coutput)  #  We compute this intermediary state to be used in Hebbian computations below
        
        # Finally, compute the new cell and hidden states
        cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, inputstocell) 
        hactiv = torch.mul(opt, F.tanh(cell))
        
        # Now we need to update the Hebbian traces, including any neuromodulation.

        deltahebb = torch.bmm(hidden[0].unsqueeze(2), inputstocell.unsqueeze(1))
        myeta = F.tanh(self.h2mod(hactiv)).unsqueeze(2)  # Shape: BatchSize x 1 x 1
        
        # The output of the following line has shape BatchSize x 1 x NHidden, i.e. 1 line and NHidden columns for each 
        # batch element. 
        # When multiplying by deltahebb (BatchSize x NHidden x NHidden), broadcasting will provide a different
        # value for each column but the same value for all rows within each column. This is equivalent to providing
        # the same neuromodulation to all the inputs to a given cell, while letting neuromodulation differ from 
        # cell to cell, as required for the fanout concept.
        
        myeta = self.modfanout(myeta).squeeze().unsqueeze(1)              

        hebb = torch.clamp(hebb + myeta * deltahebb, min=-2.0, max=2.0)

        # Note that "hactiv" (i.e. the new h-state) is duplicated in the return
        # values. This is to maintain the signature used by main.py/model.py (which is from Merity et al.'s code)
        # and is not necessary for other applications.

        hidden = (hactiv, cell, hebb) 
        activout = hactiv 

        return activout, hidden 



# A more customizable version of plastic LSTMs, using parameters passed in the 'params' argument.

class PlasticLSTM(nn.Module):
    def __init__(self, isize, hsize, params):
        super(PlasticLSTM, self).__init__()
        self.softmax= torch.nn.functional.softmax
        #if params['activ'] == 'tanh':
        self.activ = F.tanh
    
        # Default values for configuration parameters:
        self.cliptype, self.modultype, self.hebboutput, self.modulout, self.clipval, self.alphatype = 'clip', 'modplasth2mod', 'i2c', 'fanout', '2.0', 'perneuron'

        # Description of the parameters:

        # alphatype: do we have one alpha coefficient for each connection
        # ('full'), one per neuron ('perneuron' - i.e. all input connections to
        # a given neuron share the same alpha), or one for the entire network
        # ('single')?

        # modultype: 'none' (non-modulated plasticity) , 'modplasth2mod'
        # (neuromodulation takes input from the current h-state) or
        # 'modplastc2mod' (neuromodulation takes input from the currrent
        # c-state).

        # cliptype: 'clip', 'aditya' or 'decay' - specifies how the Hebbian traces should be constrained.

        # clipval: maximum magnitude of the Hebbian trace values  (default 2.0)

        # modulout: 'single' (all connections receive the same neuromodulator
        # output) or 'fanout' (neuromodulator input goes through a 1xN linear layer to reach each neuron)

        # hebboutput: what counts as the "output" in the Hebbian product of input by output. Better to leave it at 'i2c'.

        if 'cliptype' in params:
            self.cliptype = params['cliptype']
        if 'modultype' in params:
            self.modultype = params['modultype']
        if 'hebboutput' in params:
            self.hebboutput = params['hebboutput']
        if 'modulout' in params:
            self.modulout= params['modulout']
        if 'clipval' in params:
            self.clipval= params['clipval']
        if 'alphatype' in params:
            self.alphatype= params['alphatype']

        # Plastic connection trainable parameters, i.e. w and alpha:
        self.w =  torch.nn.Parameter(.02 * torch.rand(hsize, hsize) - .01)
        if self.alphatype == 'perneuron':
            self.alpha = torch.nn.Parameter(.0001 * torch.rand(1,1,hsize))
        elif self.alphatype == 'single':
            self.alpha = torch.nn.Parameter(.0001 * torch.ones(1))
        elif self.alphatype == 'full':
            self.alpha =  torch.nn.Parameter(.0001 * torch.rand(hsize, hsize))
        else:
            raise ValueError("Must select appropriate alpha type (current incorrect value is:", str(self.alphatype), ")")
        if self.modultype == 'none':
            self.eta = torch.nn.Parameter(.01 * torch.ones(1))  # Everyone has the same eta (Note: if a parameter is not actually used, there can be problems with ASGD handling in main.py) 
        
        self.h2f = torch.nn.Linear(hsize, hsize)
        self.h2i = torch.nn.Linear(hsize, hsize)
        self.h2opt = torch.nn.Linear(hsize, hsize)
        #self.h2c = torch.nn.Linear(hsize, hsize)  # This (equivalent to Whg in PyTorch LSTM docs / Uc in Wikipedia description of LSTM) is replaced by the plastic connection
        self.x2f = torch.nn.Linear(isize, hsize)
        self.x2opt = torch.nn.Linear(isize, hsize)
        self.x2i = torch.nn.Linear(isize, hsize)
        self.x2c = torch.nn.Linear(isize, hsize)  
       
        if self.modultype != 'none':
            # This is the layer that computes the neuromodulator output at any time step, based on current hidden state.
            # Although called 'h2mod', it may take input from h or c depending on modultype value
            self.h2mod = torch.nn.Linear(hsize, 1)  
            # Is the modulation just a single scalar, or do we pass it through a 'fanout' weight matrix to get one different value for each target neuron?
            if self.modulout == 'fanout':
                self.modfanout = torch.nn.Linear(1, hsize)  
        
        self.isize = isize
        self.hsize = hsize


    def forward(self, inputs, hidden): #, hebb, et, pw):  # hidden is a tuple of h, c and hebb
        hebb = hidden[2]
        fgt = F.sigmoid(self.x2f(inputs) + self.h2f(hidden[0]))
        ipt = F.sigmoid(self.x2i(inputs) + self.h2i(hidden[0]))
        opt = F.sigmoid(self.x2opt(inputs) + self.h2opt(hidden[0]))
        
        # To implement plasticity, we replace h2c / Whg / Uc with a plastic connection composed of w, alpha and hebb
        # Note that h2c / Whg / Uc is the matrix of weights that takes in the
        # previous time-step h, and whose output (after adding the current input 
        # and passing through tanh) is multiplied by the input gates before being 
        # added to the cell state
        # Note: Each *column* in w, hebb and alpha constitutes the inputs to a single cell
        # For w and alpha, columns are 2nd dimension (i.e. dim 1); for hebb, it's dimension 2 (dimension 0 is batch)
        if self.cliptype == 'aditya':   # Clipping Hebbian traces a posteriori
            h2coutput = hidden[0].unsqueeze(1).bmm(self.w + torch.mul(self.alpha, torch.clamp(hebb, min=-self.clipval, max=self.clipval))).squeeze(1)  
        else:
            h2coutput = hidden[0].unsqueeze(1).bmm(self.w + torch.mul(self.alpha, hebb)).squeeze(1)  

        x2coutput = self.x2c(inputs)
        inputstocell =  F.tanh(self.x2c(inputs) + h2coutput)  #  We compute this intermediary state to be used in Hebbian computations below
        
        # Finally, compute the new cell and hidden states
        cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, inputstocell) 
        hactiv = torch.mul(opt, F.tanh(cell))
        
        # Now we need to compute the updates to the Hebbian traces, including any neuromodulation.

        # For the Hebbian computation, what counts as "output"?
        if self.hebboutput == 'i2c':
            deltahebb = torch.bmm(hidden[0].unsqueeze(2), inputstocell.unsqueeze(1))
        elif self.hebboutput == 'h2co': 
            deltahebb = torch.bmm(hidden[0].unsqueeze(2), h2coutput.unsqueeze(1))
        elif self.hebboutput == 'cell': 
            deltahebb = torch.bmm(hidden[0].unsqueeze(2), cell.unsqueeze(1))
        elif self.hebboutput == 'hidden': 
            deltahebb = torch.bmm(hidden[0].unsqueeze(2), hactiv.unsqueeze(1)) 
        else: 
            raise ValueError("Must choose Hebbian target output")

        # What is the source of the neuromodulator computation (if any)?
        if self.modultype == 'none':
            myeta = self.eta
        elif self.modultype == 'modplasth2mod': # The neuromodulation takes input from the h-state
            myeta = F.tanh(self.h2mod(hactiv)).unsqueeze(2)  # Shape: BatchSize x 1 x 1
        elif self.modultype == 'modplastc2mod': # The neuromodulation takes input from the c-state
            myeta = F.tanh(self.h2mod(cell)).unsqueeze(2)
        else: 
            raise ValueError("Must choose modulation type")
        

        # If we use "fanout" neuromodulation, the neuromodulator output is passed through a (trainable) linear layer before hitting the neurons. 
        if self.modultype != 'none' and self.modulout == 'fanout':
            # The output of the following line has shape BatchSize x 1 x NHidden, i.e. 1 line and NHidden columns for each 
            # batch element. 
            # When multiplying by deltahebb (BatchSize x NHidden x NHidden), broadcasting will provide a different
            # value for each column but the same value for all rows within each column. This is equivalent to providing
            # the same neuromodulation to all the inputs to a given cell, while letting neuromodulation differ from 
            # cell to cell, as required for the fanout concept.
            
            myeta = self.modfanout(myeta).squeeze().unsqueeze(1)              

        # Various possible ways to clip the Hebbian trace 
        if self.cliptype == 'decay':    # Exponential decay
            hebb = (1 - myeta) * hebb + myeta * deltahebb
        elif self.cliptype == 'clip':   # Just a hard clip
            hebb = torch.clamp(hebb + myeta * deltahebb, min=-self.clipval, max=self.clipval)
        elif self.cliptype == 'aditya': # For this one, the clipping only occurs a posteriori (see above); hebb itself can grow arbitrarily
            hebb = hebb + myeta * deltahebb   
        else: 
            raise ValueError("Must choose clip type")


        # Note that "hactiv" (i.e. the new h-state) is duplicated in the return
        # values. This is to maintain the signature used by main.py/model.py
        # and is not necessary for other applications.

        hidden = (hactiv, cell, hebb) 
        activout = hactiv 

        return activout, hidden 



# This is a slightly faster implementation of Plastic Lstms: cut time by ~30%  by grouping all matrix multiplications into two. Not fully debugged, use at own risk.
class MyFastPlasticLSTM(nn.Module):
    def __init__(self, isize, hsize, params):
        super(MyFastPlasticLSTM, self).__init__()
        self.softmax= torch.nn.functional.softmax
        self.activ = F.tanh

        ok=0
        if 'cliptype' in params:
            self.cliptype = params['cliptype']
            ok+=1
        if 'modultype' in params:
            self.modultype = params['modultype']
            ok+=1
        if 'hebboutput' in params:
            self.hebboutput = params['hebboutput']
            ok+=1
        if 'modulout' in params:
            self.modulout= params['modulout']
            ok+=1
        if 'clipval' in params:
            self.clipval= params['clipval']
            ok+=1
        if 'alphatype' in params:
            self.alphatype= params['alphatype']
            ok+=1
        if ok < 6:
            raise ValueError('When constructing PlasticLSTM, must pass "params" dictionary including cliptype, clipval, modultype, modulout, alphatype and hebboutput')

        # We group all weight matrices into two, just like the C implementation of LSTMs in PyTorch does. Faster!
        # Note: this creates some redundant biases (though not many)
        self.h2f_i_opt_c = torch.nn.Linear(hsize, 4*hsize) # Weights from h to f, i, o and c
        self.x2f_i_opt_c = torch.nn.Linear(isize, 4*hsize) # Weights from x to f, i, o and c
        self.isize = isize
        self.hsize = hsize
        
        if self.modultype != 'none':
            self.h2mod = torch.nn.Linear(hsize, 1)  # Although called 'h2mod', it may take input from h or c depending on modultype value
            if self.modulout == 'fanout':
                self.modfanout = torch.nn.Linear(1, hsize)  
        
        if self.alphatype == 'perneuron':
            self.alpha = torch.nn.Parameter(.0001 * torch.rand(1,1,hsize))
            #self.alpha = Variable(.0001 * torch.ones(1).cuda(), requires_grad=True) #torch.rand(1,1,hsize))
        elif self.alphatype == 'single':
            self.alpha = torch.nn.Parameter(.0001 * torch.ones(1))
        elif self.alphatype == 'full':
            self.alpha =  torch.nn.Parameter(.0001 * torch.rand(hsize, hsize))
        else:
            raise ValueError("Must select alpha type (current incorrect value is:", str(self.alphatype), ")")
        if self.modultype == 'none':
            self.eta = torch.nn.Parameter(.01 * torch.ones(1))  # Everyone has the same eta (Note: if a parameter is not actually used, there can be problems with ASGD handling in main.py) 



    def forward(self, inputs, hidden): #, hebb, et, pw):  # hidden is a tuple of h and c states
            
        hsize = self.hsize
        #fgt = F.sigmoid(self.x2f(inputs) + self.h2f(hidden[0]))
        #ipt = F.sigmoid(self.x2i(inputs) + self.h2i(hidden[0])) 
        #opt = F.sigmoid(self.x2opt(inputs) + self.h2opt(hidden[0])) 
        alloutputs = self.x2f_i_opt_c(inputs) + self.h2f_i_opt_c(hidden[0])
        
        # hidden[0] and hidden[1] are the h state and the c state; hidden[2] is the hebbian trace
        hebb = hidden[2]

        fgt = F.sigmoid(alloutputs[:,:hsize])
        ipt = F.sigmoid(alloutputs[:,hsize:2*hsize])
        opt = F.sigmoid(alloutputs[:,2*hsize:3*hsize])
        handx2coutput_w = alloutputs[:,3*hsize:]
        if self.cliptype == 'aditya':
            h2coutput_hebb = hidden[0].unsqueeze(1).bmm(torch.mul(self.alpha, self.clipval * torch.tanh(hebb))).squeeze(1)  # Slightly different version
        else:
            h2coutput_hebb = hidden[0].unsqueeze(1).bmm(torch.mul(self.alpha, hebb)).squeeze(1)  
        inputtoc = F.tanh(handx2coutput_w + h2coutput_hebb)
        
            # Each *column* in w, hebb and alpha constitutes the inputs to a single cell
            # For w and alpha, columns are 2nd dimension (i.e. dim 1); for hebb, it's dimension 2 (dimension 0 is batch)
        
        cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, inputtoc)
        hactiv = torch.mul(opt, F.tanh(cell))


        #if self.hebboutput == 'i2c':
        deltahebb = torch.bmm(hidden[0].unsqueeze(2), inputtoc.unsqueeze(1))
        if self.modultype == 'none':
            myeta = self.eta
        elif self.modultype == 'modplasth2mod':
            myeta = F.tanh(self.h2mod(hactiv)).unsqueeze(2)  # Shape: BatchSize x 1 x 1
        elif self.modultype == 'modplastc2mod':
            myeta = F.tanh(self.h2mod(cell)).unsqueeze(2)
        else: 
            raise ValueError("Must choose modulation type")
        
        if self.modultype != 'none' and self.modulout == 'fanout':
            # Each *column* in w, hebb and alpha constitutes the inputs to a single cell
            # For w and alpha, columns are 2nd dimension (i.e. dim 1); for hebb, it's dimension 2 (dimension 0 is batch)
            # The output of the following line has shape BatchSize x 1 x NHidden, i.e. 1 line and NHidden columns for each 
            # batch element. When multiplying by hebb (BatchSize x NHidden x NHidden), broadcasting will provide a different
            # value of myeta for each cell but the same value for all inputs of a cell, as required by fanout concept.
             myeta = self.modfanout(myeta).squeeze().unsqueeze(1)              

        if self.cliptype == 'decay':
            hebb = (1 - myeta) * hebb + myeta * deltahebb
        elif self.cliptype == 'clip':
            hebb = torch.clamp(hebb + myeta * deltahebb, min=-self.clipval, max=self.clipval)
        elif self.cliptype == 'aditya' :
            hebb = hebb + myeta * deltahebb   
        else: 
            raise ValueError("Must choose clip type")

        hidden = (hactiv, cell, hebb)
        activout = hactiv 
        


        return activout, hidden #, hebb, et, pw







# Standard, non-plastic LSTM, reimplemented "by hand" to check if our
# implementation is correct, and to ensure that our comparisons use the closest
# possible non-plastic equivalent to our plastic LSTMs. Gets almost identical
# results to the PyTorch internal LSTM used by the original smerity code.

class MyLSTM(nn.Module):
    def __init__(self, isize, hsize):
        super(MyLSTM, self).__init__()
        self.softmax= torch.nn.functional.softmax
        #if params['activ'] == 'tanh':
        self.activ = F.tanh
        self.h2f = torch.nn.Linear(hsize, hsize)
        self.h2i = torch.nn.Linear(hsize, hsize)
        self.h2opt = torch.nn.Linear(hsize, hsize)
        self.h2c = torch.nn.Linear(hsize, hsize)
        self.x2f = torch.nn.Linear(isize, hsize)
        self.x2opt = torch.nn.Linear(isize, hsize)
        self.x2i = torch.nn.Linear(isize, hsize)
        self.x2c = torch.nn.Linear(isize, hsize)
        self.isize = isize
        self.hsize = hsize



    def forward(self, inputs, hidden): #, hebb, et, pw):  # hidden is a tuple of h and c states
            
        fgt = F.sigmoid(self.x2f(inputs) + self.h2f(hidden[0]))
        ipt = F.sigmoid(self.x2i(inputs) + self.h2i(hidden[0]))
        opt = F.sigmoid(self.x2opt(inputs) + self.h2opt(hidden[0]))
        cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, F.tanh(self.x2c(inputs) + self.h2c(hidden[0])))
        hactiv = torch.mul(opt, F.tanh(cell))
        #pdb.set_trace()
        hidden = (hactiv, cell)
        activout = hactiv #self.h2o(hactiv)
        #pdb.set_trace()
        return activout, hidden #, hebb, et, pw




# Faster MyLSTM - by ~30% in comparison to MyLSTM, by grouping matrices and matrix multiplications. Not fully debugged, use at own risk.
class MyFastLSTM(nn.Module):
    def __init__(self, isize, hsize):
        super(MyFastLSTM, self).__init__()
        self.softmax= torch.nn.functional.softmax
        #if params['activ'] == 'tanh':
        self.activ = F.tanh
        # We group all weight matrices into two, just like the C implementation of LSTMs in PyTorch does
        # Note: this creates some redundant biases (though not many)
        self.h2f_i_opt_c = torch.nn.Linear(hsize, 4*hsize) # Weights from h to f, i, o and c
        self.x2f_i_opt_c = torch.nn.Linear(isize, 4*hsize) # Weights from x to f, i, o and c
        self.isize = isize
        self.hsize = hsize



    def forward(self, inputs, hidden): #, hebb, et, pw):  # hidden is a tuple of h and c states
            
        #fgt = F.sigmoid(self.x2f(inputs) + self.h2f(hidden[0])) #
        #ipt = F.sigmoid(self.x2i(inputs) + self.h2i(hidden[0])) #
        #opt = F.sigmoid(self.x2opt(inputs) + self.h2opt(hidden[0])) #
        alloutputs = self.x2f_i_opt_c(inputs) + self.h2f_i_opt_c(hidden[0])
        
        hsize = self.hsize
        # You can gain ~ 5% in speed by grouping these three :
        fgt = F.sigmoid(alloutputs[:,:hsize])
        ipt = F.sigmoid(alloutputs[:,hsize:2*hsize])
        opt = F.sigmoid(alloutputs[:,2*hsize:3*hsize])
        inputtoc = F.tanh(alloutputs[:,3*hsize:])
        #cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, F.tanh(self.x2c(inputs) + self.h2c(hidden[0])))#
        cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, inputtoc)
        hactiv = torch.mul(opt, F.tanh(cell))
        hidden = (hactiv, cell)
        activout = hactiv 
        #pdb.set_trace()
        return activout, hidden #, hebb, et, pw


