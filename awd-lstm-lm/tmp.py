import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


import pdb


class PlasticLSTM(nn.Module):
    def __init__(self, isize, hsize, params):
        super(PlasticLSTM, self).__init__()
        self.softmax= torch.nn.functional.softmax
        #if params['activ'] == 'tanh':
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
        if 'alphatype' in params:
            self.alphatype= params['alphatype']
            ok+=1
        if ok < 5:
            raise ValueError('When using PlasticLSTM, must specify cliptype, modultype, modulout, alphatype and hebboutput in params')

        # Plastic connection parameters:
        self.w =  torch.nn.Parameter(.02 * torch.rand(hsize, hsize) - .01)
        if self.alphatype == 'fanout':
            self.alpha = torch.nn.Parameter(.001 * torch.ones(1)) #torch.rand(1,1,hsize))
        else:
            self.alpha =  torch.nn.Parameter(.00001 * torch.rand(hsize, hsize))
        if self.modultype == 'none':
            self.eta = torch.nn.Parameter(.01 * torch.ones(1))  # Everyone has the same eta (Note: if a parameter is not actually used, there can be problems with ASGD handling in main.py) 
        #self.eta = .01
        
        self.h2f = torch.nn.Linear(hsize, hsize)
        self.h2i = torch.nn.Linear(hsize, hsize)
        self.h2opt = torch.nn.Linear(hsize, hsize)
        #self.h2c = torch.nn.Linear(hsize, hsize)  # This (equivalent to Whg in the PyTorch docs, Uc in Wikipedia) is replaced by the plastic connection
        self.x2f = torch.nn.Linear(isize, hsize)
        self.x2opt = torch.nn.Linear(isize, hsize)
        self.x2i = torch.nn.Linear(isize, hsize)
        self.x2c = torch.nn.Linear(isize, hsize)
        
        if self.modultype != 'none':
            self.h2mod = torch.nn.Linear(hsize, 1)  # Although called 'h2mod', it may take input from h or c depending on modultype value
        if self.modulout == 'fanout':
            self.modfanout = torch.nn.Linear(1, hsize)  
        
        self.isize = isize
        self.hsize = hsize


    def forward(self, inputs, hidden): #, hebb, et, pw):  # hidden is a tuple of h, c and hebb
        
        hebb = hidden[2]
        fgt = F.sigmoid(self.x2f(inputs) + self.h2f(hidden[0]))
        ipt = F.sigmoid(self.x2i(inputs) + self.h2i(hidden[0]))
        opt = F.sigmoid(self.x2opt(inputs) + self.h2opt(hidden[0]))
        #cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, F.tanh(self.x2c(inputs) + self.h2c(hidden[0])))
        
        # To implement plasticity, we replace h2c / Whg / Uc with a plastic connection composed of w, alpha and hebb
        # Note that h2c / Whg / Uc is the matrix of weights that takes in the
        # previous time-step h, and whose output (after adding the current input 
        # and passing through tanh) is multiplied by the input gates before being 
        # added to the cell state
        if self.cliptype == 'aditya':
            # Each *column* in w, hebb and alpha constitutes the inputs to a single cell
            # For w and alpha, columns are 2nd dimension (i.e. dim 1); for hebb, it's dimension 2 (dimension 0 is batch)
            h2coutput = hidden[0].unsqueeze(1).bmm(self.w + torch.mul(self.alpha, torch.clamp(hebb, min=-1.0, max=1.0))).squeeze()  
        else:
            h2coutput = hidden[0].unsqueeze(1).bmm(self.w + torch.mul(self.alpha, hebb)).squeeze()  
            #if np.random.rand() < .1:
            #    pdb.set_trace()
        inputstocell =  F.tanh(self.x2c(inputs) + h2coutput)
        #inputstocell =  F.tanh(self.x2c(inputs) + torch.matmul(hidden[0].unsqueeze(1), self.w.unsqueeze(0)).squeeze(1)) 
        cell = torch.mul(fgt, hidden[1]) + torch.mul(ipt, inputstocell) #  self.h2c(hidden[0])))

        
        #pdb.set_trace()
        
        hactiv = torch.mul(opt, F.tanh(cell))
        #pdb.set_trace()
        
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

        if self.modultype == 'none':
            myeta = self.eta
        elif self.modultype == 'modplasth2mod':
            myeta = F.tanh(self.h2mod(hactiv)).unsqueeze(2)  # Shape: BatchSize x 1 x 1
        elif self.modultype == 'modplastc2mod':
            myeta = F.tanh(self.h2mod(cell)).unsqueeze(2)
        else: 
            raise ValueError("Must choose modulation type")
        
        #pdb.set_trace()
        if self.modultype != 'none' and self.modulout == 'fanout':
            # Each *column* in w, hebb and alpha constitutes the inputs to a single cell
            # For w and alpha, columns are 2nd dimension (i.e. dim 1); for hebb, it's dimension 2 (dimension 0 is batch)
            # The output of the following line has shape BatchSize x 1 x NHidden, i.e. 1 line and NHidden columns for each 
            # batch element. When multiplying by hebb (BatchSize x NHidden x NHidden), broadcasting will provide a different
            # value for each cell but the same value for all inputs of a cell, as required by fanout concept.
             myeta = self.modfanout(myeta).squeeze().unsqueeze(1)              

        if self.cliptype == 'decay':
            hebb = (1 - myeta) * hebb + myeta * deltahebb
        elif self.cliptype == 'clip':
            hebb = torch.clamp(hebb + myeta * deltahebb, min=-1.0, max=1.0)
        elif self.cliptype == 'aditya':
            hebb = hebb + myeta * deltahebb   
        else: 
            raise ValueError("Must choose clip type")

        hidden = (hactiv, cell, hebb)
        activout = hactiv #self.h2o(hactiv)
        #if np.isnan(np.sum(hactiv.data.cpu().numpy())) or np.isnan(np.sum(hidden[1].data.cpu().numpy())) :
        #    raise ValueError("Nan detected !")

        return activout, hidden #, hebb, et, pw



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
        #if np.isnan(np.sum(hactiv.data.cpu().numpy())) or np.isnan(np.sum(hidden[1].data.cpu().numpy())) :
        #    raise ValueError("Nan detected !")

        #pdb.set_trace()

        return activout, hidden #, hebb, et, pw



