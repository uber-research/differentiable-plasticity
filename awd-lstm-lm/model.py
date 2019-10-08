import torch
import torch.nn as nn
#from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

import random, pdb

import mylstm

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, proplstm, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, params={}):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU', 'MYLSTM', 'MYFASTLSTM', 'SIMPLEPLASTICLSTM', 'FASTPLASTICLSTM', 'PLASTICLSTM', 'SPLITLSTM'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]

            #for rr in self.rnns:
            #    rr.flatten_parameters()
            if wdrop:
                print("Using WeightDrop!")
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]

        elif rnn_type == 'MYLSTM': 
            self.rnns = [mylstm.MyLSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid)) for l in range(nlayers)]

        elif rnn_type == 'MYFASTLSTM': 
            self.rnns = [mylstm.MyFastLSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid)) for l in range(nlayers)]

        elif rnn_type == 'PLASTICLSTM':
            self.rnns = [mylstm.PlasticLSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), params) for l in range(nlayers)]

        elif rnn_type == 'SIMPLEPLASTICLSTM':
            # Note that this one ignores the 'params' argument, which is only kept to preserve identical signature with PlasticLSTM
            self.rnns = [mylstm.SimplePlasticLSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), params) for l in range(nlayers)]

        elif rnn_type == 'FASTPLASTICLSTM':
            self.rnns = [mylstm.MyFastPlasticLSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), params) for l in range(nlayers)]

        elif rnn_type == 'SPLITLSTM': # Not used
            self.rnns = [mylstm.SplitLSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), proplstm, params) for l in range(nlayers)]

        elif rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.proplstm = proplstm
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights



    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            # Each rnn is a layer!
            # each raw_output has shape seq_len x batch_size x nb_hidden
            # new_h is a tuple of 2 elements, each of size 1 x batch_size x nb_hidden (last h and last c)
            if self.rnn_type != 'MYLSTM' and self.rnn_type != 'MYFASTLSTM' and self.rnn_type != 'SIMPLEPLASTICLSTM' and self.rnn_type != 'PLASTICLSTM' and self.rnn_type != 'FASTPLASTICLSTM' and self.rnn_type != 'SPLITLSTM':
                raw_output, new_h = rnn(raw_output, hidden[l])
            else:
                single_h = hidden[l]  # actually a tuple, includes the h and the c (and for plastic LTMS, includes Hebb as third element!)
                singleouts = []
                for z in range(raw_output.shape[0]):
                    singleout, single_h = rnn(raw_output[z], single_h)
                    #if z==0:
                    #    print("RANDOM NUMBER 1:",float(torch.rand(1)))
                    singleouts.append(singleout)
                new_h = single_h  # the last (h,c[,hebb]) after the sequence is processed
                raw_output = torch.stack(singleouts)
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                # lockdrop will zero out some output units over the whole sequence (separately chosen for each batch, but fixed across sequence)
                #pdb.set_trace()
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
                #pdb.set_trace()
        hidden = new_hidden
        #pdb.set_trace()

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'MYLSTM' or self.rnn_type == 'MYFASTLSTM':
            return [((weight.new(bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()),
                    (weight.new(bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()))
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'PLASTICLSTM' or self.rnn_type == 'SIMPLEPLASTICLSTM':
            return [(
                    (weight.new(bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()), # h state
                    (weight.new(bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()), # c state
                    (weight.new(bsz, self.rnns[l].w.shape[0], self.rnns[l].w.shape[1]).zero_()) # hebbian trace for the recurrent weights
                    #(weight.new(bsz, self.rnns[l].isize, self.rnns[l].hsize).zero_())  # hebbian trace for the input weights (not necessarily used)
                    )
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'FASTPLASTICLSTM':
            return [(
                    (weight.new(bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()), # h state
                    (weight.new(bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()), # c state
                    (weight.new(bsz, self.rnns[l].hsize, self.rnns[l].hsize).zero_()) # hebbian trace of recurrent weights
                    #(weight.new(bsz, self.rnns[l].isize, self.rnns[l].hsize).zero_())  # hebbian trace for the input weights (not necessarily used)
                    #(weight.new(bsz, self.rnns[l].w.shape[0], self.rnns[l].w.shape[1]).zero_()), # hebbian trace for the recurrent weights
                    #(weight.new(bsz, self.rnns[l].win.shape[0], self.rnns[l].win.shape[1]).zero_())  # hebbian trace for the input weights (not necessarily used)
                    )
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'SPLITLSTM':
            return [(
                    (weight.new(bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()),   # H state
                    (weight.new(bsz, self.rnns[l].lsize ).zero_()),   # C state
                    (weight.new(bsz, self.rnns[l].w.shape[0], self.rnns[l].w.shape[1]).zero_()),   # hebb
                    (weight.new(bsz, self.rnns[l].win.shape[0], self.rnns[l].win.shape[1]).zero_())  # hebbin
                    )
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'LSTM' :
            return [((weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()),
                    (weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()))
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
