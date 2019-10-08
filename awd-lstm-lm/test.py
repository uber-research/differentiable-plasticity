import OpusHdfsCopy
from OpusHdfsCopy import transferFileToHdfsDir, checkHdfs
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import pdb

import data
import model

from utils import batchify, get_batch, repackage_hidden

torch.nn.Module.dump_patches=True

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model Testing of Saved Models')
parser.add_argument('--file', type=str, default='',
                    help='name of the file containing the saved model to be tested')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--alphatype', type=str, default='full',
        help="type of alpha matrix: (full, fanout)")
parser.add_argument('--modultype', type=str, default='none',
        help="type of modulation: (none, modplasth2mod, modplastc2mod)")
parser.add_argument('--modulout', type=str, default='single',
        help="modulatory output (single or fanout)")
parser.add_argument('--cliptype', type=str, default='clip',
                    help="clip type (decay, clip, aditya)")
parser.add_argument('--hebboutput', type=str, default='i2c',
                    help='output used for hebbian computations (i2c, h2co, cell, hidden)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--numgpu', type=int, default=0,
                    help='which GPU to use? (no effect if GPU not used at all)')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
args = parser.parse_args()
args.tied = True

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f, map_location=torch.device(args.numgpu))

import platform
print("Torch version:", torch.__version__, "Numpy version:", np.version.version, "Python version:", platform.python_version())

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)


#train_data = train_data[:5000,:]   # For debugging

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = len(corpus.dictionary)
myparams={}
myparams['cliptype'] = args.cliptype
myparams['modultype'] = args.modultype
myparams['modulout'] = args.modulout
myparams['hebboutput'] = args.hebboutput
myparams['alphatype'] = args.alphatype

suffix = args.model+'_'+myparams['cliptype']+'_'+myparams['modultype']+'_'+myparams['modulout']+'_'+myparams['hebboutput']+'_'+myparams['alphatype']+'_lr'+str(args.lr)+'_'+str(args.nlayers)+'l_'+str(args.nhid)+'h'
RESULTSFILENAME = 'results_'+suffix+'.txt'

MODELFILENAME = args.file

###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
#params = list(model.parameters()) + list(criterion.parameters())
#if args.cuda:
#    model = model.cuda()
#    criterion = criterion.cuda()
#    params = list(model.parameters()) + list(criterion.parameters())
####
#total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
#print('Args:', args)
#print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    with torch.no_grad():
        if args.model == 'QRNN': model.reset()
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(batch_size)
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args, evaluation=True)
            output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
            hidden = repackage_hidden(hidden)
        #return total_loss[0] / len(data_source)
    return total_loss / len(data_source)


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

print("MyParams:", myparams)
print("Args:", args)

# Load the best saved model.
model_load(MODELFILENAME)


NUMGPU = args.numgpu
params = list(model.parameters()) + list(criterion.parameters())
if args.cuda:
    model = model.cuda(device=NUMGPU)
    criterion = criterion.cuda(device=NUMGPU)
    params = list(model.parameters()) + list(criterion.parameters())
###
total_params = sum(x.numel() for x in params)#  if x.numel())
print('Args:', args)
print('Model total parameters:', total_params)

#pdb.set_trace()

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
