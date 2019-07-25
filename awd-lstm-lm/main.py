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

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='PLASTICLSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU, PLASTICLSTM, MYLSTM, FASTPLASTICLSTM, SIMPLEPLASTICLSTM)')
parser.add_argument('--alphatype', type=str, default='full',
        help="type of alpha matrix: (full, perneuron, single)")
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
parser.add_argument('--clipval', type=float, default=2.0,
                    help='value of the hebbian trace clipping')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--agdiv', type=float, default=1150.0,
                    help='divider of the gradient of alpha')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=300,
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
parser.add_argument('--proplstm', type=float, default=0.5,
                    help='for split-lstms: proportion of LSTM cells in the recurrent layer')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--asgdtime', type=int, default=-1,
                    help='number of iterations before switch to ASGD (if positive)')
parser.add_argument('--nonmono', type=int, default=5,
                    help='range of non monotonicity before switch to ASGD (if asgdtime is negative)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--numgpu', type=int, default=0,
                    help='which GPU to use? (no effect if GPU not used at all)')
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
    if  not args.cuda :
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
else:
    print("NOTE: no CUDA device detected.")

import platform
print("PyTorch version:", torch.__version__, "Numpy version:", np.version.version, "Python version:", platform.python_version(), "GPU used (if any):", args.numgpu)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

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

# Configuration parameters of the plastic LSTM. See mylstm.py for details.
myparams={}
myparams['clipval'] = args.clipval
myparams['cliptype'] = args.cliptype
myparams['modultype'] = args.modultype
myparams['modulout'] = args.modulout
myparams['hebboutput'] = args.hebboutput
myparams['alphatype'] = args.alphatype

suffix = '_SqUsq_'+args.model+'_'+myparams['cliptype']+'_cv'+str(myparams['clipval'])+'_'+myparams['modultype']+'_'+myparams['modulout']+'_'+myparams['hebboutput']+'_'+myparams['alphatype']+'_asgdtime'+str(args.asgdtime)+'_agdiv'+str(int(args.agdiv))+'_lr'+str(args.lr)+'_'+str(args.nlayers)+'l_'+str(args.nhid)+'h_'+str(args.proplstm)+'lstm_rngseed'+str(args.seed)
print("Suffix:", suffix)
MODELFILENAME = 'model_'+suffix+'.dat'
RESULTSFILENAME = 'results_'+suffix+'.txt'
FILENAMESTOSAVE = [MODELFILENAME, RESULTSFILENAME]  # We will append to this list the additional files at each learning rate reduction, if any

print("Plasticity and neuromodulation parameters:", myparams)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.proplstm, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied, myparams)
###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
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
params = list(model.parameters()) + list(criterion.parameters())
if args.cuda:
    model = model.cuda(args.numgpu)
    criterion = criterion.cuda(args.numgpu)
    params = list(model.parameters()) + list(criterion.parameters())
###
#total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size()) # Smerity version, doesn't work when size==3
total_params = sum(x.numel() for x in params if x.numel())
print('Args:', args)
print('Model total parameters:', total_params)


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
        #return total_loss[0] / len(data_source) # Error under modern PyTorch
    return total_loss / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # NOTE: this was commented out in smerity's code!
        seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # NOTE: Now 'hidden' includes the Hebbian traces if using plasticity.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # When using plastic LSTMs, 
        # We divide the gradient on the alphas by the number of inputs, i.e.
        # the number of recurrent neurons, but only if plasticity is
        # 'perneuron' or 'single' (as opposed to 'full'). 
        # This is necessary to preserve stability while using the same learning rate as Merity et al.
        if args.model == 'PLASTICLSTM' or args.model == 'SPLITLSTM' or args.model == 'FASTPLASTICLSTM':
            if args.alphatype == 'perneuron' or args.alphatype == 'single':  # Based on other experiments, this is actually not good for full-plasticity
                for x in model.rnns:
                    if hasattr(x.alpha.grad, 'data'):
                        x.alpha.grad.data /= args.agdiv
        
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        
        # OPTIMIZATION STEP
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000


# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    allvallosses = []
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]:  # Are we in the ASGD regime?
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                # NOTE (TM): the following line may cause trouble after the switch to ASGD if some declared pytorch Parameters of the network are not actually used in the computational graph
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} (t0 on) | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valloss2 ppl {:8.2f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), math.exp(val_loss2)))
            print('-' * 89)

            if val_loss2 < stored_loss:
                model_save(MODELFILENAME)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

            allvallosses.append(val_loss2)

        else:
            val_loss = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)

            if val_loss < stored_loss:
                model_save(MODELFILENAME)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0]:
                if (args.asgdtime < 0 and len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])) or (args.asgdtime > 0 and len(best_val_loss) == args.asgdtime) :

                    print('Switching to ASGD')
                    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                EPOCHFILENAME = '{}.e{}'.format(MODELFILENAME, epoch)
                model_save(EPOCHFILENAME)
                FILENAMESTOSAVE.append(EPOCHFILENAME)
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)
            
            allvallosses.append(val_loss)

        np.savetxt(RESULTSFILENAME, allvallosses)

        # Saving files remotely.... (Uber only!)
        if os.path.isdir('/mnt/share/tmiconi'):
            print("Transferring to NFS storage...")
            for fn in FILENAMESTOSAVE:
                result = os.system(
                    'cp {} {}'.format(fn, '/mnt/share/tmiconi/ptb/'+fn))
            print("Done!")
        #if checkHdfs():
        #    print("Transfering to HDFS...")
        #    for fn in FILENAMESTOSAVE:
        #        transferFileToHdfsDir(fn, '/ailabs/tmiconi/ptb/')


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(MODELFILENAME)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
