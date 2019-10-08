# LSTMs with neuromodulated plasticity


This code implements language modelling on the Penn Treebank dataset, using LSTMs with neuromodulated plasticity ("backpropamine"), as described in [Backpropamine: training self-modifying neural networks with differentiable neuromodulated plasticity (Miconi et al., ICLR 2016)](https://openreview.net/forum?id=r1lrAiA5Ym), a paper from Uber AI labs.

The code is forked from [Salesforce Language model toolkit](https://github.com/Smerity/awd-lstm-lm) and uses most of their parameters and design choices. The main differences are that we do not implement DropConnect and reduce batch size to 6 for computational reasons. This code requires Python 3 and PyTorch 1.0.

To comment, please open an issue. Note that the code is provided "as is": we cannot provide support or accept pull requests at this time.

## Usage

Before running this code, run `getdata.sh` to obtain the Penn Treebank data.

Plasticity and neuromodulation: `python3 main.py --batch_size 6 --data data/penn --dropouti 0.4 --dropouth 0.25  --epoch 500 --save PTB.pt --wdrop 0 --model PLASTICLSTM --modultype modplasth2mod --modulout fanout --nhid 1149  --alphatype perneuron --asgdtime 125 --agdiv 1149`

Plasticity without neuromodulation: `python3 main.py --batch_size 6 --data data/penn --dropouti 0.4 --dropouth 0.25  --epoch 500 --save PTB.pt --wdrop 0 --model PLASTICLSTM --modultype none --modulout none --nhid 1149  --alphatype perneuron --asgdtime 125 --agdiv 1149`

No plasticity, just plain LSTM: `python3 main.py --batch_size 6 --data data/penn --dropouti 0.4 --dropouth 0.25  --epoch 500 --save PTB.pt --wdrop 0 --model MYLSTM --modultype modplasth2mod --modulout fanout --nhid 1150  --alphatype full --asgdtime 125 --agdiv 1150`

Note that in all of the above, we use per-neuron plasticity coefficients and reduce the number of neurons in plastic LSTMs (`nhid`) to ensure that plastic LSTMs do not have more trainable parameters.

## Code organization.

The main program is `main.py`. There is some interface code in `model.py`. The code for actual plastic LSTMs is in `mylstm.py`.

## Plastic LSTMs

The code for plastic LSTMs is relatively straightforward, as can be seen in `mylstm.py`.

However, note that in `main.py` we selectively reduce the gradient for `alpha`
parameters when using plastic LSTMs with either per-neuron or single `alpha`.
More precisely, we divide the gradient on `alpha` coefficients by a value that should be roughly equal
to the number of neurons in the LSTM. This greatly enhances stability without
forcing a reduction in learning rates.



