# Target discovery task

A simple stimulus-response ("SR") association task.

At the start of each episode, we generate four random "cues" (i.e. random
binary vectors of length 20). One of them is randomly chosen as the "target".
Then, we repeatedly show pairs of cues (randomly chosen among the four) in
succession, and ask the network to specify whether one of these two is the
target. If the network's answer is correct, a reward is issued, otherwise
nothing happens. The network's task is to obtain as much reward as possible
during each episode.

Note that the network must identify the target (from reward information alone),
then detect it and respond adequately afterwards. Furthermore, because cues are
shown in pairs, the target can never be fully identified in a single "trial": the
network is forced to integrate information across successive "trials".

The outer-loop metal-learning algorithm is Advantage Actor critic. All
within-episode learning occurs through the self-modulated plasticity of network
connections.

Usage:

`python3 srbatch.py --eplen 120 --hs 200 --lr 1e-4 --l2 0 --pe 500 --bv 0.1 --bent 0.1 --rew 1 --wp 0 --save_every 2000 --type modul --da tanh --clamp 0 --nbiter 200000 --fm 1 --ni 4 --pf .0 --alg A3C --cs 20 --eps 1e-6 --is 0 --bs 30 --gc 2.0 --rngseed 0`


`eplen' is the length of an episode, `hs` is the hidden/recurrent layer size, `bs` is batch size and `gc` is gradient clipping.
`type` can be "modplast" (simple neuromodulation), "modul" (retroactive modulation), "plastic" (non-modulated plasticity) or "rnn" (no plasticity at all, plain rnn).
