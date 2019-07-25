# Grid Maze task

The agent's task is to hit the (invisible) reward location as many times as
possible within a fixed number time steps. Because the reward location is
randomized at the start of each episode, and the agent is randomly teleported
every time it hits the reward, the agent must discover and memorize the reward
location for each episode.

The agent's only inputs consist of a 3x3 neighborhood around the agent's
location, as well as the reward obtained (if any) and the action chosen at the
previous time step.

The outer-loop metal-learning algorithm is Advantage Actor critic. All
within-episode learning occurs through the self-modulated plasticity of network
connections.

For a simpler (but less flexible) implementation of the same task, see the `simplemaze` directory in this repo.

## Visualizations of agent behavior

We show the behavior of the agent over two successive episodes, after 0 and 200,000 meta-learning iterations. The reward location is indicated only for visualization purposes: it is invisible to the agent.

### Episode 0

![Episode 0](anim0_maze.gif)

### Episode 200,000

![Episode 200,000](anim200K_maze.gif)


## Usage

`python3 batch.py  --eplen 200 --hs 100  --lr 1e-4 --l2 0 --addpw 3 --pe 1000 --blossv 0.1 --bent 0.03 --rew 10 --save_every 1000 --rsp 1 --type modplast --da tanh  --nbiter 200002 --msize 13  --wp 0.0 --bs 30 --gc 4.0 --rngseed 0`

`eplen' is the length of an episode, `hs` is the hidden/recurrent layer size, `bs` is batch size and `gc` is gradient clipping.
`type` can be "modplast" (simple neuromodulation), "modul" (retroactive modulation), "plastic" (non-modulated plasticity) or "rnn" (no plasticity at all, plain rnn).
