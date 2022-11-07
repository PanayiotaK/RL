# RL

In the code bellow I have implemented a Rainbow Cnn DQN agent with prioritized replay buffer, gradient clipping,  
double-q learning  and for pre-processing the game frame is resized to 84 x 84. 
According to the [paper](https://arxiv.org/pdf/1710.02298.pdf ) rainbow DQN is able to combine different
rl techniques(e.g. noisy linear layers for exploration, multi-step learning , dueling networks etc) to produce
state-of-the-art results in atari games. Also, the prioritized replay buffer is a great improvement for 
gravitar as the rewards in the game are rare. The prioritised buffer makes sure that those rewards have high 
significance. 

the code is based on https://github.com/seungeunrho/minimalRL/blob/master/dqn.py, https://github.com/higgsfield/RL-Adventure
the prioritized replay buffer is based on:  https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
double-q learning: https://github.com/higgsfield/RL-Adventure

