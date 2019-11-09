# Report

Following document describes solution algorithm and its results in detail.

Implementation is based on the code provided in Udacity github repo for solving pendulum environment.

## Learning Algorithm

Both tennis players are trained with a single agent and shared memory.
Deep Deterministic Policy Gradient (DDPG) algorithm is used, 
which finds both Q-function and policy using 2 neural networks trained concurrently.

This algorithm works well for continuous action spaces.

Policy network is called "Actor" and returns actions. Q-function network is called "Critic".
Networks are trained on data collected in shared memory aka replay buffer.
Same trick with target/local networks (as in DQN) is used to avoid training networks on direct outputs of themselves.

Additionally, during training the noise is added to agent predictions output to have exporation/exploitation balance. 
Amount of noise is gradually reduced during training.

Additionally, following improvement was implemented: episodes where the score was higher than average were saved to priority replay memory, 
which is used for additional training cycle. 


Following hyper-params used:
 * Experience Replay buffer size: 30000
 * Priority Experience Replay buffer size: 15000
 * Batch size: 128
 * Gamma (discount factor): 0.99
 * Target/local networks update rate: 1e-2
 * Actor learning rate: 1e-3
 * Critic learning rate: 1e-3
 * Gaussian noise initial strength: 0.2
 * Noise decay: 0.9999
 * Networks training frequency: 2 updates (1 on full and 1 on priority replay) / every 2 step

### Actor Network
Actor has following layers:
 * input: 24 features (environment state vector dimension)
 * linear (fully connected) layer: 24 x 196, with ReLU activation
 * Batch norm
 * linear (fully connected) layer: 196 x 196, with ReLU activation
 * Batch norm
 * linear (fully connected) layer: 196 x 2 (action vector size) with Tanh activation

### Critic Network
Critic has following layers:
 * input: 24 features (environment state vector dimension), 2 action vector
 * linear (fully connected) layer: 24 x 196, with ReLU activation
 * Batch norm
 * Action vector is concatenated with layer output
 * linear (fully connected) layer: 198 x 196, with ReLU activation
 * Batch norm
 * linear (fully connected) layer: 196 x 1

## Results
With given algorithm, Actor/Critic architecture and hyperparameters, the environment was solved in 395 episodes.
Same agent can be used to play both rakets.

Following plot shows recorded episodes score and score averaged over 100 episodes:

![DDPG Score Plot](images/plot.png?raw=true "DDPG Score Plot")

## Ideas for Future Work
 * Try different algorithms: MADDPG, PPO
 * Try different network architectures
