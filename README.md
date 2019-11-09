# Collaboration and Competition Project for Udacity DRLND

This project solves Tennis environment created with Unity.

## Environment Description

In this environment, two agents control rackets to bounce a ball over a net. 
If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. 
Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 
Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 
(over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

  * After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.

  * This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started

Following steps were tested for **Windows 10** with Anaconda installed. For other OS,
please refer to installation steps in the original [DLRND repo](https://github.com/udacity/deep-reinforcement-learning#dependencies).

1. Install [SWIG](https://sourceforge.net/projects/swig/files/swigwin/swigwin-4.0.1/swigwin-4.0.1.zip/download).
Unpack zip archive to any folder, open "Edit system environment variables", edit `Path` to add folder where `swig.exe` is located.

2. Create new Python environment in Anaconda prompt:
    ```
    conda create --name drlnd python=3.6
    activate drlnd
    conda install pytorch=0.4.0 -c pytorch
    ```

3. Checkout code and install dependencies:
    ```
    activate drlnd
    git clone https://github.com/vvmnnnkv/drlnd-competition
    cd drlnd-competition
    pip install -r requirements.txt
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

4. Download the [Tennis environment executable](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip) for Windows (64-bit) and unpack in the project folder.

5. Run jupyter notebook in the project root (this should open browser):
    ```
    jupyter notebook --notebook-dir=.
    ```

## Run Trained Agent
In jupyter, open `Demo.ipynb` and follow instructions.


## Train Agent
In jupyter, open `Train.ipynb` and follow instructions.

