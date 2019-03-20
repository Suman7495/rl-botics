# Overview
This repository contains a set of popular Deep Reinforcement Learning algorithms tested on OpenAI's Gym. The objective was to have simple and clear implementations.

Each folder corresponds to a particular algorithm which contains the required theoretical material to get started.

## Installation
To install the packages locally, the following steps are suggested. 
```
conda create -n RL python=3
source activate RL
pip install -r requirements.txt
git clone https://github.com/Suman7495/rl-botics.git
cd rl-botics
```

To run any of the algorithms, go to the specific directory and run `main.py`. For example:
```
cd COPOS
python main.py
```
Additionally, you can enter optional commands. For example:
```
python main.py --env 'CartPole-v0'
```

## Resources for Reinforcement Learning
Here are some key resources to get started with Reinforcement Learning.
### Books
1. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf) by Richard S. Sutton and Andrew G. Barto.
2. [Reinforcement Learning: State of the Art](https://www.springer.com/it/book/9783642276446)

### Online Courses
1. [RL Course by David Silver](https://www.youtube.com/watch?v=2pWv7GOvuf0)
2. [Deep Reinforcement Learning UC Berkeley](http://rail.eecs.berkeley.edu/deeprlcourse/)
