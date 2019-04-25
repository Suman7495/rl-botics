# RL-botics
RL-botics is a toolbox with highly optimized implementations of Deep Reinforcement Learning algorithms for robotics developed with Keras and TensorFlow in Python3.

The objective was to have modular, clean and easy to read codebase so that the research community may build on top with ease. The implementations can be integrated with [OpenAI Gym](https://github.com/openai/gym) environments. The majority of the algorithms are Policy Search Methods as the toolbox is targetted for robotic applications.

## Requirements
Requirements:
- [python3 (>=3.5)](https://www.python.org/downloads/)
- [Scipy](https://www.scipy.org/)
- [Numpy](http://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Probability](https://www.tensorflow.org/probability)
- [Keras](https://keras.io/)
- [OpenAI Gym](https://github.com/openai/gym)

### Conda Environment
It is highly recommended to install this package in a virtual environment, such as Miniconda. Please find the Conda installation [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

To create a new conda environment called `RL`:
```
conda create -n RL python=3
```

To activate the environment:
```
source activate RL
```
To deactivate the environment:
```
source deactivate
```
## Installation
To install the package, we recommend cloning the original package:
```
git clone https://github.com/Suman7495/rl-botics.git
cd rl-botics
pip install -e .
```

## Usage
To run any algorithm in the default setting, simply run:
```
cd rl_botics/<algo>/
python main.py
```
For example, to run TRPO:
```
cd rl_botics/trpo/
python main.py
```

Numerous other options can be added too, but it is recommended to modify the hyerperparameters in `hyperparameters.py`.

### Algorithms
The algorithms implemented are:
- [Q-Learning](https://link.springer.com/article/10.1007/BF00992698)
- [Deep Q-Network](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Vanilla Policy Gradient](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
- [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971)
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [Proximal Policy Optmization with Intrinsic Curiosity Module (ICM)](https://arxiv.org/abs/1705.05363)
- [Compatible Natural Policy Gradient](https://arxiv.org/abs/1902.02823)

To be added:
- [Relative Entropy Search](http://jmlr.org/papers/v18/16-142.html)
- [Soft Actor Critic](https://arxiv.org/abs/1801.01290)
- [A3C](https://arxiv.org/abs/1602.01783)

## Toolbox Structure
All the algorithms are in the `rl_botics` directory. Each algorithm specified above has an individual directory.

### Common
The directory `common` contains common modular classes to easily build new algorithms.
- `approximators`: Basic Deep Neural Networks (Dense, Conv, LSTM).
- `data_collection`: Performs rollouts and collect observations and rewards
- `logger`: Log training data and other information
- `plotter`: Plot graphs
- `policies`: Common policies such as Random, Softmax, Parametrized Softmax and Gaussian Policy
- `utils`: Functions to compute the expected return, the Generalized Advantage Estimation (GAE), etc.

### Algorithm Directories
Each algorithm directory contains at least 3 files:
- `main.py`: Main script to run the algorithm
- `hyperparameters.py`: File to contain the default hyperparameters
- `<algo>.py`: Implementation of the algorithm
- `utils.py`: (Optional) File containing some utility functions

Some algorithm directories may have additional files specific to the algorithm.

## Contributing
To contribute to this package, it is recommended to follow this structure:
- The new algorithm directory should at least contain the 3 files mentioned above.
- `main.py` should contain at least the following functions:
  - `main`: Parses input argument, builds the environment and agent, and train the agent.
  - `argparse`: Parses input argument and loads default hyperparameters from `hyperparameter.py`.
- `<algo>.py` should contain at least the following methods:
  - `__init__`: Initializes the classes
  - `_build_graph`: Calls the following methods to build the TensorFlow graph: 
    - `_init_placeholders`: Initialize TensorFlow placeholders
    - `_build_policy`: Build policy TensorFlow graph
    - `_build_value_function`: Build value function TensorFlow graph
    - `_loss`: Build policy loss function TensorFlwo graph
  - `train`: Main training loop called by `main.py`
  - `update_policy`: Update the policy
  - `update_value`: Update the value function
  - `print_results`: Print the training results
  - `process_paths`: (optional) Process collected trajectories to return the feed dictionary for TensorFlow

It is recommended to check the structure of `ppo.py` and follow a similar structure.

## Credits
Suman Pal

## License
MIT License.
