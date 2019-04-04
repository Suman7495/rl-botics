# RL-botics
RL-botics is a toolbox with state-of-the-art implemntations of deep Reinforcement Learning algorithms for robotics. 

The objective was to have modular, clean and easy to read codebase so that the research community may build on top with ease. The implementations can be integrated with OpenAI Gym environments.

## Prerequisites
RL-botics requires:
- [python3 (>=3.5)](https://www.python.org/downloads/)
- [Scipy](https://www.scipy.org/)
- [Numpy](http://www.numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Probability](https://www.tensorflow.org/probability)
- [Keras](https://keras.io/)

It is strongly suggested to also have:
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
## Installation
To install the package, we recommend cloning the original package:
```
git clone https://github.com/Suman7495/rl-botics.git
cd rl-botics
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
### Algorithms
The algorithms implemented are:
- Q-Learning
- Deep Q-Network
- Vanilla Policy Gradient
- Deep Deterministic Policy Gradient
- Trust Region Policy Optimization
- Proximal Policy Optimization
- Compatible Natural Policy Gradient

## Toolbox Structure
All the algorithms are in the `rl_botics` directory. Each algorithm specified above has an individual directory.

### Common
The directory `common` contains common modular classes to easily build new algorithms.
- `approximators`: Basic Deep Neural Networks
- `data_collection`: Methods to interact with the environment and collect data
- `logger`: Log training data and other information
- `plotter`: Plot graphs
- `policies`: Common policies such as Random Policy, Softmax Policy and Gaussian Policy
- `utils`: Functions to compute the expected return and the Generalized Advantage Estimation (GAE)

### Algorithm Directories
Each algorithm directory contains at least 3 files:
- `main.py`: Main script to run the algorithm
- `hyperparameters.py`: File to contain the default hyperparameters
- `<name>.py`: Implementation of the algorithm

Some algorithm directories may have additional files specific to the algorithm.

## Contributing

## Credits
Suman Pal

## License
MIT License.
