# General parameters
env_name        = 'CartPole-v0'
gamma           = 0.99
lr              = 0.001
num_ep          = 1e2
render          = False

# DQN specific
eps             = 0.1
min_eps         = 0.1
eps_decay       = 0.995
batch_size      = 32
buffer_size  = 2000

# Policy Network
hidden_sizes    = [64, 64] #Dimension has to be one less than activations and layertypes
activations     = ['relu', 'relu', 'linear']
layer_types     = ['dense', 'dense', 'dense']
loss            = 'mse'