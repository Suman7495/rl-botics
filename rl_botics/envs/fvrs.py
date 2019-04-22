import numpy as np

#from rllab.core.serializable import Serializable
#from rllab.envs.base import Step
#from rllab.envs.proxy_env import ProxyEnv
#from rllab.misc.overrides import overrides

import gym
from gym import Wrapper
from gym.spaces import Discrete, Box
from rl_botics.env.gym_pomdp.envs.rock import RockEnv, Obs


class HistoryEnv(Wrapper):

    """
    takes observations from an environment and stacks to history given hist_len of history length
    """

    def __init__(self, env_id, hist_len=4, history_type='standard', kwargs={}):
        """
        Parameters
        ----------
        env_id - id of registered gym environment (currently only implemented for Rock-v0)
        history_type - * one_hot: encodes the actions as one hot vector in the history
                       * one_hot_pos: one hot agent position and history of 'one_hot' observations
                       * standard: encodes the actions as action_index+1 (reason for this is that the initial history is
                         all zeros and we don't want to collide with action 0, which is move north)
                       * standard_pos: one hot agent position and history of 'standard' observations
                       * field_vision: encodes the actions as action_index+1 (reason: see 'standard')
                         and noisy observation for each rock
                       * field_vision_pos: one hot agent position and history of noisy observations for each rock
                       * fully_observable: one hot agent position and history of true observations for each rock
                       * mixed_full_pomdp: flag to indicate if full information is avail + true observations for each rock +
                         one hot agent position and history of 'one_hot' observations
        hist_len - length of the history
        kwargs - optional arguments for initializing the wrapped environment
        """
        if not env_id == "Rock-v0":
            raise NotImplementedError("history only implemented for Rock-v0")
        env = gym.make(env_id)
        env.__init__(**kwargs)
        super(HistoryEnv, self).__init__(env)
        self._wrapped_env = env
        self.hist_len = hist_len
        self.hist_type = history_type
        self.history = None
        self.num_rocks = self._wrapped_env.num_rocks
        self.size_x, self.size_y = self._wrapped_env.grid.get_size
        # for full observations
        # self.historyIgnoreIdx = self._wrapped_env.grid.x_size + self._wrapped_env.grid.y_size
        if self.hist_type == "standard":
            self.historyIgnoreIdx = 0
            self.total_obs_dim = (1+1) # standard obs
            self.observation_space = Box(low=0, high=(4+1)+self.num_rocks, shape=(self.total_obs_dim*self.hist_len,)) # history of: ac + ob pairs
            self.genObservation = self.generateObservationStandard
        elif self.hist_type == "standard_pos":
            self.historyIgnoreIdx = self.size_x + self.size_y
            self.total_obs_dim = self.historyIgnoreIdx+(1+1) # agent pos + standard obs
            self.observation_space = Box(low=0, high=(4+1)+self.num_rocks, shape=(self.historyIgnoreIdx + (1+1)*self.hist_len,)) # agent pos + history of: ac + ob pairs
            self.genObservation = self.generateObservationStandardPos
        elif self.hist_type == "one_hot":
            self.historyIgnoreIdx = 0
            self.nact = self._wrapped_env.action_space.n
            self.total_obs_dim = (self.nact+1) # one hot encoded actaion + single ob
            self.observation_space = Box(low=0, high=len(Obs)-1, shape=(self.total_obs_dim*self.hist_len,)) # history of: one_hot_ac + ob pairs
            self.genObservation = self.generateObservationOneHot
        elif self.hist_type == "one_hot_pos":
            self.historyIgnoreIdx = self.size_x + self.size_y
            self.nact = self._wrapped_env.action_space.n
            self.total_obs_dim = self.historyIgnoreIdx+(self.nact+1) # agent pos + one hot encoded actaion + single ob
            self.observation_space = Box(low=0, high=len(Obs)-1, shape=(self.historyIgnoreIdx + (self.nact+1)*self.hist_len,)) # agent pos + history of: one_hot_ac + ob pairs
            self.genObservation = self.generateObservationOneHotPos
        elif self.hist_type == "field_vision":
            self.historyIgnoreIdx = 0
            self.total_obs_dim = (1+self.num_rocks) # actaion + ob (for each rock)
            self.observation_space = Box(low=0, high=(4+1)+self.num_rocks, shape=(self.total_obs_dim*self.hist_len,)) # history of: ac + ob (for each rock) pairs
            self.genObservation = self.generateObservationFieldVision
        elif self.hist_type == "field_vision_pos":
            self.historyIgnoreIdx = self.size_x + self.size_y
            self.total_obs_dim = (self.historyIgnoreIdx+self.num_rocks) # oneHot agent position + ob (for each rock)
            self.observation_space = Box(low=0, high=len(Obs)-1, shape=(self.historyIgnoreIdx + self.num_rocks*self.hist_len,)) # agent pos + history of: ac + ob (for each rock) pairs
            self.genObservation = self.generateObservationFieldVisionPos
        elif self.hist_type == "fully_observable":
            self.historyIgnoreIdx = self.size_x + self.size_y
            self.total_obs_dim = (self.historyIgnoreIdx+self.num_rocks) # oneHot agent position + ob (for each rock)
            self.observation_space = Box(low=0, high=len(Obs)-1, shape=(self.historyIgnoreIdx + self.num_rocks*self.hist_len,)) # agent pos + history of: ac + ob (for each rock) pairs
            self.genObservation = self.generateObservationFullState
        elif self.hist_type == "mixed_full_pomdp":
            self.historyIgnoreIdx = 1 + self.num_rocks + self.size_x + self.size_y
            self.nact = self._wrapped_env.action_space.n
            self.total_obs_dim = self.historyIgnoreIdx+(self.nact+1) # ignore index + agent pos + one hot encoded actaion + single ob
            self.observation_space = Box(low=0, high=len(Obs)-1, shape=(self.historyIgnoreIdx + (self.nact+1)*self.hist_len,)) # flag + full obs + agent pos + history of: one_hot_ac + ob pairs
            self.genObservation = self.generateObservationMixed
        else:
            raise NameError("error: wrong history type")

        self.observation_dim_hist_part = self.total_obs_dim - self.historyIgnoreIdx
        print('total obs dim:', self.total_obs_dim)
        print('history obs_dim:', self.observation_dim_hist_part)

    def reset_history(self, new_):
        self.history = np.zeros((self.observation_space.shape[0]-self.historyIgnoreIdx, ))
        self.history[0:self.observation_dim_hist_part] = new_[self.historyIgnoreIdx:]
        #self.history_full = np.zeros((self.hist_len*self.total_obs_dim, ))
        #self.history_full[0:self.total_obs_dim] = new_

    def add_to_history(self, new_):
        self.history[self.observation_dim_hist_part:] = self.history[:-self.observation_dim_hist_part]
        self.history[0:self.observation_dim_hist_part] = new_[self.historyIgnoreIdx:]
        #self.history_full[self.total_obs_dim:] = self.history_full[:-self.total_obs_dim]
        #self.history_full[0:self.total_obs_dim] = new_

    def reset(self):
        obs = self._wrapped_env.reset()
        if self.hist_type == "standard":
            new_ob = np.array([np.zeros(1), obs])
        elif self.hist_type == "standard_pos":
            std_ob = np.array([np.zeros(1), obs])
            xpos, ypos = self.generatePosOneHot(False)
            new_ob = np.concatenate([xpos, ypos, std_ob])
        elif self.hist_type == "one_hot":
            new_ob = np.concatenate([np.zeros(self.nact), [obs]])
        elif self.hist_type == "one_hot_pos":
            xpos, ypos = self.generatePosOneHot(False)
            new_ob = np.concatenate([xpos, ypos,np.zeros(self.nact), [obs]])
        elif self.hist_type == "field_vision":
            observation_rocks = self.generateFieldVisionRockObservation(False)
            new_ob = np.concatenate([np.zeros(1), observation_rocks])
        elif self.hist_type == "field_vision_pos":
            observation_rocks = self.generateFieldVisionRockObservation(False)
            xpos, ypos = self.generatePosOneHot(False)
            new_ob = np.concatenate([xpos, ypos, observation_rocks])
        elif self.hist_type == "fully_observable":
            observation_rocks = self.generateTrueRockOvservation(False)
            xpos, ypos = self.generatePosOneHot(False)
            new_ob = np.concatenate([xpos, ypos, observation_rocks])
        elif self.hist_type == "mixed_full_pomdp":
            observation_rocks = self.generateTrueRockOvservation(False)
            xpos, ypos = self.generatePosOneHot(False)
            flag = 1
            new_ob = np.concatenate([[flag],observation_rocks,xpos,ypos,np.zeros(self.nact),[obs]])
        else:
            raise NameError("error: wrong history type")
        self.reset_history(new_ob)
        # we return copy so that we can modify the history without changing already returned histories
        return np.concatenate([new_ob[0:self.historyIgnoreIdx],self.history])
        #return np.copy(self.history)

    def step(self, action):
        next_obs, reward, done, info = self._wrapped_env.step(action)
        ob = self.genObservation(next_obs, action, done)
        self.add_to_history(ob)
        # we return copy so that we can modify the history without changing already returned histories
        return np.concatenate([ob[0:self.historyIgnoreIdx],self.history]), reward, done, info
        #return np.copy(self.history), reward, done, info

    def generateObservationStandard(self, ob, a, done):
        return np.array([a+1, ob])

    def generateObservationStandardPos(self, ob, a, done):
        xpos, ypos = self.generatePosOneHot(done)
        std_ob = np.array([a+1, ob])
        return np.concatenate([xpos,ypos,std_ob])

    def generateObservationOneHot(self, ob, a, done):
        one_hot_a = np.zeros(self.nact, dtype=np.int)
        one_hot_a[int(a)] = 1
        return np.concatenate([one_hot_a, [ob]])

    def generateObservationOneHotPos(self, ob, a, done):
        xpos, ypos = self.generatePosOneHot(done)
        one_hot_a = np.zeros(self.nact, dtype=np.int)
        one_hot_a[int(a)] = 1
        return np.concatenate([xpos,ypos,one_hot_a,[ob]])

    def generateObservationFieldVision(self, ob, a, done):
        # action + noisy value of all rocks
        observation_rocks = self.generateFieldVisionRockObservation(done)
        return np.concatenate([[a+1], observation_rocks])

    def generateObservationFieldVisionPos(self, ob, a, done):
        # agent pos + noisy value of all rocks
        observation_rocks = self.generateFieldVisionRockObservation(done)
        xpos, ypos = self.generatePosOneHot(done)
        return np.concatenate([xpos,ypos,observation_rocks])

    def generateObservationFullState(self, ob, a, done):
        # agent pos + true value of all rocks
        observation_rocks = self.generateTrueRockOvservation(done)
        xpos, ypos = self.generatePosOneHot(done)
        return np.concatenate([xpos,ypos,observation_rocks])

    def generateObservationMixed(self, ob, a, done):
        # flag + true value of all rocks + agent pos + history of: one_hot_ac + noisy ob pairs
        flag = 1
        observation_rocks = self.generateTrueRockOvservation(done)
        xpos, ypos = self.generatePosOneHot(done)
        one_hot_a = np.zeros(self.nact, dtype=np.int)
        one_hot_a[int(a)] = 1
        return np.concatenate([[flag],observation_rocks,xpos,ypos,one_hot_a,[ob]])

    def generateFieldVisionRockObservation(self, done):
        # noisy value of all rocks
        observation_rocks = np.zeros((self.num_rocks,))
        if not done:
            for rock in range(0, self.num_rocks):
                if self._wrapped_env.state.rocks[rock].status == 0:  # collected
                    ob = Obs.NULL.value
                else:
                    ob = self._wrapped_env._sample_ob(self._wrapped_env.state.agent_pos, self._wrapped_env.state.rocks[rock])
                observation_rocks[rock] = ob
        return observation_rocks

    def generateTrueRockOvservation(self, done):
        # true value of all rocks
        observation_rocks = np.zeros((self.num_rocks,))
        if not done:
            for rock in range(0, self.num_rocks):
                rock_status = self._wrapped_env.state.rocks[rock].status
                if rock_status == 1:    #good
                    observation_rocks[rock] = Obs.GOOD.value
                elif rock_status == -1: #bad
                    observation_rocks[rock] = Obs.BAD.value
                else:   # collected
                    observation_rocks[rock] = Obs.NULL.value
        return observation_rocks

    def generatePosOneHot(self, done):
        xpos=np.zeros(self.size_x)
        ypos=np.zeros(self.size_y)
        if not done:
            # one hot encoded x and y position of the agent
            xpos = np.zeros(self.size_x, dtype=np.int)
            xpos[int(self._wrapped_env.state.agent_pos.x)] = 1
            ypos = np.zeros(self.size_y, dtype=np.int)
            ypos[int(self._wrapped_env.state.agent_pos.y)] = 1
        return xpos, ypos