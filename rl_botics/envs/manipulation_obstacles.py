from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
from gym.spaces import Discrete


class ManipulationObstacles:
    def __init__(self):
        self.new_obs = 0
        self.rew = 0
        self.done = False
        self.info = 0

        self.low = 0.0
        self.high = 1.0

        self.start_loc = np.asarray([0.0, 0.0, 0.0])
        self.goal_loc  = np.asarray([1.0, 1.0, 1.0])

        self.action_space = Discrete(5)
        self.observation_space = Discrete(7)

        # Create plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def _gen_obstacles(self):
        return NotImplementedError

    def reset(self):
        return

    def step(self, action):
        self.start_loc += np.asarray([0.01, 0.01, 0.01])
        return self.new_obs, self.rew, self.done, self.info

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self):
        plt.cla()
        markers = ['o', 'g']
        self.ax.scatter(self.start_loc[0],
                        self.start_loc[1],
                        self.start_loc[2],
                        c='r',
                        marker='o',
                        s=100,
                        label='End effector')
        self.ax.scatter(self.goal_loc[0],
                        self.goal_loc[1],
                        self.goal_loc[2],
                        c='g',
                        marker='^',
                        s=100,
                        label='Goal')

        self.ax.quiver(self.start_loc[0],
                       self.start_loc[1],
                       self.start_loc[2],
                       0.1, 0.1, 0.1)


        self.ax.set_xlim(self.low - 0.1, self.high+0.1)
        self.ax.set_ylim(self.low - 0.1, self.high+0.1)
        self.ax.set_zlim(self.low - 0.1, self.high+0.1)

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        plt.title("Manipulation with Obstacles")
        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)
        plt.locator_params(axis='z', nbins=4)
        plt.tight_layout()
        plt.legend()
        plt.grid(False)
        plt.pause(0.05)
        plt.show(block=False)



if __name__ == "__main__":
    env = ManipulationObstacles()
    env.reset()
    env.render()
    act_dim = env.action_space.n
    obs_dim = env.observation_space.n
    ep_rew = 0.0
    ep_count = 0
    for _ in range(100):
        env.render()
        a = np.random.randint(act_dim)
        obs, rew, done, info = env.step(a)
        # print("Reward: ", rew)
        ep_rew += rew
        if done:
            ep_count += 1
            print("Episode reward: ", ep_rew)
            ep_rew = 0
            env.reset()
            continue