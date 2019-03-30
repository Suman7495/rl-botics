from collections import deque
import numpy as np


def rollout(env, agent, render=True, timestep_limit=1000):
    """
        Execute one episode
    """
    data = deque(maxlen=timestep_limit)
    obs = env.reset()
    for t in range(timestep_limit):
        if render:
            env.render()
        action = agent.pick_action(obs)
        new_obs, rew, done, info = env.step(action)

        # Store transition
        transition = deque((obs, action, rew, new_obs, done))
        data.append(transition)

        if done:
            #print("Terminated after %s timesteps" % t)
            break

        obs = new_obs
    return np.array(data)


def get_trajectories(env, agent, num_path_limit = 256):
    """
    :param env: Environment
    :param agent: Policy pi
    :return: Trajectories
             Each trajectory contains:
             [1] obs: Observation of the current state
             [2] action: Action taken at the current state
             [3] rew: Reward collected
             [4] new_obs: New observation of the next state
             [5] done: Boolean to determine if episode has completed

    """
    keys = ["obs", "act", "rew", "new_obs", "done"]
    paths = {keys[0]: [], keys[1]: [], keys[2]: [], keys[3]: [], keys[4]: []}
    num_paths = 0
    while True:
        path = rollout(env, agent)
        for c, v in enumerate(keys):
            paths[v].append(path[:, c])
        num_paths += path.shape[0]
        if num_paths > num_path_limit:
            break

    return paths
