from collections import deque
import numpy as np
import gym, gym.spaces


def rollout(env, agent, render=False, timestep_limit=1000, partial=True, hist_size=25):
    """
        Execute one episode
    """
    obs = env.reset()
    ep_rew = 0
    if partial:
        history = deque(maxlen=hist_size)
        history.append(np.zeros_like(obs))
        history.append(obs)

    for t in range(timestep_limit):
        if render:
            env.render()
        if partial:
            input = np.asarray(history).reshape(-1, obs.shape[0])
            action = agent.pick_action(input)[-1]
        else:
            action = agent.pick_action(obs)

        new_obs, rew, done, info = env.step(action)
        if partial:
            history.append(new_obs)
        ep_rew += rew

        # Store transition
        transition = deque((obs, action, rew, new_obs, done))
        yield transition

        if done:
            # print("Terminated after %s timesteps with reward %s" % (str(t+1), str(ep_rew)))
            break

        obs = new_obs


def get_trajectories(env, agent, render=False, min_transitions=512):
    """
    :param env: Environment
    :param agent: Policy pi
    :param render: Boolean. True to render env
    :param min_transitions: Minimum timesteps to collect per iteration
    :return: Trajectories
             Each trajectory contains:
             [0] obs: Observation of the current state
             [1] action: Action taken at the current state
             [2] rew: Reward collected
             [3] new_obs: New observation of the next state
             [4] done: Boolean to determine if episode has completed

    """
    data = deque()
    num_transitions = 0
    while True:
        for transition in rollout(env, agent, render):
            data.append(transition)
            num_transitions += 1

        if num_transitions > min_transitions:
            break

    return data
