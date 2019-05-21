from collections import deque
import numpy as np
import gym, gym.spaces


def rollout(env, agent, render=False, timestep_limit=1000, partial=False, hist_size=25):
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
            obs = np.asarray(history).reshape(1, -1)
            action = agent.pick_action(obs)
        else:
            action = agent.pick_action(obs)

        new_obs, rew, done, info = env.step(action)
        if partial:
            history.append(new_obs)
            new_obs = np.asarray(history).reshape(1, -1)
        ep_rew += rew

        # Store transition
        transition = deque((obs, action, rew, new_obs, done, info))
        yield transition

        if done:
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

    num_timeout = 0.0
    num_success = 0.0
    num_collisions = 0.0

    num_ep = 0.0
    while True:
        for transition in rollout(env, agent, render):

            info = transition.pop()
            if transition[-1]:
                num_ep += 1.0
                if info == 1:
                    num_success += 1.0
                elif info == 2:
                    num_timeout += 1.0
                elif info == 3:
                    num_collisions += 1.0
            data.append(transition)
            num_transitions += 1

        if num_transitions > min_transitions:
            if num_ep:
                scale = 100 / num_ep
                print("--------------------------------------------")
                print("Success percentage: %f" % (num_success * scale))
                print("Collision percentage: %f" % (num_collisions * scale))
                print("Timeout percentage: %f" % (num_timeout * scale))
                print("--------------------------------------------")
                filename = '/tmp/rl_success.txt'
                with open(filename, 'a') as f:
                    f.write("\n%f" % (num_success * scale))
            break

    return data
