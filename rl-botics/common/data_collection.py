def rollout(env, agent, render=True, timestep_limit=1000):
    """
        Simulate the agent for fixed timesteps
    """
    data = deque(maxlen=timestep_limit)
    obs = env.reset()
    tot_rew = 0
    done = False
    for t in range(timestep_limit):
        t += 1
        if render and t % 50 == 0:
            env.render()
        action = agent.pick_action(obs)
        new_obs, rew, done, info = env.step(action)

        # Store transition
        transition = deque((obs, action, rew, new_obs, done))
        data.append(transition)

        if done:
            print("Terminated after %s timesteps" % t)
            return data

        obs = new_obs

# TODO: add function to perform rollouts