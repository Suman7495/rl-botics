from rl_botics.envs.table_continuous import *
# from rl_botics.envs.table_continuous_with_time import *


def main():
    """
        Main script
        Default environment: Table-v0
    """
    env = ContinuousTable(max_clean=2,
                          max_dirty=3,
                          max_human=1,
                          partial=True,
                          noise=False,
                          hist_len=1,
                          obj_width=0.1
                          )
    # env = ContinuousTableWithTime()

    obs_dim = env.observation_space.n
    act_dim = env.action_space.n
    num_obj = 6
    moved = []
    obs = env.reset()
    cur_obs = obs[0, 0:num_obj*4].reshape(num_obj, 4)
    ep_rew = 0
    ep_count = 0
    all_ep_rews = []
    success_count = 0
    max_iter = 500
    max_timesteps = 1024
    for iter in range(max_iter):
        print("Iteration: ", iter)
        for t in range(max_timesteps):

            # All clean and dirty objects
            clean = np.where(cur_obs[:, 3] == 1)[0]
            dirty = np.where(cur_obs[:, 3] == 2)[0]

            # Remove objects which are not on the tabl.e
            clean = [obj for obj in clean if obj not in moved]
            dirty = [obj for obj in dirty if obj not in moved]
            if dirty:
                ind = dirty[0]
                moved.append(ind)
                action = dirty[0] + num_obj + 1
            elif clean:
                ind = clean[0]
                moved.append(ind)
                action = clean[0] + 1
            else:
                action = 0
            obs, rew, done, info = env.step(action)
            cur_obs = obs[0, 0:num_obj * 4].reshape(num_obj, 4)
            ep_rew += rew
            if done:
                if info == 1:
                    success_count += 1
                ep_count += 1
                all_ep_rews.append(ep_rew)
                # print("Episode reward: ", ep_rew)
                ep_rew = 0
                moved = []
                env.reset()
                continue
        print("Mean reward: ", np.asarray(all_ep_rews).mean())
        print("Success percentage: ", success_count/ep_count)


if __name__ == '__main__':
    main()