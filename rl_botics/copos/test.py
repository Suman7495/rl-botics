import numpy as np
import os
import tensorflow as tf
from rl_botics.envs.table_continuous import *
from rl_botics.common.utils import load_model


class Policy:
    """
    Load and run trained policy
    """
    def __init__(self):
        self.graph = tf.get_default_graph()
        self.sess = tf.get_default_session()
        self.obs_ph = self.graph.get_tensor_by_name("obs:0")
        self.probs_ph = self.graph.get_tensor_by_name("ParametrizedSoftmax_1/probs:0")

    def pick_action(self, obs):
        feed_dict = {self.obs_ph: obs}
        probs = np.squeeze(self.sess.run(self.probs_ph, feed_dict))
        action = np.argmax(probs)
        return action


def main():
    """
        Main script
        Default environment: CartPole-v0
    """
    env = ContinuousTable(max_clean=3,
                          max_dirty=3,
                          max_human=1,
                          partial=True,
                          noise=True,
                          hist_len=8,
                          obj_width=0.1
                          )
    with tf.Session() as sess:
        load_model(os.getcwd() + "/models/POMDP_Occ_Noise_3_3_1/model.ckpt", sess)
        agent = Policy()

        num_actions = 0
        tot_actions = 0.0
        num_timeout = 0.0
        num_success = 0.0
        num_collisions = 0.0

        max_ep = 100
        render = True

        for i_episode in range(max_ep):
            obs = env.reset()
            while True:
                if render:
                    env.render()
                num_actions += 1
                tot_actions += 1.0
                action = agent.pick_action(obs)
                if action == 0:
                    print("Claimed done")
                elif action > 7:
                    print("Removing obj_id ", action - 8)
                else:
                    print("Moving obj_id ", action-1)
                # print(action)
                obs, reward, done, info = env.step(action)
                # _ = input("Press any key to continue:")
                if done:
                    print("Episode finished after {} timesteps".format(num_actions))
                    num_actions = 0
                    if info == 1:
                        print("Successful episode.")
                        num_success += 1.0
                    elif info == 2:
                        num_timeout += 1.0
                    elif info == 3:
                        num_collisions += 1.0
                    break

        scale = 100 / max_ep
        print("--------------------------------------------")
        print("Success percentage: %f" % (num_success * scale))
        print("Collision percentage: %f" % (num_collisions * scale))
        print("Timeout percentage: %f" % (num_timeout * scale))
        print("Mean number of actions: %f" % (tot_actions/max_ep))
        print("--------------------------------------------")


if __name__ == '__main__':
    main()