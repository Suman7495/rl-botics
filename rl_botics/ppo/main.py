import gym, gym.spaces
import argparse
from ppo import *
import tensorflow as tf
import hyperparameters as h
import rl_botics
import rl_botics.env.gym_pomdp

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default=h.env_name)
    parser.add_argument('--gamma', type=float, default=h.gamma)
    parser.add_argument('--pi_lr', type=float, default=h.pi_lr)
    parser.add_argument('--maxiter', type=int, default=h.maxiter)
    parser.add_argument('--render', type=bool, default=h.render)
    parser.add_argument('--batch_size', type=int, default=h.pi_batch_size)
    parser.add_argument('--cg_damping', type=float, default=h.cg_damping)
    parser.add_argument('--kl_bound', type=float, default=h.kl_bound)
    parser.add_argument('--min_trans_per_iter', type=float, default=h.min_trans_per_iter)
    return parser.parse_args()

def main():
    """
        Main script
        Default environment: CartPole-v0
    """
    args = argparser()
    env = gym.make(args.env)
    with tf.Session() as sess:
        agent = PPO(args, sess, env)
        print("Training agent...\n")
        agent.train()
        agent.print_results()

    # env = gym.make("Rock-v0")
    # ob = env.reset()
    # print(ob)
    # env.render()
    # r = 0
    # discount = 1.
    # for i in range(400):
    #     action =  np.random.choice(env._generate_legal(), 1)[0]
    #     ob, rw, done, info = env.step(action)
    #     print(ob)
    #     env.render()
    #     r += rw * discount
    #     discount *= env._discount
    #     if done:
    #         break
    # print(r)

if __name__ == '__main__':
    main()