import numpy as np
import time
import argparse

import torch
import utils
from rlbench.gym.rlbench_wrapper import RLBenchWrapper_v1
from train_rlbench import make_agent


def parse_args():
    parser = argparse.ArgumentParser()

    # Testing arguments
    parser.add_argument('--n_tests', default=10, type=int)
    parser.add_argument('--dir', default='.', type=str)
    parser.add_argument('--step', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # environment
    parser.add_argument('--domain_name', default='panda')
    parser.add_argument('--task_name', default='reach_target-state-v0')
    parser.add_argument('--agent', default='rad_sac', type=str)

    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # critic
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.001, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='identity', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-3, type=float)
    parser.add_argument('--alpha_beta', default=0.9, type=float)

    parser.add_argument('--log_interval', default=100, type=int)
    # misc
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    # data augs
    parser.add_argument('--data_augs', default='crop', type=str)

    args = parser.parse_args()
    return args


def main(args):
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1,1000000)
    utils.set_seed_everywhere(args.seed)

    env = RLBenchWrapper_v1(args.task_name, action_scale=0.05, render=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    action_shape = env.action_space.shape
    if args.encoder_type == 'pixel':
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
    else:
        obs_shape = env.observation_space.shape
    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    agent.load(model_dir=args.dir, step=args.step)

    n_tests = args.n_tests
    
    all_ep_rewards = []

    start_time = time.time()

    success_rate = 0
    for i in range(n_tests):
        obs = env.reset()
        done, info = False, {}
        episode_reward = 0
        while not done:
            # center crop image
            if args.encoder_type == 'pixel' and 'crop' in args.data_augs:
                obs = utils.center_crop_image(obs, args.image_size)
            with utils.eval_mode(agent):
                if args.encoder_type == 'pixel':
                    action = agent.select_action(obs / 255.)
                else:
                    action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        if info['success']:
            success_rate += 1
            print('Success.')
        else:
            print('Failure.')
        all_ep_rewards.append(episode_reward)
    success_rate = success_rate / n_tests
    print('eval/eval_time', time.time() - start_time)
    mean_ep_reward = np.mean(all_ep_rewards)
    std_ep_reward = np.std(all_ep_rewards)
    best_ep_reward = np.max(all_ep_rewards)
    print('eval/mean_episode_reward', mean_ep_reward)
    print('eval/std_episode_reward', std_ep_reward)
    print('eval/best_episode_reward', best_ep_reward)
    print('eval/success_rate_of_{}_episodes'.format(n_tests), success_rate)
    print('')



if __name__ == '__main__':
    args = parse_args()
    main(args)
