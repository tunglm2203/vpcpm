from rlbench.gym.rlbench_wrapper import RLBenchWrapper_v1
import numpy as np
import time
import argparse
import glob
import json
import os

import torch
import utils
from train_rlbench import make_agent


def parse_args():
    parser = argparse.ArgumentParser()

    # Testing arguments
    parser.add_argument('dir', default='.', type=str)
    parser.add_argument('--n_tests', default=10, type=int)
    parser.add_argument('--step', default=None, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--render', default=False, action='store_true')

    # environment
    parser.add_argument('--domain_name', default='panda')
    parser.add_argument('--task_name', default='reach_target-state-v0')
    parser.add_argument('--agent', default='rad_sac', type=str)

    parser.add_argument('--padding_random_crop', default=False, action='store_true')
    parser.add_argument('--use_depth', default=False, action='store_true')
    parser.add_argument('--use_rgb', default=False, action='store_true')
    parser.add_argument('--channels_first', default=True, action='store_false')
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
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

"""
Example to run:
python run_policy.py ./logs/../pixel-baseline-200k-stack3-harder-env-rgb-crop-s1-2020_10_12_22_46_25 --step 100000
if `step` is not provided, it takes the last step
"""

def get_args_from_checkpoint(path):
    json_file = os.path.join(path, 'args.json')
    with open(json_file) as f:
        ckpt_args = json.load(f)
    return ckpt_args


def update_args(src_args_dict, des_args):
    if des_args.seed is None:
        args.seed = np.random.randint(1, 1000000)
        print('[INFO] Seed: ', args.seed)

    if args.step is None:
        all_ckpt_actors = glob.glob(os.path.join(des_args.dir, 'model', 'actor_*'))
        ckpt_step = []
        for ckpt in all_ckpt_actors:
            ckpt_step.append(int(ckpt.split('/')[-1].split('.')[0].split('_')[-1]))
        args.step = max(ckpt_step)
    assert isinstance(src_args_dict, dict), 'src_args_dict must be dictionary for paring'
    exclude_args = ['seed', 'dir', 'n_tests', 'step']
    for arg in src_args_dict.keys():
        if arg in exclude_args or arg not in des_args.__dict__.keys():
            continue
        des_args.__dict__[arg] = src_args_dict[arg]
    return des_args

def main(args):
    ckpt_args = get_args_from_checkpoint(args.dir)
    args = update_args(ckpt_args, args)
    channels_first = ckpt_args['channels_first']
    if 'padding_random_crop' in ckpt_args.keys() and ckpt_args['padding_random_crop']:
        pre_transform_image_size = ckpt_args['image_size']
    else:
        pre_transform_image_size = ckpt_args['pre_transform_image_size'] if 'crop' in ckpt_args['data_augs'] else ckpt_args['image_size']

    utils.set_seed_everywhere(args.seed)

    env = RLBenchWrapper_v1(
        args.task_name,
        seed=args.seed,
        action_scale=0.05,
        height=pre_transform_image_size,
        width=pre_transform_image_size,
        frame_skip=args.action_repeat,  # args.action_repeat
        channels_first=channels_first,
        pixel_normalize=False,
        render=args.render,
        use_depth=args.use_depth,
        use_rgb=args.use_rgb
    )
    env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        if args.use_rgb and args.use_depth:
            n_channels = 4
            print('\n[INFO] Using RGB-D image.')
        elif args.use_rgb and not args.use_depth:
            n_channels = 3
            print('\n[INFO] Using RGB image.')
        elif not args.use_rgb and args.use_depth:
            n_channels = 1
            print('\n[INFO] Using D-only image.')
        else:
            raise NotImplementedError
        obs_shape = (n_channels * args.frame_stack, args.image_size, args.image_size)
    else:
        obs_shape = env.observation_space.shape

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    model_dir = os.path.join(args.dir, 'model')
    agent.load(model_dir=model_dir, step=args.step)

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
    print("eval/eval_time: %.4f (s)" % (time.time() - start_time))
    mean_ep_reward = np.mean(all_ep_rewards)
    std_ep_reward = np.std(all_ep_rewards)
    best_ep_reward = np.max(all_ep_rewards)
    print("eval/episode_reward: mean=%.4f/std=%.4f" % (mean_ep_reward, std_ep_reward))
    print("eval/best_episode_reward: %.4f" % best_ep_reward)
    print('eval/success_rate_of_%s_episodes: %.4f\n' % (n_tests, success_rate))



if __name__ == '__main__':
    args = parse_args()
    main(args)
