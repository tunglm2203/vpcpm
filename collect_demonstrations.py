import numpy as np
import torch
import argparse
import os
import time
import json
import dmc2gym
import glob
import matplotlib.pyplot as plt

import utils
from train_from_pretrained import make_agent
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    # pre-trained encoder
    parser.add_argument('--dir', default='.', type=str)
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--n_episodes', default=50, type=int)
    parser.add_argument('--render', action='store_true')
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--pre_transform_image_size', default=100, type=int)

    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=2, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='rad_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', default=1000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int) # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--latent_dim', default=128, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--exp', default='exp', type=str)
    parser.add_argument('--work_dir', default='logs_dmc', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    # data augs
    parser.add_argument('--data_augs', default='crop', type=str)

    parser.add_argument('--log_interval', default=100, type=int)
    args = parser.parse_args()
    return args


def get_args_from_checkpoint(path):
    json_file = os.path.join(path, 'args.json')
    with open(json_file) as f:
        ckpt_args = json.load(f)
    return ckpt_args


def update_args(src_args_dict, des_args):
    if des_args.seed is None:
        args.seed = np.random.randint(1, 1000000)
        print('[INFO] Seed: ', args.seed)

    if args.step == -1:
        all_ckpt_actors = glob.glob(os.path.join(des_args.dir, 'model', 'actor_*'))
        ckpt_step = []
        for ckpt in all_ckpt_actors:
            ckpt_step.append(int(ckpt.split('/')[-1].split('.')[0].split('_')[-1]))
        args.step = max(ckpt_step)
    assert isinstance(src_args_dict, dict), 'src_args_dict must be dictionary for paring'
    exclude_args = ['seed', 'dir', 'work_dir', 'step', 'n_episodes', 'pretrained', 'render']
    for arg in src_args_dict.keys():
        if arg in exclude_args or arg not in des_args.__dict__.keys():
            continue
        des_args.__dict__[arg] = src_args_dict[arg]
    return des_args

def imshow(obs, pause=0.001):
    plt.imshow(obs)
    plt.axis('off')
    plt.tight_layout()
    plt.pause(pause)
    plt.show(block=False)

def render(obs, stack_frame=1):
    img = obs[:3, :, :].transpose(1, 2, 0)
    imshow(img)


def make_logdir(args):
    # make directory
    env_name = args.domain_name + '-' + args.task_name
    if args.encoder_type == 'pixel':
        exp_name = env_name + '/' + 'img' + str(args.image_size) + '-s' + str(args.seed)
    elif args.encoder_type == 'identity':
        exp_name = env_name + '/' + 'state' + '-s' + str(args.seed) + '-' + args.encoder_type
    else:
        raise NotImplementedError('Not support: {}'.format(args.encoder_type))

    args.work_dir = args.work_dir + '/demonstrations/' + exp_name + '/buffer'

    utils.make_dir(args.work_dir)


"""
Example to run:
python collect_demonstrations.py --dir ./logs/../pixel-rgb-crop-s1-2020_10_12_22_46_25 --step 100000 \
    --domain_name cheetah --task_name run --n_episodes 50 --render
if `step` is not provided, it takes the last step
"""

def main(args):
    ckpt_args = get_args_from_checkpoint(args.dir)
    args = update_args(ckpt_args, args)

    pre_transform_image_size = args.pre_transform_image_size if 'crop' in args.data_augs else args.image_size
    pre_image_size = args.pre_transform_image_size

    utils.set_seed_everywhere(args.seed)

    print('[INFO] Loading environment...')
    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=pre_transform_image_size,
        width=pre_transform_image_size,
        frame_skip=args.action_repeat
    )
    print('[INFO] Load environment done.')
    env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)

    make_logdir(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3 * args.frame_stack, pre_transform_image_size, pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        pre_image_size=pre_image_size
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device,
    )

    model_dir = os.path.join(args.dir, 'model')
    agent.load(model_dir=model_dir, step=args.step)

    n_episodes = args.n_episodes
    all_ep_rewards = []

    start_time = time.time()
    for i in tqdm(range(n_episodes)):
        obs = env.reset()
        cur_obs = obs.copy()
        if args.render:
            render(obs)
        done = False
        episode_reward, episode_step = 0, 0
        while not done:
            # center crop image
            if args.encoder_type == 'pixel' and 'crop' in args.data_augs:
                obs = utils.center_crop_image(obs, args.image_size)
            if args.encoder_type == 'pixel' and 'translate' in args.data_augs:
                # first crop the center with pre_image_size
                obs = utils.center_crop_image(obs, args.pre_transform_image_size)
                # then translate cropped to center
                obs = utils.center_translate(obs, args.image_size)
            with utils.eval_mode(agent):
                if args.encoder_type == 'pixel':
                    action = agent.select_action(obs / 255.)
                else:
                    action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
                done
            )
            if args.render:
                render(obs)

            next_obs = obs
            replay_buffer.add(cur_obs, action, reward, next_obs, done_bool)

            cur_obs = obs
            episode_step += 1
            episode_reward += reward

        all_ep_rewards.append(episode_reward)

    replay_buffer.save(args.work_dir)
    print("eval/eval_time: %.4f (s)" % (time.time() - start_time))
    mean_ep_reward = np.mean(all_ep_rewards)
    std_ep_reward = np.std(all_ep_rewards)
    best_ep_reward = np.max(all_ep_rewards)
    print("eval/episode_reward: mean=%.4f/std=%.4f" % (mean_ep_reward, std_ep_reward))
    print("eval/best_episode_reward: %.4f" % best_ep_reward)



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    main(args)
