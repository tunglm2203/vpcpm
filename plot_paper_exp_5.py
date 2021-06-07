import argparse
import ast
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob2
import matplotlib.ticker as ticker

from baselines.common import plot_util
from matplotlib.ticker import StrMethodFormatter
from plot_dmc_v1 import plot_drq_results, get_data_in_subdir, pad, get_values_with_range

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, nargs='+')
parser.add_argument('--radius', type=int, default=0)
parser.add_argument('--range', type=int, default=-1, help='Number of transitions want to plot')
parser.add_argument('--legend', type=str, default='', nargs='+')
parser.add_argument('--ylim', type=float, default=0)
parser.add_argument('--title', type=str, default='')
parser.add_argument('--shaded_std', type=bool, default=True)
parser.add_argument('--shaded_err', type=bool, default=False)
parser.add_argument('--train_test', action='store_true')

parser.add_argument('--env', type=str, default='cheetah_run')

parser.add_argument('--ar', type=int, nargs='+', default=None, help='Multiple w/ action repeat')

args = parser.parse_args()


def plot_multiple_results(directories):
    # color_table = ['k', '#ff7c00', '#e8000b', '#1ac938', '#9f4800', '#8b2be2', '#023eff', '#f14cc1','#a3a3a3', '#ffc400', '#00d7ff']
    # Color:        den,   cam,       do,        xanh la,    tim,       brown      xanh nuoc bien
    color_table = ['#1ac938', '#023eff', '#e8000b', '#023eff']
    linestyle = ['-', '-', '-', '-']

    # User config:
    x_key = 'step'
    y_key = 'mean_episode_reward'

    rc = {'axes.facecolor': 'white',
          'legend.fontsize': 15,
          'axes.titlesize': 15,
          'axes.labelsize': 15,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'xtick.direction': 'in',
          'ytick.direction': 'in',
          'axes.formatter.useoffset': False,
          'axes.formatter.offset_threshold': 1}

    plt.rcParams.update(rc)

    fig, ax = plt.subplots()

    collect_data, plot_titles, info_envs = [], [], []
    for directory in directories:
        print(directory)
        data_in_subdir, task_name, info_env = get_data_in_subdir(directory, x_key, y_key)
        collect_data.append(data_in_subdir)
        plot_titles.append(task_name)
        info_envs.append(info_env)

    # Plot data.
    for i in range(len(collect_data)):
        xs, ys = collect_data[i]
        n_experiments = len(xs)

        if args.ar is not None and i in args.ar:
            if info_envs[i] is not None:
                for exp_i in range(n_experiments):
                    xs[exp_i] = xs[exp_i] * info_envs[i]['action_repeat']

        if args.range != -1:
            xs, ys = get_values_with_range(xs, ys, args.range)
        xs, ys = pad(xs), pad(ys)
        assert xs.shape == ys.shape

        usex = xs[0]
        ymean = np.nanmean(ys, axis=0)
        ystd = np.nanstd(ys, axis=0)

        if i == 3:
            plt.plot(usex, ymean, color=color_table[i], linestyle=linestyle[i], linewidth=2)
        else:
            plt.plot(usex, ymean, color=color_table[i], linestyle=linestyle[i])
        if args.shaded_std:
            plt.fill_between(usex, ymean - ystd, ymean + ystd, alpha=0.1, color=color_table[i])

    # plt.grid(True, which='major', color='grey', linestyle='--')

    if args.legend != '':
        assert len(args.legend) == len(
            directories), "Provided legend is not match with number of directories"
        legend_name = args.legend
    else:
        legend_name = [directories[i].split('/')[-1] for i in range(len(directories))]

    plt.legend(legend_name, loc='lower right', fontsize='x-large')
    # plt.legend(legend_name, loc='lower right', frameon=True,
    #            facecolor='#f2f2f2', edgecolor='grey')
    # plt.legend(legend_name, loc='lower right', frameon=True)
    plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    env_titles = dict(
        finger_spin='Finger-spin',
        cartpole_swingup='Cartpole-swingup',
        reacher_easy='Reacher-easy',
        cheetah_run='Cheetah-run',
        walker_walk='Walker-walk',
        walker_stand='Walker-stand',
        ball_in_cup_catch='Ball_in_cup-catch'
    )
    plot_xlabels = dict(
        finger_spin=r'Environment steps ($\times 1e5$)',
        cartpole_swingup=r'Environment steps ($\times 1e5$)',
        reacher_easy=r'Environment steps ($\times 1e5$)',
        cheetah_run=r'Environment steps ($\times 1e5$)',
        walker_walk=r'Environment steps ($\times 1e5$)',
        walker_stand=r'Environment steps ($\times 1e5$)',
        ball_in_cup_catch=r'Environment steps ($\times 1e5$)'
    )
    plt.title(env_titles[args.env])
    plt.xlabel(plot_xlabels[args.env])
    plt.ylabel('Episode Return')

    env_lims = dict(
        finger_spin=[[0, 200000], [1, 1050]],
        cartpole_swingup=[[0, 300000], [1, 950]],
        reacher_easy=[[0, 400000], [1, 1050]],
        ball_in_cup_catch=[[0, 300000], [1, 1050]],
        cheetah_run=[[0, 500000], [1, 600]],
        walker_walk=[[0, 500000], [1, 1050]],
        walker_stand=[[0, 300000], [1, 1050]],
    )
    plt.xlim(env_lims[args.env][0][0], env_lims[args.env][0][1])
    plt.ylim(env_lims[args.env][1][0], env_lims[args.env][1][1])

    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.1f}'.format(x / 1e5)))

    plt.tight_layout()
    plt.savefig('/home/tung/workspace/rlbench/rad/figures/frozen_alt_{}.pdf'.format(args.env))
    # plt.show()

if __name__ == '__main__':
    directory = []
    # for i in range(len(args.dir)):
    #     if args.dir[i][-1] == '/':
    #         directory.append(args.dir[i][:-1])
    #     else:
    #         directory.append(args.dir[i])

    experiments = dict(
        finger_spin=[
            '../sac_baselines/s131_logs/finger-spin/sac_cpm_alternative/',
            's131_logs/finger-spin/crop_from_pre_enc_cpc_lr1e-3_idm_lr1e-3_bs512_50k_samples_50k_train_frozen/'
        ],
        cartpole_swingup=[
            '../sac_baselines/s34_logs/cartpole-swingup/sac_cpm_alt/',
            's162_logs/cartpole-swingup/crop_from_pre_enc_cpc_lr1e-3_idm_lr1e-3_bs512_50k_samples_50k_train_frozen/'
        ],
        reacher_easy=[
            '../sac_baselines/s34_logs/reacher-easy/sac_cpm_alt/',
            's162_logs/reacher-easy/crop_from_pre_enc_cpc_lr1e-3_idm_lr1e-3_bs512_50k_samples_50k_train_frozen/'
        ],
        ball_in_cup_catch=[
            '../sac_baselines/s34_logs/ball_in_cup-catch/sac_cpm_alt/',
            's41_logs/ball_in_cup-catch/crop_from_pre_enc_cpc_lr1e-3_idm_lr1e-3_bs512_50k_samples_50k_train_frozen/'
        ],
        cheetah_run=[
            '../sac_baselines/s41_logs/cheetah-run/sac_cpm_alt/',
            'logs/cheetah-run/crop_bs512_from_pre_enc_cpc_lr2e-4_idm_lr2e-4_bs512_50k_samples_50k_train_frozen/'
        ],
        walker_walk=[
            '../sac_baselines/logs/walker-walk/icml21/sac_cpm_alternative/',
            's131_logs/walker-walk/crop_from_pre_enc_cpc_lr1e-4_idm_lr1e-4_bs512_50k_samples_50k_train_frozen/'
        ],
        walker_stand=[
            '../sac_baselines/s131_logs/walker-stand/sac_cpm_alt/',
            'logs/walker-stand/crop_from_pre_enc_cpc_lr1e-4_idm_lr1e-4_bs512_50k_samples_50k_train_frozen/'
        ]
    )

    env_ranges = dict(
        finger_spin=500000,
        cartpole_swingup=500000,
        reacher_easy=500000,
        ball_in_cup_catch=500000,
        cheetah_run=500000,
        walker_walk=500000,
        walker_stand=500000,
    )

    envs = [args.env]
    # Override param
    for env in envs:
        args.range = env_ranges[env]
    for env in envs:
        folder = experiments[env]
        plot_multiple_results(folder)
