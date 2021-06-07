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
args = parser.parse_args()


def plot_multiple_results(directories):
    # color_table = ['k', '#ff7c00', '#e8000b', '#1ac938', '#9f4800', '#8b2be2', '#023eff', '#f14cc1','#a3a3a3', '#ffc400', '#00d7ff']
    # Color:        den,   cam,       do,        xanh la,    tim,       brown      xanh nuoc bien
    color_table = ['#1ac938', '#e8000b', '#023eff']
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

        if args.range != -1:
            xs, ys = get_values_with_range(xs, ys, args.range)
        xs, ys = pad(xs), pad(ys)
        assert xs.shape == ys.shape

        usex = xs[0]
        ymean = np.nanmean(ys, axis=0)
        ystd = np.nanstd(ys, axis=0)

        if i == 2:
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
        walker_stand='Walker-stand',
        reacher_hard='Reacher-hard',
        cartpole_swingup_sparse='Cartpole-swingup_sparse',
        cartpole_balance='Cartpole-balance',
        cartpole_balance_sparse='Cartpole-balance_sparse',
    )
    plot_xlabels = dict(
        walker_stand=r'Environment steps ($\times 1e5$)',
        reacher_hard=r'Environment steps ($\times 1e5$)',
        cartpole_swingup_sparse=r'Environment steps ($\times 1e5$)',
        cartpole_balance=r'Environment steps ($\times 1e5$)',
        cartpole_balance_sparse=r'Environment steps ($\times 1e5$)',
    )
    plt.title(env_titles[args.env])
    plt.xlabel(plot_xlabels[args.env])
    plt.ylabel('Episode Return')

    env_lims = dict(
        walker_stand=[[0, 150000], [1, 1050]],
        reacher_hard=[[0, 490000], [1, 950]],
        cartpole_swingup_sparse=[[0, 490000], [1, 850]],
        cartpole_balance=[[0, 100000], [1, 1050]],
        cartpole_balance_sparse=[[0, 250000], [1, 1100]],
    )
    plt.xlim(env_lims[args.env][0][0], env_lims[args.env][0][1])
    plt.ylim(env_lims[args.env][1][0], env_lims[args.env][1][1])

    # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.1f}'.format(x / 1e5)))

    plt.tight_layout()
    plt.savefig('/home/tung/workspace/rlbench/rad/figures/gen_{}.pdf'.format(args.env))
    # plt.show()

if __name__ == '__main__':
    directory = []
    # for i in range(len(args.dir)):
    #     if args.dir[i][-1] == '/':
    #         directory.append(args.dir[i][:-1])
    #     else:
    #         directory.append(args.dir[i])

    experiments = dict(
        walker_stand=[
            's131_logs/walker-stand/crop/',
            's131_logs/walker-stand/crop_from_pre_enc_cpc_lr1e-4_idm_lr1e-4_bs512_50k_samples_50k_train_frozen/',
            's131_logs/walker-stand/crop_from_pre_enc_cpc_lr1e-4_idm_lr1e-4_bs512_50k_samples_50k_train_finetune/'
        ],
        reacher_hard=[
            's41_logs/reacher-hard/crop/',
            's41_logs/reacher-hard/crop_from_pre_enc_cpc_lr1e-3_idm_lr1e-3_bs512_50k_samples_50k_train_frozen/',
            's41_logs/reacher-hard/crop_from_pre_enc_cpc_lr1e-3_idm_lr1e-3_bs512_50k_samples_50k_train_finetune/'
        ],
        cartpole_swingup_sparse=[
            's162_logs/cartpole-swingup_sparse/crop/',
            's162_logs/cartpole-swingup_sparse/crop_from_pre_enc_cpc_lr1e-3_idm_lr1e-3_bs512_50k_samples_50k_train_frozen/',
            's162_logs/cartpole-swingup_sparse/crop_from_pre_enc_cpc_lr1e-3_idm_lr1e-3_bs512_50k_samples_50k_train_finetune/'
        ],
        cartpole_balance=[
            's162_logs/cartpole-balance/crop/',
            's162_logs/cartpole-balance/crop_from_pre_enc_cpc_lr1e-3_idm_lr1e-3_bs512_50k_samples_50k_train_frozen/',
            's162_logs/cartpole-balance/crop_from_pre_enc_cpc_lr1e-3_idm_lr1e-3_bs512_50k_samples_50k_train_finetune/'
        ],
        cartpole_balance_sparse=[
            's162_logs/cartpole-balance_sparse/crop/',
            's162_logs/cartpole-balance_sparse/crop_from_pre_enc_cpc_lr1e-3_idm_lr1e-3_bs512_50k_samples_50k_train_frozen/',
            's162_logs/cartpole-balance_sparse/crop_from_pre_enc_cpc_lr1e-3_idm_lr1e-3_bs512_50k_samples_50k_train_finetune/'
        ]
    )

    env_ranges = dict(
        walker_stand=500000,
        reacher_hard=500000,
        cartpole_swingup_sparse=500000,
        cartpole_balance=500000,
        cartpole_balance_sparse=500000,
    )

    envs = [args.env]
    # Override param
    for env in envs:
        args.range = env_ranges[env]
    for env in envs:
        folder = experiments[env]
        plot_multiple_results(folder)
