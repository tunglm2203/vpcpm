import argparse
import ast
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob2

from baselines.common import plot_util


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, nargs='+')
parser.add_argument('--radius', type=int, default=0)
parser.add_argument('--range', type=int, default=-1, help='Number of transitions want to plot')
parser.add_argument('--legend', type=str, default='', nargs='+')
parser.add_argument('--shaded_std', type=bool, default=True)
parser.add_argument('--shaded_err', type=bool, default=False)
parser.add_argument('--train_test', action='store_true')
args = parser.parse_args()


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)

        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


def get_data_in_subdir(parent_path, x_key, y_key):
    child_paths = [os.path.abspath(os.path.join(path, '..'))
                   for path in glob2.glob(os.path.join(parent_path, '**', 'eval.log'))]

    data_in_subdir = []
    for path in child_paths:
        json_file = os.path.join(path, 'eval.log')
        data = []
        for line in open(json_file, 'r'):
            data.append(json.loads(line))

        len_data = len(data)
        x, y = [], []
        for i in range(len_data):
            x.append(data[i][x_key])
            y.append(data[i][y_key])
        x = np.array(x)
        y = np.array(y)
        y = plot_util.smooth(y, radius=args.radius)
        data_in_subdir.append((x, y))

    info_env = get_info_env(child_paths[0])
    plot_info = info_env['domain_name'] + '-' + info_env['task_name'] + '-' + info_env['data_augs']

    return data_in_subdir, plot_info


def get_info_env(path):
    info = dict(
        domain_name=None,
        task_name=None,
        data_augs=None
    )
    json_file = os.path.join(path, 'args.json')
    with open(json_file, 'r') as f:
        data = json.load(f)
    for k in info.keys():
        info[k] = data[k]
    return info

def plot_multiple_results(directories):
    x_key = 'step'
    y_key = 'mean_episode_reward'

    collect_data = []
    plot_titles = []
    for directory in directories:
        data_in_subdir, plot_info = get_data_in_subdir(directory, x_key, y_key)
        collect_data.append(data_in_subdir)
        plot_titles.append(plot_info)

    # Plot data.
    for i in range(len(collect_data)):
        data = collect_data[i]
        xs, ys = zip(*data)
        n_experiments = len(xs)

        if args.range != -1:
            _xs = []
            _ys = []
            for k in range(n_experiments):
                _xs.append(xs[k][:args.range])
                _ys.append(ys[k][:args.range])
            xs = _xs
            ys = _ys
        xs, ys = pad(xs), pad(ys)
        assert xs.shape == ys.shape

        usex = xs[0]
        ymean = np.nanmean(ys, axis=0)
        ystd = np.nanstd(ys, axis=0)
        ystderr = ystd / np.sqrt(len(ys))
        plt.plot(usex, ymean, label='config')
        if args.shaded_err:
            plt.fill_between(usex, ymean - ystderr, ymean + ystderr, alpha=0.4)
        if args.shaded_std:
            plt.fill_between(usex, ymean - ystd, ymean + ystd, alpha=0.2)
        plt.title(plot_titles[i], fontsize='x-large')
        plt.xlabel('Number of steps', fontsize='x-large')
        plt.ylabel('Episode reward', fontsize='x-large')

    plt.tight_layout()
    if args.legend != '':
        assert len(args.legend) == len(
            directories), "Provided legend is not match with number of directories"
        legend_name = args.legend
    else:
        legend_name = [directories[i].split('/')[-1] for i in range(len(directories))]

    plt.legend(legend_name, loc='best', fontsize='x-large')
    # plt.legend(legend_name, loc='upper left', fontsize='x-large')
    plt.show()

if __name__ == '__main__':
    directory = []
    for i in range(len(args.dir)):
        if args.dir[i][-1] == '/':
            directory.append(args.dir[i][:-1])
        else:
            directory.append(args.dir[i])
    plot_multiple_results(directory)