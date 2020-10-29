import json
import numpy as np
from glob import glob
from matplotlib import pylab as plt; plt.rcdefaults()
from baselines.bench import load_results


def plot(experiment_path, save_dir=None, save_name=None,
         limit_steps=None, min_max=True):

    fig = plt.figure(figsize=(20, 10))

    if len(glob(os.path.join(experiment_path, "*monitor*"))) != 0:

        exps = glob(experiment_path)
        print(exps)

        df = load_results(experiment_path)
        df['steps'] = df['l'].cumsum() / 1000000
        df['reward'] = df['r'] / 10
        total_time = df['t'].iloc[-1]
        total_steps = df['l'].sum()
        title = " {:.1f} steps, {:.1f} h, FPS {:.1f}".format(total_steps, total_time / 3600, total_steps / total_time)

        roll = 5
        rdf = df.rolling(roll)

        ax = plt.subplot(1, 1, 1)
        if min_max:
            rdf.max().iloc[0:-1:40].plot('steps', 'r', style='-', ax=ax, legend=False, color="#28B463", alpha=0.65)
            rdf.min().iloc[0:-1:40].plot('steps', 'r', style='-', ax=ax, legend=False, color="#F39C12", alpha=0.65)
        df.rolling(roll).mean().iloc[0:-1:40].plot('steps', 'r',  style='-',  ax=ax,  legend=False)
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(2))
        ax.set_xticks(np.arange(0, 100, 5))
        ax.set_yticks(np.arange(-250, 250, 25))
        ax.set_xlabel('Num steps (M)')
        ax.set_ylabel('Reward')
        ax.grid(True)
        if limit_steps:
            plt.xlim(0, limit_steps)

        plt.title(title)

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.2)

    if not save_dir or not os.path.isdir(save_dir):
        save_dir = "/tmp/"
    if not save_name:
        save_name = "results"

    ax.get_figure().savefig(os.path.join(save_dir, save_name) + ".jpg")
    plt.clf()


if __name__ == "__main__":

    import os
    import argparse

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--log-dir', default='/tmp/',
        help='experiment directory or directory containing '
             'several experiments (default: /tmp/ppo)')
    parser.add_argument(
        '--save-dir', default='/tmp/',
        help='path to desired save directory (default: /tmp/)')
    parser.add_argument(
        '--save-name', default='results',
        help='plot save name (default: results)')
    parser.add_argument(
        '--black-list', action='store', type=str, nargs='*', default=[],
        help="experiments to be ignored. Example: -i item1 item2 -i item3 "
             "(default [])")
    parser.add_argument(
        '--limit-steps', type=int, default=None,
        help='truncate plots at this number of steps (default: None)')
    parser.add_argument(
        '--min-max', action='store_true', default=True,
        help='whether or not to plot rolling window min and max values')

    args = parser.parse_args()
    args.log_dir = os.path.expanduser(args.log_dir)

    args.log_dir = "/tmp/atari_ppo/"
    args.save_dir = "/tmp/"

    plot(experiment_path=args.log_dir,
         save_dir=args.save_dir, save_name=args.save_name,
         limit_steps=args.limit_steps,
         min_max=args.min_max)

    quit()