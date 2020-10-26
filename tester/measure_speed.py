from rlbench.gym.rlbench_wrapper import RLBenchWrapper_v1
import gym
import dmc2gym
from procgen import ProcgenEnv

from tqdm import tqdm
import numpy as np
import time
import os


def speedtest(env, n_steps=100000, framework=None, summary=dict()):
    env.reset()
    if framework in ['procgen']:
        a_max = env.action_space.n
        a_rand = np.random.randint(a_max, size=1)
    else:
        a_dim = env.action_space.shape[0]
        a_rand = np.random.uniform(-1., 1., a_dim)
    start = time.time()
    for i in tqdm(range(n_steps)):
        _, _, d, _ = env.step(a_rand)
        if d:
            env.reset()

    duration = time.time() - start
    time_per_step = duration/n_steps

    summary.update(
        dict(time_per_step=time_per_step,
             duration=duration)
    )
    del(env)
    return summary

def make_env(framework, env_name, env_args):
    env_args = env_args[framework]
    env = None
    if framework == 'rlbench':
        env = RLBenchWrapper_v1(env_name,
                                frame_skip=1,
                                action_scale=0.05,
                                height=100,
                                width=100,
                                seed=1,
                                channels_first=False,
                                use_gripper=False,
                                z_offset=0.0,
                                **env_args)
    elif framework == 'gym':
        env = gym.make(env_name)
    elif framework == 'dmcontrol':
        domain = env_name[0]
        task = env_name[1]
        env = dmc2gym.make(
            domain_name=domain,
            task_name=task,
            seed=1,
            visualize_reward=False,
            height=100,
            width=100,
            frame_skip=1,
            **env_args
        )
    elif framework == 'procgen':
        env = ProcgenEnv(**env_args)
    else:
        NotImplementedError()

    return env


if __name__ == '__main__':
    n_steps = 50000

    env_args=dict(
        rlbench=dict(
            render=False,
            use_rgb=True,
            use_depth=True,
        ),
        gym=dict(),
        dmcontrol=dict(
            from_pixels=False,
        ),
        procgen=dict(
            num_envs=1,
            env_name='bigfish',
            num_levels=200,
            start_level=0,
            distribution_mode='easy',
            rand_seed=1,
        )
    )

    framework_ut = ['rlbench', 'gym', 'dmcontrol', 'procgen']

    env_ut = dict(
        # rlbench=['reach_target-state-v0', 'reach_target-vision-v0', 'push_button-state-v0', 'push_button-vision-v0'],
        rlbench=['reach_target-vision-v0'],
        # rlbench=[],
        # gym=['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1', 'FetchSlide-v1'],
        gym=[],
        # dmcontrol=[['cartpole', 'swingup'], ['finger', 'spin'], ['reacher', 'easy'], ['cheetah', 'run']],
        dmcontrol=[],
        # procgen=['bigfish', 'coinrun', 'starpilot', 'bossfight'],
        procgen=[]
    )

    dir_name = 'speed_results'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    all_summaries = []
    for framework in framework_ut:
        for env_name in env_ut[framework]:
            if isinstance(env_name, list):
                print('\nDomain: {}, task: {}\n'.format(env_name[0], env_name[1]))
            else:
                print('\nenv_name\n', env_name)
            env = make_env(framework, env_name, env_args)
            summary = {}
            summary = speedtest(env=env, n_steps=n_steps, framework=framework, summary=summary)

            all_summaries.append(summary)
            if isinstance(env_name, list):
                file_name = 'speedtest_{}_{}.txt'.format(env_name[0], env_name[1])
            else:
                file_name = 'speedtest_{}.txt'.format(env_name.replace('-', '_'))
            file = open(os.path.join(dir_name, file_name), 'w')
            print('\n*************** TIME CONSUMING ***************')
            for k in summary.keys():
                str_print = "%s: %.5f" % (k, summary[k])
                print(str_print)
                file.write("%s\n" % str_print)
            file.close()

    print('\n*************** SUMMARY: TIME CONSUMING ***************')
    i = 0
    for framework in framework_ut:
        for env_name in env_ut[framework]:
            print('Framework: %20s, Env: %30s, time/step: %.5f' % (framework, env_name, all_summaries[i]['time_per_step']))
            i += 1

