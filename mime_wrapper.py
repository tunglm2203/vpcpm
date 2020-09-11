import gym
from gym import core, spaces
import numpy as np
import mime     # Import this package to register all UR5 environments to gym
from collections import OrderedDict


STATE_KEY = ['joint_position',
             'tool_position',
             'tool_orientation',
             'linear_velocity',
             'angular_velocity',
             'grip_velocity',
             'distance_to_goal',
             'target_position']

IMAGE_KEY = ['rgb0', 'depth0', 'mask0']

ACTION_DICT = OrderedDict(grip_velocity=1,
                          linear_velocity=3)    # format: key: length of vector

ENV_SUPPORT = ['UR5-PickEnv-v0',
               'UR5-PickCamEnv-v0', 'UR5-EGL-PickCamEnv-v0',
               'UR5-Pick5RandCamEnv-v0', 'UR5-EGL-Pick5RandCamEnv-v0']

def _flatten_dict_by_keys(obs, keys):
    obs_pieces = []
    for key in keys:
        if np.isscalar(obs[key]):
            flat = [obs[key]]
        else:
            flat = list(obs[key])
        obs_pieces.extend(flat)
    return np.array(obs_pieces)


def _convert_dict2vec_space(dict_of_box):
    mins, maxs = [], []
    for key in dict_of_box.spaces.keys():
        min = dict_of_box.spaces[key].low.tolist()
        max = dict_of_box.spaces[key].high.tolist()
        mins.extend(min)
        maxs.extend(max)
    low = np.array(mins)
    high = np.array(maxs)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _convert_vec2dict_action(action, action_dict):
    _action = OrderedDict()
    last_idx = 0
    for k, v in action_dict.items():
        _action.update({k: action[last_idx:last_idx + v]})
        last_idx = v

    return _action

class MimeWrapper(core.Env):
    def __init__(self,
                 env_name,
                 from_pixels=False,
                 height=240,
                 width=240,
                 frame_skip=1,
                 channels_first=False,):
        assert env_name in ENV_SUPPORT, 'Environment {} is not supported'.format(env_name)

        if 'Pick' in env_name:
            self._max_episode_steps = 200
        elif 'Push' in env_name:
            self._max_episode_steps = 800
        elif 'Tower' in env_name:
            self._max_episode_steps = 1500
        elif 'Pour' in env_name:
            self._max_episode_steps = 400
        elif 'Bowl' in env_name:
            self._max_episode_steps = 600
        else:
            raise NotImplementedError('No support environments')

        if 'Cam' not in env_name:
            assert from_pixels is False, 'Env {} does not support pixel-based state'.format(env_name)

        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._channels_first = channels_first
        self._frame_skip = frame_skip

        # Create task
        self._env = gym.make(env_name)
        self._env.unwrapped._cam_resolution = (self._height, self._width)

        # Create observation space
        _tmp = self._env.reset()
        _obs = None
        if self._from_pixels:
            _obs = _tmp[IMAGE_KEY[0]]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=_obs.shape, dtype=np.uint8
            )
        else:
            _obs = _flatten_dict_by_keys(_tmp, STATE_KEY)
            self._observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=_obs.shape, dtype=np.float32
            )

        # Create true and normalized action spaces
        self._true_action_space = _convert_dict2vec_space(self._env.action_space)
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        self.current_state = None

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_norm2true_action(action)
        assert self._true_action_space.contains(action)
        action = _convert_vec2dict_action(action, ACTION_DICT)
        assert self._env.action_space.contains(action)

        next_obs, reward, done, info = None, 0, False, {}
        for _ in range(self._frame_skip):
            next_obs, rew, done, info = self._env.step(action)
            reward += rew
            if done:
                break
        obs = self._get_obs(next_obs)
        self.current_state = obs

        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs = self._get_obs(obs)
        self.current_state = obs
        return obs

    def _convert_norm2true_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    def _get_obs(self, obs_full):
        if self._from_pixels:
            obs = obs_full[IMAGE_KEY[0]]
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_dict_by_keys(obs_full, STATE_KEY)
        return obs