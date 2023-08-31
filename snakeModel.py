import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box

DEFAULT_CAMERA_CONFIG = {}

class SnakeEnvironment1(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        model_path,
        forwardWeight=10,
        twiggleWeight=100,
        moveWeight=0.005,
        resetNoiseScale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            forwardWeight,
            twiggleWeight,
            moveWeight,
            resetNoiseScale,
            exclude_current_positions_from_observation,
            **kwargs
        )

        self.forwardWeight = forwardWeight
        self.twiggleWeight = twiggleWeight
        self.moveWeight = moveWeight

        self.resetNoiseScale = resetNoiseScale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(35,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(35,), dtype=np.float64
            )
        MujocoEnv.__init__(
            self, model_path, 4, observation_space=observation_space, **kwargs
        )

    def movePenalty(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):

        self.do_simulation(action, self.frame_skip)

        tip_position = self.data.xpos[25, 1]

        forwardReward = self.forwardWeight * tip_position
        movePenalty = self.moveWeight * np.sum(np.square(action))
        twiggleReward = self.twiggleWeight * np.mean(np.abs(self.data.qvel))

        observation = self._get_obs()
        reward = forwardReward + twiggleReward - movePenalty
        info = {
            "reward_fwd": forwardReward,
            "reward_ctrl": -movePenalty,
            "velocity reward": twiggleReward,
            "tip position": tip_position
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observation =  np.concatenate([position, velocity]).ravel()
        return observation

    def reset_model(self):
        noise_low = -self.resetNoiseScale
        noise_high = self.resetNoiseScale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

