"""
将 dm_control 环境和任务 包裹在 Gym 环境中。任务假设 CMU 位置控制的人形代理存在。

改编自：
https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
"""
import os.path as osp
import numpy as np
import tree
import mujoco

from typing import Any, Callable, Dict, Optional, Text, Tuple
from dm_env import TimeStep
from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.mocap import loader
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.tasks.reference_pose import utils
from dm_control.locomotion.walkers import initializers
from dm_control.suite.wrappers import action_noise
from gym import core
from gym import spaces

from envs.tasks.arenas import EmptyCorridor, WallsCorridor, GapsCorridor, StairsCorridor, HurdlesCorridor, Floor
from envs.walkers import cmu_humanoid


class StandInitializer(initializers.WalkerInitializer):
    """
    给 CMU 人形提供“站立”初始姿态，避免每次 reset 随机倒地。
    构造函数里读一段固定 mocap 片段（CMU_040_12 第 0 帧）作为站立模板。
    initialize_pose() 把 walker 的关节、根位姿直接设成该模板，再跑一遍 MuJoCo 正运动学确保坐标系一致。
    """
    def __init__(self):
        # 负责解析内置的 CMU 运动捕捉数据集，返回一个本地 .hdf5文件
        ref_path = cmu_mocap_data.get_path_for_cmu(version='2020')

        # 把 .hdf5 文件包装成可查询的对象
        mocap_loader = loader.HDF5TrajectoryLoader(ref_path)

        # CMU_040_12 是 CMU 数据集中的一条具体动作（040：subject 40，12：第 12 段动作）
        trajectory = mocap_loader.get_trajectory('CMU_040_12')

        # 转成字典并去掉前缀'walker/' strip:夺去
        clip_reference_features = trajectory.as_dict()
        clip_reference_features = tracking._strip_reference_prefix(clip_reference_features, 'walker/')

        # 把整条轨迹第 0 帧抽出来，得到“站立姿态”特征字典，用作后续归一化或初始参考。
        self._stand_features = tree.map_structure(lambda x: x[0], clip_reference_features)

    def initialize_pose(self, physics, walker, random_state):
        """
        忽略随机因子，把 walker 瞬间拉回到站立帧，再让物理引擎刷新一遍骨骼数据，保证初始状态干净。
        """
        del random_state
        # 把 _stand_features（第 0 帧的关节位置、根位姿等）直接写进 physics.data.qpos，一步到位重置全身姿态。
        utils.set_walker_from_features(physics, walker, self._stand_features)

        # 强制 MuJoCo 重新计算一次正运动学，确保关节位置、末端坐标等派生量与刚才写入的 qpos 完全同步，防止后续观测出现“旧缓存”偏差。
        mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)


class DmControlWrapper(core.Env):
    """
    将dm_control环境和任务包装到Gym环境中。任务假设CMU(卡内基梅隆大学图形实验室动作捕捉数据库)位置控制的人形机器人的存在。

    Adapted from:
    https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py
    """

    # 与 "human" 模式不同，"rgb_array" 不涉及任何图形界面，纯内存数组，适合远程服务器、Docker、Colab 等无显示场景。帧数：30
    metadata = {"render.modes": ["rgb_array"], "videos.frames_per_second": 30}

    def __init__(
        self,
        task_type: Callable[..., composer.Task],
        task_kwargs: Optional[Dict[str, Any]] = None,  # Dict[str, Any]：键为字符串类型，值随便
        environment_kwargs: Optional[Dict[str, Any]] = None,
        act_noise: float = 0.,

        # 渲染
        width: int = 640,
        height: int = 480,
        camera_id: int = 3,

        # 测试的最大帧数
        max_eval_steps: int = 100,
    ):
        """
        task_kwargs: 传递给任务构造函数
        environment_kwargs: 传递给 composer.Environment 构造函数
        """
        task_kwargs = task_kwargs or dict()  # task_kwargs即使为空也传入空字典
        environment_kwargs = environment_kwargs or dict()

        # 创建环境
        self._env = self._create_env(
            task_type,
            task_kwargs,
            environment_kwargs,
            act_noise=act_noise,
        )
        self._original_rng_state = self._env.random_state.get_state()

        # 设置观察和操作空间
        self._observation_space = self._create_observation_space()
        action_spec = self._env.action_spec()
        dtype = np.float32
        self._action_space = spaces.Box(
            low=action_spec.minimum.astype(dtype),
            high=action_spec.maximum.astype(dtype),
            shape=action_spec.shape,
            dtype=dtype
        )

        # set seed
        self.seed()

        self._height = height
        self._width = width
        self._camera_id = camera_id

        self._success_so_far = True

        # --- 新增 ---
        self._max_eval_steps = max_eval_steps
        self._step_in_clip = 0
        self._err_hist = []  # 每帧误差
        self._comic_hist = []  # 每帧 CoMic
        self._success_so_far = True

        # 参考整段数据，reset() 里填充
        self._ref_jpos = None  # (T, J)
        self._ref_jvel = None  # (T, J)
        self._ref_ee = None  # (T, E, 3)
        self._ref_hgt = None  # (T,)
        self._ref_root = None  # (T, 3)

        self._stand_jpos = None  # 第一帧关节位置
        self._stand_jvel = None
        self._stand_ee = None
        self._stand_hgt = None
        self._stand_root = None

        self._ref_features = None

    @staticmethod
    def make_env_constructor(task_type: Callable[..., composer.Task]):
        """
        写 Callable(可调用的) 而不是直接写 Type，个别任务为了兼容旧接口，可能是工厂函数（带默认参数的偏函数、lambda 等）
        ... 表示参数表任意（*args, **kwargs 都行），但返回值必须是 composer.Task 或其子类实例
        """
        return lambda *args, **kwargs: DmControlWrapper(task_type, *args, **kwargs)  # lambda 参数:操作(参数)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def dm_env(self) -> composer.Environment:
        return self._env

    @property
    def observation_space(self) -> spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def np_random(self):
        return self._env.random_state

    def seed(self, seed: Optional[int] = None):
        if seed:
            srng = np.random.RandomState(seed=seed)
            self._env.random_state.set_state(srng.get_state())
        else:
            self._env.random_state.set_state(self._original_rng_state)
        return self._env.random_state.get_state()[1]

    def _create_env(
        self,
        task_type,
        task_kwargs,
        environment_kwargs,
        act_noise=0.,
    ) -> composer.Environment:
        try:  # if和try的区别：能预判用 if，不可控用 try；异常别乱捕，逻辑要清晰。
            # 先按 enable_rgb 参数拿带相机/不带相机的 walker；万一派生类没实现带参版本，就退回到无参默认构造。
            walker = self._get_walker(
                enable_rgb=task_kwargs.get('enable_rgb', False))
        except:
            walker = self._get_walker()

        # 按 arena_type/arena_size 取出对应场地（floor、corridor、gaps 等）。
        arena = self._get_arena(
            arena_type=task_kwargs.get('arena_type', 'floor'),
            arena_size=task_kwargs.get('arena_size', 12.))

        # 防止这些场地/视觉参数被继续透传到 task_type，避免构造函数报 unexpected keyword。
        for key in ['enable_rgb', 'arena_type', 'arena_size']:
            if key in task_kwargs:
                del task_kwargs[key]
        task = task_type(
            walker,
            arena,
            **task_kwargs
        )
        env = composer.Environment(
            task=task,
            **environment_kwargs
        )
        task.random = env.random_state  # for action noise
        if act_noise > 0.:
            env = action_noise.Wrapper(env, scale=act_noise / 2)
        return env

    def _get_walker(self, enable_rgb):
        directory = osp.dirname(osp.abspath(__file__))  # 获取绝对路径
        initializer = StandInitializer()  #获取站立姿势
        return cmu_humanoid.CMUHumanoidPositionControlledV2020(
            initializer=initializer,
            observable_options={'egocentric_camera': dict(enabled=enable_rgb)})

    def _get_arena(self, arena_type, arena_size):
        if arena_type == 'floor':
            return Floor((arena_size, arena_size,))
        elif arena_type == 'corridor':
            return EmptyCorridor(
                corridor_width=5,
                corridor_length=40,
                visible_side_planes=True,
            )
        elif arena_type == 'gaps-corridor':
            return GapsCorridor(
                platform_length=distributions.Uniform(2.0, 3.0),
                gap_length=distributions.Uniform(0.1, 0.4),
                corridor_width=5,
                corridor_length=40,
                visible_side_planes=True,
            )
        elif arena_type == 'walls-corridor':
            return WallsCorridor(
                wall_gap=distributions.Uniform(4.0, 5.5),
                wall_width=distributions.Uniform(1.5, 2.5),
                wall_height=2.,
                wall_rgba=(.7, .3, .5, 1.),
                corridor_width=5,
                corridor_length=40,
                visible_side_planes=True,
            )
        elif arena_type == 'stairs-corridor':
            return StairsCorridor(
                stair_length=distributions.Uniform(1.0, 1.5),
                stair_height=distributions.Uniform(.08, .12),
                corridor_width=5,
                corridor_length=40,
                visible_side_planes=True,
            )
        elif arena_type == 'hurdles-corridor':
            return HurdlesCorridor(
                hurdle_length=0.06,
                hurdle_height=distributions.Uniform(0.15, 0.25),
                hurdle_spacing=distributions.Uniform(3.0, 4.0),
                corridor_width=5,
                corridor_length=40,
                visible_side_planes=True,
            )
        else:
            raise ValueError(f"Unknown arena type: {arena_type}")

    def _create_observation_space(self) -> spaces.Dict:
        obs_spaces = dict()

        for k, v in self._env.observation_spec().items():
            if v.dtype == np.float64 and np.prod(v.shape) > 0:
                if np.prod(v.shape) > 0:
                    if v.shape == ():
                        obs_spaces[k] = spaces.Box(
                            -np.infty,
                            np.infty,
                            shape=(1,),
                            dtype=np.float32)
                        continue
                    obs_spaces[k] = spaces.Box(
                        -np.infty,
                        np.infty,
                        shape=(np.prod(v.shape),),
                        dtype=np.float32
                    )
            elif v.dtype == np.uint8:
                tmp = v.generate_value()
                obs_spaces[k] = spaces.Box(
                    v.minimum.item(),
                    v.maximum.item(),
                    shape=tmp.shape,
                    dtype=np.uint8
                )
        return spaces.Dict(obs_spaces)

    def get_observation(self, time_step: TimeStep) -> Dict[str, np.ndarray]:
        dm_obs = time_step.observation
        obs = dict()
        for k in self.observation_space.spaces:
            if self.observation_space[k].dtype == np.uint8:  # image
                obs[k] = dm_obs[k].squeeze()
            else:
                obs[k] = dm_obs[k].ravel().astype(self.observation_space[k].dtype)
        return obs

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        assert self.action_space.contains(action)

        time_step = self._env.step(action)
        reward = time_step.reward or 0.
        done = time_step.last()
        obs = self.get_observation(time_step)

        # 2. 取出当前帧关节位置（不用 t 当下标）
        curr_pos = time_step.observation['walker/joints_pos']  # shape (J,)
        curr_vel = time_step.observation['walker/joints_vel']  # 如果想算速度误差同理
        curr_ee = time_step.observation['walker/end_effectors_pos']  # shape (E,3)
        height = float(time_step.observation['walker/body_height'])
        # print(f'height:{height:03f}')

        # 3. 计算“当前帧 vs 参考帧”误差
        t = self._step_in_clip
        joint_pos_err = 0.
        joint_vel_err = 0.
        ee_pos_err = 0.

        # if self._ref_jpos is not None and t < len(self._ref_jpos):
        #     ref_pos = self._ref_jpos[t]  # 当前参考关节位置
        #     ref_vel = self._ref_jvel[t]  # 当前参考关节速度
        #     ref_ee = self._ref_ee[t]  # 当前参考末端位置
        #
        #     joint_pos_err = np.linalg.norm(curr_pos - ref_pos) / np.sqrt(curr_pos.size)
        #     joint_vel_err = np.linalg.norm(curr_vel - ref_vel) / np.sqrt(curr_vel.size)
        #     ee_pos_err = np.linalg.norm(curr_ee - ref_ee) / np.sqrt(curr_ee.size)

        stand_jpos = self._stand_jpos
        ref_jpos = self._ref_jpos
        joint_pos_err = np.linalg.norm(curr_pos - stand_jpos) / np.sqrt(curr_pos.size)

        info = dict(
            internal_state=self._env.physics.get_state().copy(),
            discount=time_step.discount,
            # tracking_error=tracking_error,
            joint_pos_err=joint_pos_err,
            joint_vel_err=joint_vel_err,
            ee_pos_err=ee_pos_err,
            height=height
        )

        return obs, reward, done, info

    def reset(self) -> Dict[str, np.ndarray]:
        time_step = self._env.reset()
        self._step_in_clip = 0
        # self._err_hist.clear()
        # self._comic_hist.clear()
        task = self._env.task
        print(task)
        if isinstance(task, tracking.MultiClipMocapTracking):
            ref_features = task._current_reference_features
            self._ref_jpos = ref_features['joints_pos']  # (T, J)
            self._ref_jvel = ref_features['joints_vel']  # (T, J)
            self._ref_ee = ref_features['end_effectors_pos']  # (T, E, 3)  E=4（头、左右手、双脚）
            self._ref_root = ref_features['position']  # (T, 3)   根节点世界坐标
            self._ref_hgt = ref_features.get('body_height', np.full(len(ref_features['position']), 1.4))
            print(f"[DEBUG] Loaded ref_jpos shape: {self._ref_jpos.shape}, len: {self._ref_len}")

            # 轨迹长度记下来，后面防越界
            self._ref_len = self._ref_jpos.shape[0]
        else:
            self._ref_jpos = None
            self._ref_jvel = None
            self._ref_ee = None
            self._ref_root = None
            self._ref_hgt = None
            self._ref_len = 0

        self._stand_jpos = time_step.observation['walker/joints_pos'].copy()
        self._stand_jvel = time_step.observation['walker/joints_vel'].copy()
        self._stand_ee = time_step.observation['walker/end_effectors_pos'].copy()
        self._stand_hgt = time_step.observation['walker/body_height'].copy()
        self._stand_root = time_step.observation['walker/position'].copy()

        self._success_so_far = True

        return self.get_observation(time_step)

    def render(
        self,
        mode: Text = 'rgb_array',
        height: Optional[int] = None,
        width: Optional[int] = None,
        camera_id: Optional[int] = None
    ) -> np.ndarray:
        assert mode == 'rgb_array', "This wrapper only supports rgb_array mode, given %s" % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(height=height, width=width, camera_id=camera_id)
