# %% [markdown]
# # 果蝇强化学习环境的示例
# * 模仿飞行
# * 模拟步行
# * 视觉引导的飞行
# 
# 有关更多详细信息，请参见 `flybody` 的[出版物](https://www.biorxiv.org/content/10.1101/2024.03.11.584515v1) 。

# %% [markdown]
# #### 要运行此脚本，请按照以下步骤操作：
# 1. 在 https://doi.org/10.25378/janelia.25309105 下载参考飞行和步行数据以及训练有素的策略
# 2. 解压下载的文件
# 3. 在下面的代码中指定数据和策略文件的路径

# %%
import os

cur_dir = os.path.split(os.path.realpath(__file__))[0]

# wingbeat 模式发生器（pattern generator）的基线模式。
wpg_pattern_path = os.path.join(cur_dir, 'data/wing_pattern_fmech.npy')

# 飞行和步行参考数据。
ref_flight_path = os.path.join(cur_dir, 'data/flight-dataset_saccade-evasion_augmented.hdf5')
ref_walking_path = os.path.join(cur_dir, 'data/walking-dataset_female-only_snippets-16252_trk-files-0-9.hdf5')

# 训练有素的策略。
#flight_policy_path = '../data/policy/flight'
flight_policy_path = os.path.split(os.path.realpath(__file__))[0] + '\\data\\policy\\flight'
walk_policy_path = os.path.join(cur_dir, 'data/policy/walking')
vision_bumps_path = os.path.join(cur_dir, 'data/policy/vision-bumps')
vision_trench_path = os.path.join(cur_dir, 'data/policy/vision-trench')


# # 包导入
import numpy as np
import PIL.ImageDraw

import tensorflow as tf
import tensorflow_probability as tfp
from acme import wrappers  # pip install git+https://github.com/deepmind/acme.git

from flybody.fly_envs import (
    flight_imitation,
    walk_imitation,
    vision_guided_flight,
)
from flybody.tasks.task_utils import (
    get_random_policy,
    real2canonical,
)
# dm-reverb 仅支持 Linux
# from flybody.agents.utils_tf import TestPolicyWrapper
from flybody.utils import (
    display_video,
    rollout_and_render,
)

# print(tf.__version__)
# print(tfp.__version__)

# 直接运行可以，但是在Interactive环境中，运行报错：
# ValueError: The type 'tensorflow_probability.python.distributions.independent.Independent_ACTTypeSpec' has not been registered.  It must be registered before you load this object (typically by importing its module).
# flight_policy = tf.saved_model.load(flight_policy_path)
# print("Flight policy loaded successfully.")


# %%
# 防止 TensorFlow 窃取所有 GPU 内存。
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# %% [markdown]
# # 有用的渲染功能
def blow(x, repeats=2):
    """Repeat columns and rows requested number of times."""
    return np.repeat(np.repeat(x, repeats, axis=0), repeats, axis=1)


def vision_rollout_and_render(env, policy, camera_id=1,
                              eye_blow_factor=5, **render_kwargs):
    """Run vision-guided flight episode and render frames, including eyes."""
    from acme.tf import utils as tf2_utils
    import numpy
    frames = []
    timestep = env.reset()
    # Run full episode until it ends.
    while timestep.step_type != 2:
        # Render eyes and scene.
        pixels = env.physics.render(camera_id=camera_id, **render_kwargs)
        eyes = eye_pixels_from_observation(
            timestep, blow_factor=eye_blow_factor)
        # Add eye pixels to scene.
        pixels[0:eyes.shape[0], 0:eyes.shape[1], :] = eyes
        frames.append(pixels)
        # Step environment.
        # 没有使用分布式策略，在这里补充TestPolicyWrapper里的一些操作
        batched_observation = tf2_utils.add_batch_dim(timestep.observation)
        distribution = policy(batched_observation)
        action = distribution.mean()  # 不是测试模式（测试模式返回均值和方差）
        if not (type(action) is numpy.float64):
            action = action[0, :].numpy()  # Remove batch dimension.
        timestep = env.step(action)
    return frames


def eye_pixels_from_observation(timestep, blow_factor=4):
    """Get current eye view from timestep.observation."""
    # In the actual task, the averaging over axis=-1 is done by the visual
    # network as a pre-processing step, so effectively the visual observations
    # are gray-scale.
    left_eye = timestep.observation['walker/left_eye'].mean(axis=-1)
    right_eye = timestep.observation['walker/right_eye'].mean(axis=-1)
    pixels = np.concatenate((left_eye, right_eye), axis=1)
    pixels = np.tile(pixels[:, :, None], reps=(1, 1, 3))
    pixels = blow(pixels, blow_factor)
    half_size = pixels.shape[1] // 2
    # Add white line to separate eyes.
    pixels = np.concatenate((pixels[:, :half_size, :], 
                            255*np.ones((blow_factor*32, 2, 3)),
                            pixels[:, half_size:, :]), axis=1)
    pixels = pixels.astype('uint8')
    return pixels


def eye_pixels_from_cameras(physics, **render_kwargs):
    """Render two-eye view, assuming eye cameras have particular names."""
    for i in range(physics.model.ncam):
        name = physics.model.id2name(i, 'camera')
        if 'eye_left' in name:
            left_eye = physics.render(camera_id=i, **render_kwargs)
        if 'eye_right' in name:
            right_eye = physics.render(camera_id=i, **render_kwargs)
    pixels = np.hstack((left_eye, right_eye))
    return pixels


# 用于渲染的帧宽度和高度。
render_kwargs = {'width': 640, 'height': 480}





# %% [markdown]
# # 1. 飞行模仿环境
# 
# 在此任务中，需要果蝇模型来通过匹配其质量中心位置和身体方向来跟踪真实参考果蝇的飞行轨迹。翅膀运动由策略网络和翅膀模式发生器的组合控制。对单个策略进行了训练，可以跟踪飞行数据集中的所有轨迹。

# 让我们创建飞行模仿环境并可视化初始状态。
env = flight_imitation(ref_flight_path,
                       wpg_pattern_path,
                       terminal_com_dist=float('inf'))
env = wrappers.SinglePrecisionWrapper(env)
env = wrappers.CanonicalSpecWrapper(env, clip=True)

_ = env.reset()
pixels = env.physics.render(camera_id=1, **render_kwargs)
PIL.Image.fromarray(pixels)


# %% [markdown]
# 创建虚拟随机动作策略，运行轮次并制作视频。灰色“幽灵”果蝇指示参考飞行位置，模型需要跟踪。

random_policy = get_random_policy(env.action_spec())

frames = rollout_and_render(env, random_policy, run_until_termination=True,
                            camera_ids=1, **render_kwargs)
display_video(frames)


# %% [markdown]
# 加载训练有素的策略，运行轮次并制作视频。
# import tensorflow_probability
# from tensorflow_probability.python.distributions.independent import *
flight_policy = tf.saved_model.load(flight_policy_path)

# 包装策略在测试时与非批次观察一起工作。
# flight_policy = TestPolicyWrapper(flight_policy)

frames = rollout_and_render(env, flight_policy, run_until_termination=True,
                            camera_ids=1, **render_kwargs)
display_video(frames)




# %% [markdown]
# # 2. 步行模仿环境
# 
# 在此任务中，需要果蝇模型来通过匹配（i）质量中心位置，（ii）身体方向以及（iii）详细的腿部运动来跟踪真实参考果蝇的步行轨迹。对单个策略进行了训练，可以跟踪步行数据集中的所有轨迹。

# 让我们创建步行环境并可视化初始状态。

env = walk_imitation(ref_path=ref_walking_path,
                     terminal_com_dist=float('inf'))
env = wrappers.SinglePrecisionWrapper(env)
env = wrappers.CanonicalSpecWrapper(env, clip=True)

_ = env.reset()
pixels = env.physics.render(camera_id=1, **render_kwargs)
PIL.Image.fromarray(pixels)


# %% [markdown]
# 创建虚拟随机行动策略，运行一个轮次并制作视频。灰色“幽灵”果蝇表示参考果蝇位置。在这里，我们将通过移动随机操作使视频变得漂亮来创建一个略有不同的随机效果策略。

_random_policy = get_random_policy(env.action_spec(),
                                   minimum=-.5, maximum=.5)
def random_policy(observation):
    action = _random_policy(observation)
    # 将以零为中心的随机动作转换为规范表示，以匹配我们添加到上面的步行环境中的 CanonicalSpecWrapper。
    action = real2canonical(action, env._environment.action_spec())
    return action

# 从数据集请求特定的（足够长）的步行轨迹。
env.task.set_next_trajectory_index(idx=316)

frames = rollout_and_render(env, random_policy, run_until_termination=True,
                            camera_ids=2, **render_kwargs)
display_video(frames)



# %% [markdown]
# 让我们加载训练有素的策略，运行一个轮次并制作视频。

walking_policy = tf.saved_model.load(walk_policy_path)
# walking_policy = TestPolicyWrapper(walking_policy)

# 从数据集请求特定的（足够长）的步行轨迹。
env.task.set_next_trajectory_index(idx=316)

frames = rollout_and_render(env, walking_policy, run_until_termination=True,
                            camera_ids=2, **render_kwargs)
display_video(frames)




# %% [markdown]
# # 3. 在不规则地形上（“颠簸”）的视觉引导飞行
# 
# 在此任务中，需要果蝇模型在保持不规则的正弦状地形上，同时关于当前的地形海拔保持恒定的 z 偏移（高度） 。该模型无法直接访问其飞行高度。取而代之的是，它必须学会使用视觉来估计当前高度并调整它以匹配当前的地形高程。
# 与地形碰撞终止了这一轮次。每个轮次都会随机重新生成地形形状。

# 让我们创建“颠簸”视觉任务环境并可视化初始状态。
env = vision_guided_flight(wpg_pattern_path, bumps_or_trench='bumps')
env = wrappers.SinglePrecisionWrapper(env)
env = wrappers.CanonicalSpecWrapper(env, clip=True)

timestep = env.reset()
pixels = env.physics.render(camera_id=1, **render_kwargs)
PIL.Image.fromarray(pixels)


# %% [markdown]
# 让我们从初始轮次状态下从眼睛摄像机中呈现高分辨率的视图

pixels = eye_pixels_from_cameras(env.physics, **render_kwargs)
PIL.Image.fromarray(pixels)


# %% [markdown]
# 在同一初始轮次状态下，通过其可观察到的实际低分辨率视图看起来更像是这样：

pixels = eye_pixels_from_observation(timestep, blow_factor=10)
PIL.Image.fromarray(pixels)


# %% [markdown]
# 创建虚拟随机动作策略，运行轮次并制作视频。
random_policy = get_random_policy(env.action_spec())

frames = vision_rollout_and_render(env, random_policy, **render_kwargs)
display_video(frames)


# %% [markdown]
# 加载训练有素的策略，运行一个轮次并制作视频：
bumps_policy = tf.saved_model.load(vision_bumps_path)
# bumps_policy = TestPolicyWrapper(bumps_policy)

frames = vision_rollout_and_render(env, bumps_policy, **render_kwargs)
display_video(frames)



# %% [markdown]
# # 4. 通过沟槽的视觉引导飞行
# 
# 在此任务中，需要果蝇通过锯齿状的沟槽，而不会与沟槽墙相撞。触摸地形或沟槽墙壁终止了轮次。果蝇必须学会利用视觉来估计其在沟槽中的位置，并进行操纵以远离沟沟槽。在每个轮次中都会随机重新生成沟槽的形状。

# 让我们创建沟槽"trench"任务环境并可视化初始状态
env = vision_guided_flight(wpg_pattern_path, bumps_or_trench='trench')
env = wrappers.SinglePrecisionWrapper(env)
env = wrappers.CanonicalSpecWrapper(env, clip=True)

_ = env.reset()
pixels = env.physics.render(camera_id=6, **render_kwargs)
PIL.Image.fromarray(pixels)


# %% [markdown]
# 让我们添加一个新的跟踪摄像头，以实现更好的沟通任务可视化：

# 找到胸部并在其中添加跟踪摄像头。
thorax = env.task.root_entity.mjcf_model.find('body', 'walker/thorax')
_ = thorax.add('camera', name='rear', mode='trackcom',
               pos=(-1.566, 0.037, -0.021),
               xyaxes=(-0.014, -1, 0, -0.012, 0, 1))

# 用新相机可视化初始状态：
timestep = env.reset()
trench_camera_id = env.physics.model.name2id('walker/rear', 'camera')
pixels = env.physics.render(camera_id=trench_camera_id, **render_kwargs)
PIL.Image.fromarray(pixels)


# %% [markdown]
# 让我们从初始轮次中从眼摄像机中呈现高分辨率的视角：

pixels = eye_pixels_from_cameras(env.physics, **render_kwargs)
PIL.Image.fromarray(pixels)


# %% [markdown]
# 以及果蝇模型使用的相应低分辨率视图：
pixels = eye_pixels_from_observation(timestep, blow_factor=10)
PIL.Image.fromarray(pixels)

# %% [markdown]
# 和以前一样，让我们以随机动作策略进行一个轮次：

random_policy = get_random_policy(env.action_spec())

frames = vision_rollout_and_render(
    env, random_policy, camera_id=trench_camera_id, **render_kwargs)
display_video(frames)


# %% [markdown]
# 让我们加载训练有素的策略并进行一个轮次：

trench_policy = tf.saved_model.load(vision_trench_path)
# trench_policy = TestPolicyWrapper(trench_policy)

frames = vision_rollout_and_render(
    env, trench_policy, camera_id=trench_camera_id, **render_kwargs)
display_video(frames)



