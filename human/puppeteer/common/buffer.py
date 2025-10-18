from copy import deepcopy

import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler

from common.mocap_dataset import MocapBuffer

'''
deepcopy: 用于深拷贝配置，避免修改原始配置。
torch: PyTorch 张量操作。
TensorDict: 用于结构化存储多维张量数据。
ReplayBuffer, LazyTensorStorage, SliceSampler: torchrl 提供的回放缓冲区组件。
MocapBuffer: 自定义的离线动捕数据集加载器。
'''

# 用于 TD-MPC2 训练的回放缓冲区，优先使用 CUDA 内存，否则使用 CPU。
class Buffer():
	"""
	self._device: 默认使用 CUDA。
	self._capacity: 缓冲区容量，考虑 episode 长度。
	self._sampler: 使用 SliceSampler 采样轨迹片段。
	self._batch_size: 实际采样批量大小为 cfg.batch_size * (horizon + 1)
	"""
	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda')	# 默认使用 CUDA
		self._capacity = min(cfg.buffer_size, cfg.steps) + cfg.episode_length
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0

	# 只读属性
	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity
	
	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	# 根据给定的存储方式（如 CUDA）创建一个 ReplayBuffer 实例。
	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=True,
			prefetch=2,
			batch_size=self._batch_size,
		)
	
	# 用第一个 episode 初始化缓冲区，估算所需显存并打印信息。
	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		print(f'Using CUDA memory for replay buffer.')
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=torch.device('cuda'))
		)

	# 将张量异步转移到指定设备（默认 CUDA），支持 non_blocking 加速
	def _to_device(self, *args, device=None):
		if device is None:
			device = self._device
		return (arg.to(device, non_blocking=True) \
			if arg is not None else None for arg in args)

	'''
	对采样得到的轨迹进行后处理，提取训练所需字段：
	obs	观测	直接使用
	action	动作	去掉第一个时间步 [1:]
	reward	奖励	去掉第一个时间步并增加维度
	terminated	是否终止	只取最后一个时间步并增加维度
	'''
	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		obs = td['obs']
		action = td['action'][1:]
		reward = td['reward'][1:].unsqueeze(-1)
		terminated = td['terminated'][-1].unsqueeze(-1)
		return self._to_device(obs, action, reward, terminated)

	'''
	添加一个 episode 到缓冲区。
	若 episode 长度不足（≤ horizon+1），则跳过。
	给每个时间步添加 episode ID。
	第一个 episode 用于初始化缓冲区。
	更新 episode 计数器。
	'''
	def add(self, td):
		"""Add an episode to the buffer."""
		if len(td) <= self.cfg.horizon+1:
			print(f'Warning: episode of length {len(td)} is too short and will be ignored.')
			return self._num_eps
		td['episode'] = torch.ones_like(td['reward'], dtype=torch.int64) * self._num_eps
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		return self._num_eps

	'''
	从缓冲区采样一个子轨迹批次，并处理后返回：
	返回格式：obs, action, reward, terminated
	使用 SliceSampler 采样。
	调整维度为 (horizon+1, batch_size)。
	调用 _prepare_batch 提取训练数据。
	'''
	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
		return self._prepare_batch(td)

# 集成离线数据（Mocap）与在线交互数据的混合采样器。
class EnsembleBuffer(Buffer):
	'''
	初始化两个缓冲区：
	self._offline: 使用 MocapBuffer 加载离线动捕数据。
	父类 Buffer: 用于在线交互数据。
	'''
	def __init__(self, cfg):
		_cfg = deepcopy(cfg)
		_cfg.batch_size //= 2
		self._offline = MocapBuffer(_cfg)
		super().__init__(_cfg)
		
	# 从离线和在线缓冲区各采样一半数据，拼接后返回。
	def sample(self):
		"""从两个缓冲中 采样一个序列的批次"""
		obs0, action0, reward0, terminated0 = self._offline.sample()  # 从 MocapBuffer 离线数据中采样
		obs1, action1, reward1, terminated1 = super().sample()
		return torch.cat([obs0, obs1], dim=1), \
			torch.cat([action0, action1], dim=1), \
			torch.cat([reward0, reward1], dim=1), \
			torch.cat([terminated0, terminated1], dim=0)


