import re
from pathlib import Path

# pip install hydra-core
# 通过结构化的YAML文件定义配置，并通过命令行进行灵活的覆盖和组合，同时自动化地管理实验输出目录
import hydra
from omegaconf import OmegaConf


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
	"""
	解析 Hydra 配置。大多数是为了方便。
	"""

	# 字符串None转真正的None，其中cfg.keys()代表config.yaml中的关键字
	for k in cfg.keys():
		try:
			v = cfg[k]
			if v in {'None', 'none'}:
				cfg[k] = None
		except:
			pass

	# 代数表达式
	for k in cfg.keys():
		try:
			v = cfg[k]
			if isinstance(v, str):
				match = re.match(r"(\d+)([+\-*/])(\d+)", v)  # 判断字符串 v 是不是“两个整数＋一个四则运算符”的简单算术式
				if match:
					cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))  # eval 主要用于执行字符串表达式，group(1),(2),(3)代表match中(\d+)([+\-*/])(\d+):32 + 24
					if isinstance(cfg[k], float) and cfg[k].is_integer():  # is_integer()是浮点数中的方法(3.0.is_integer()返回True，而3.1.is_integer()返回False)
						cfg[k] = int(cfg[k])
		except:
			pass

	# Convenience
	# 获取当前工作目录（current working directory--cwd），如果运行方式为 python puppeteer/train.py，则 logs 位于当前目录下
	cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name
	cfg.task_title = cfg.task.replace("-", " ").title()  # title()：首字母大写
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins-1)  # Bin size for discrete regression

	# Task-specific
	if cfg.task == 'tracking': # high variance
		cfg.eval_episodes = 100
	cfg.obs = 'rgb' if 'corridor' in cfg.task else 'state'  # 观测类型是图像（'rgb'）还是状态向量（'state'）

	return cfg
