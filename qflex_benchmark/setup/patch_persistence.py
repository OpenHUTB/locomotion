#!/usr/bin/env python3
"""幂等修补 Qflex 的 relax/utils/persistence.py，使其兼容 jax 0.4.35+ / 0.4.38。

三处修补（均在本机 jax 0.4.38 上实测定界并验证）：

  FIX-1 named_shape
      0.4.35 起 ShapeDtypeStruct.named_shape 属性被移除（0.4.27 尚存）
      → 改用 getattr(x, "named_shape", {})，语义等价。

  FIX-2 jax 配置 flag（True 真凶）
      jax._src.config.State 实例（79 个配置 flag）基类是 pybind11 的
      xla_extension.config.Config，CPython 无法 pickle：
          TypeError: cannot pickle 'State' object
      它在真实 jaxpr 图里以「绑定方法」形式出现（State.__call__，经
      jax._src.compute_on.extend_compute_type 挂进 JaxprEqnContext），所以
      **不能丢成 None** —— 否则反序列化时 types.MethodType(func, None) 会报
          TypeError: instance must not be None
      正确做法：按名字持久化（"S:<属性名>"），反序列化时取回当前进程里的同名活
      对象；与文件里已有的 jax.core.Primitive 按名恢复（"P:<name>"）同一思路。

  FIX-3 兜底与回读校验
      ① Pickler.save 捕获其它 "cannot pickle ..." 叶子并打印（丢成 None），
         使训练不再在 trainer.setup() 阶段中断；QFLX_PICKLE_STRICT=1 可关闭。
      ② PersistFunction.save 写完后立即回读校验，避免"能存不能读"的静默损坏。

用法： python3 patch_persistence.py <path/to/persistence.py> [--check]
"""
import argparse
import sys
from pathlib import Path

MARK_F1 = '_ns = lambda x: getattr(x, "named_shape", {})'
MARK_F2 = "_JAX_CONFIG_STATE_MODULE"
MARK_F3 = "write_ok = False"

FIX1_OLD = """        assert all(x.named_shape == {} and x.sharding is None for x in in_descr_flat)
        assert all(x.named_shape == {} and x.sharding is None for x in out_descr_flat)"""

FIX1_NEW = """        # FIX-1: jax>=0.4.35 移除了 ShapeDtypeStruct.named_shape（0.4.27 尚存）
        _ns = lambda x: getattr(x, "named_shape", {})  # noqa: E731
        assert all(_ns(x) == {} and x.sharding is None for x in in_descr_flat)
        assert all(_ns(x) == {} and x.sharding is None for x in out_descr_flat)"""

# ---- FIX-3 的 save 回读校验：替换 PersistFunction.save 方法体 ----
SAVE_OLD = """    def save(self, path: os.PathLike):
        pickler = PersistFunctionPickler(open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickler.dump(self)"""

SAVE_NEW = """    def save(self, path: os.PathLike):
        with open(path, "wb") as f:
            pickler = PersistFunctionPickler(f, protocol=pickle.HIGHEST_PROTOCOL)
            pickler.dump(self)
        # FIX-3②: 回读校验 —— 这一份结构稍后要由评估子进程用 Unpickler 读回，
        # 因此"能存"不等于"能用"。这里立刻读一次，失败就当场报明确错误。
        write_ok = False
        try:
            PersistFunction.load(path)
            write_ok = True
        except Exception as exc:  # pragma: no cover
            if os.environ.get("QFLX_PICKLE_STRICT") == "1":
                raise
            print(
                f"[persistence] WARNING 写出的结构无法回读（评估子进程会失败）: {path}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        return write_ok"""

FIX2_OLD = """class PersistFunctionPickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, jax.core.Primitive):
            if obj.name in BLACKLIST:
                raise pickle.PickleError(f"Cannot pickle {obj.name}")
            return f"P:{obj.name}"
        elif isinstance(obj, jaxlib.xla_client.Traceback):
            return "T"
        elif isinstance(obj, types.FunctionType) and "<locals>" in obj.__qualname__:
            return "C"
        return None"""

FIX2_BLOCK = '''# FIX-2: jax 0.4.38 的配置 flag（jax._src.config.State 实例，基类是 pybind11 的
# xla_extension.config.Config，tp_new 缺失）无法 pickle：
#     TypeError: cannot pickle 'State' object
# 它在被保存的 jaxpr 里以「绑定方法」出现（State.__call__ <- JaxprEqnContext /
# jax._src.compute_on.extend_compute_type）。因此不能丢成 None —— 反序列化时
# types.MethodType(func, None) 会抛 "instance must not be None"。
# 这里采用与 jax.core.Primitive 完全一致的做法：按名字持久化，读回时取当前进程
# 的同名活对象（配置 flag 本就是进程级单例）。
_JAX_CONFIG_STATE_TYPES: tuple = ()
_JAX_CONFIG_STATE_MODULE = None
try:  # pragma: no cover
    from jax._src import config as _jax_config_state_mod  # type: ignore
    if hasattr(_jax_config_state_mod, "State"):
        _JAX_CONFIG_STATE_TYPES = (_jax_config_state_mod.State,)
        _JAX_CONFIG_STATE_MODULE = _jax_config_state_mod
except Exception:
    pass


def _jax_config_state_key(obj) -> str:
    """给出 flag 对象在 jax._src.config 里的属性名（用于按名持久化）。"""
    if _JAX_CONFIG_STATE_MODULE is not None:
        for k, v in list(vars(_JAX_CONFIG_STATE_MODULE).items()):
            if v is obj:
                return k
    for attr in ("__name__", "_name"):
        name = getattr(obj, attr, None)
        if isinstance(name, str) and name:
            return name
    return ""


class PersistFunctionPickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, jax.core.Primitive):
            if obj.name in BLACKLIST:
                raise pickle.PickleError(f"Cannot pickle {obj.name}")
            return f"P:{obj.name}"
        elif isinstance(obj, jaxlib.xla_client.Traceback):
            return "T"
        elif _JAX_CONFIG_STATE_TYPES and isinstance(obj, _JAX_CONFIG_STATE_TYPES):
            key = _jax_config_state_key(obj)
            if key:
                return f"S:{key}"
            return "C"
        elif isinstance(obj, types.FunctionType) and "<locals>" in obj.__qualname__:
            return "C"
        return None

    # FIX-3①: 兜底 —— 其它未知的 "cannot pickle ..." 叶子丢成 None 并打印，
    # 避免整次训练在 setup 阶段被打断。QFLX_PICKLE_STRICT=1 可关掉。
    def save(self, obj, save_persistent_id=True):
        try:
            return super().save(obj, save_persistent_id)
        except TypeError as exc:
            if not str(exc).startswith("cannot pickle"):
                raise
            if os.environ.get("QFLX_PICKLE_STRICT") == "1":
                raise
            t = type(obj)
            print(
                "[persistence] dropped unpicklable object "
                f"{getattr(t, '__module__', '?')}.{getattr(t, '__qualname__', t)}: {exc}",
                flush=True,
            )
            return super().save(None, save_persistent_id)'''

UNPICK_OLD = """    def persistent_load(self, pid):
        if not isinstance(pid, str):
            raise pickle.UnpicklingError(f"Invalid persistent id {pid}")
        if pid == "T" or pid == "C":
            return None
        elif pid.startswith("P:"):"""

UNPICK_NEW = """    def persistent_load(self, pid):
        if not isinstance(pid, str):
            raise pickle.UnpicklingError(f"Invalid persistent id {pid}")
        if pid == "T" or pid == "C":
            return None
        elif pid.startswith("S:"):
            # FIX-2: 取回当前进程里同名的 jax 配置 flag（进程级单例）
            key = pid[2:]
            if _JAX_CONFIG_STATE_MODULE is not None:
                flag = getattr(_JAX_CONFIG_STATE_MODULE, key, None)
                if flag is not None:
                    return flag
            raise pickle.UnpicklingError(f"Unknown jax config flag: {key}")
        elif pid.startswith("P:"):"""


def patch(path: Path, check_only: bool = False) -> int:
    src = path.read_text(encoding="utf-8")
    orig = src
    notes = []

    def sub(old, new, mark, name):
        nonlocal src
        if mark in src:
            notes.append(f"{name} 已存在")
        elif old not in src:
            notes.append(f"{name} 锚点未找到!!")
        else:
            src = src.replace(old, new, 1)
            notes.append(f"{name} 已应用")

    if check_only:
        for mk, nm in ((MARK_F1, "FIX-1"), (MARK_F2, "FIX-2"), (MARK_F3, "FIX-3")):
            print(f"  {nm}: {'已应用' if mk in src else '缺失'}")
        return 0

    sub(FIX1_OLD, FIX1_NEW, MARK_F1, "FIX-1 named_shape")
    sub(FIX2_OLD, FIX2_BLOCK, MARK_F2, "FIX-2 config State 按名持久化 + FIX-3① 兜底")
    sub(UNPICK_OLD, UNPICK_NEW, "Unknown jax config flag", "FIX-2 unpickler")
    sub(SAVE_OLD, SAVE_NEW, MARK_F3, "FIX-3② 写后回读校验")

    if src != orig:
        backup = Path(str(path) + ".orig")
        if not backup.exists():
            backup.write_text(orig, encoding="utf-8")
            print(f"[OK] 备份原文件 -> {backup}")
        path.write_text(src, encoding="utf-8")

    # 用 compile() 而不写 .pyc，避免在被修补目录里留下 __pycache__
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    final = path.read_text(encoding="utf-8")
    print(f"[OK] 语法校验通过: {path}")
    for n in notes:
        print("   -", n)
    for mk, nm in ((MARK_F1, "FIX-1"), (MARK_F2, "FIX-2"), (MARK_F3, "FIX-3")):
        print(f"  {nm}: {'已应用' if mk in final else '缺失'}")
    missing = [n for n in notes if "未找到" in n]
    return 1 if missing else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    p = Path(a.path)
    if not p.exists():
        print(f"[FAIL] 文件不存在: {p}")
        sys.exit(2)
    sys.exit(patch(p, a.check))
