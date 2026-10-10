# -*- coding: utf-8 -*-
"""6 格学习曲线 —— 论文 Fig.2(b) 同版式，**只有我们自己的数据**（无任何对比元素）

与 export_6grid_3seed.py 的区别（那个是"带全部元数据的工作版图"）：
  1. 面板标题照论文写法：SMPL Humanoid-Jump / Unitree H1-Run / Unitree H1-Balance
     / MyoHand-PenTwirl / MyoLeg-Walk / Ostrich-Run（去掉 |S|/|A|/env_id/budget 等原信息）；
  2. **图例提到顶部、横排共用**（论文版式），不再每格塞图例；
  3. 轴名照论文：x = Steps、y = Episode Return；x 轴用 1e6 / 1e7 科学计数偏移（与论文一致）；
  4. 去掉网格、去掉 MyoLeg 面板的论文参考虚线、去掉段落式脚注（只留一行数据口径）；
  5. 配色改用**论文同一套算法色**（Qflex 红 / CrossQ 蓝 / SAC 橙 / SDAC 深棕 / DACER 品红 / QSM 灰）；
  6. 出两张：mean+band（论文样式）与 mean+band+每 seed 细线。

数据源 / 读取逻辑复用 export_6grid_3seed.py（同一份 log.csv、同一套 .formal_done 判据）。
运行：
    QFLEX_LOGS=~/work/Qflex/logs QFLEX_OUT=./figs python plot/export_6grid_paperstyle.py
    # 需要 matplotlib + numpy；路径同样用 QFLEX_LOGS / QFLEX_OUT 覆盖（读取逻辑复用 export_6grid_3seed.py）
"""
import os
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import export_6grid_3seed as E  # noqa: E402

OUT = E.OUT

# 论文 Fig.2(b) 的面板标题
PAPER_TITLE = {
    'SMPLHumanoidJump-v1': 'SMPL Humanoid-Jump',
    'h1-run-v0': 'Unitree H1-Run',
    'h1-balance_simple-v0': 'Unitree H1-Balance',
    'myoHandPenTwirlRandom-v0': 'MyoHand-PenTwirl',
    'myoLegWalk-v0': 'MyoLeg-Walk',
    'OstrichRun-v1': 'Ostrich-Run',
}

# 论文同一套算法配色 + 图例顺序（论文：Qflex, CrossQ, SAC, SDAC, DACER, QSM, ...）
ORDER = ['qflex', 'crossq', 'sac', 'sdac', 'dacer', 'qsm']
PCOLOR = {
    'qflex': '#d62728',   # red
    'crossq': '#1f77b4',  # blue
    'sac': '#ff7f0e',     # orange
    'sdac': '#8c564b',    # dark brown
    'dacer': '#e377c2',   # magenta
    'qsm': '#7f7f7f',     # gray
}

FOOT = ('Only this reproduction\'s data (no comparison drawn).  Lines = mean over 3 seeds (100/101/102), '
        'bands = +/-1 std across seeds; metric = log.csv avg_ret, i.e. the column read by the official '
        'scripts/inspect_results.py.  The paper\'s bands come from 5 independent runs, so band widths are '
        'not comparable with ours.')


def draw(ax, meta, d, show_seeds):
    name, env_id, ns, na, nenv, budget, w = meta
    mean_lo, band_hi = [], []
    for alg in ORDER:
        if alg not in d:
            continue
        seeds, x, Y = E.stack(d[alg])
        m, sd = E.mean_sd(Y)
        if show_seeds and Y.shape[0] > 1:
            for k in range(Y.shape[0]):
                ax.plot(x, Y[k], '-', color=PCOLOR[alg], lw=0.55, alpha=0.26, zorder=2)
        if sd.max() > 1e-9:
            ax.fill_between(x, m - sd, m + sd, color=PCOLOR[alg], alpha=0.16, lw=0, zorder=3)
        ax.plot(x, m, '-', color=PCOLOR[alg], lw=1.8, zorder=5)
        mean_lo.append(float(m.min()))
        band_hi.append(float((m + sd).max()))

    ax.axhline(0, color='0.78', lw=0.7, ls='--', zorder=1)
    ax.set_xlim(0, budget * 1.02)
    # 论文版式：y 轴从 0 起（或略低），误差带越界的部分直接裁掉，不把坐标轴拽到负区
    lo = min(0.0, min(mean_lo))
    hi = max(band_hi)
    span = max(hi - lo, 1.0)
    ax.set_ylim(lo - 0.04 * span, hi + 0.12 * span)

    fmt = ScalarFormatter(useMathText=False)
    fmt.set_powerlimits((6, 7))
    ax.xaxis.set_major_formatter(fmt)

    ax.set_title(PAPER_TITLE[env_id], fontsize=11.5, pad=6)
    ax.set_xlabel('Steps', fontsize=10.5)
    ax.set_ylabel('Episode Return', fontsize=10.5)
    ax.tick_params(labelsize=9)
    ax.grid(False)


def build(data, fname, show_seeds):
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.1), dpi=200)
    axes = axes.ravel()
    for i, meta in enumerate(E.ENVS):
        draw(axes[i], meta, data[meta[1]], show_seeds)

    handles = [Line2D([0], [0], color=PCOLOR[a], lw=2.2, label=a) for a in ORDER]
    fig.legend(handles=handles, ncol=6, loc='upper center', bbox_to_anchor=(0.5, 0.982),
               frameon=False, fontsize=11.5, handlelength=2.4, columnspacing=2.4)

    foot = '\n'.join(textwrap.wrap(FOOT, 168))
    fig.text(0.5, 0.012, foot, ha='center', va='bottom', fontsize=8.0, color='0.34', linespacing=1.6)
    fig.subplots_adjust(left=0.052, right=0.986, top=0.885, bottom=0.135, hspace=0.44, wspace=0.24)
    p = os.path.join(OUT, fname)
    fig.savefig(p)
    plt.close(fig)
    print('saved:', p)


def main():
    data, notes = E.load_all()
    if notes:
        print('!! incomplete:', notes)
    build(data, 'fig_6grid_paperstyle.png', show_seeds=False)
    build(data, 'fig_6grid_paperstyle_seeds.png', show_seeds=True)


if __name__ == '__main__':
    main()
