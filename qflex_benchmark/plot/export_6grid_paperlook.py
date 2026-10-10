# -*- coding: utf-8 -*-
"""6 格学习曲线 —— "论文画法"版（仍然只有我们自己的数据）。

与 fig_6grid_paperstyle.png 的唯一区别 = 两项**画法**：
  1. 主线加粗 lw 1.8 -> 3.4（论文红线实测占面板高 1.6~4.9%，我们原版只有 0.8%）；
  2. y 轴上限不再迁就 ±std 带的顶，改成按"均值曲线"的最高点定
     (原来 hi = max(m+sd)，把整幅图纵向压低 1.2~1.9 倍；论文的带同样会顶出面板、
      超出部分直接裁掉)。
数据、种数、口径、图例/标题/脚注全部与 paperstyle 版一致。
运行：
    QFLEX_LOGS=~/work/Qflex/logs QFLEX_OUT=./figs python plot/export_6grid_paperlook.py
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
import export_6grid_3seed as E            # noqa: E402
import export_6grid_paperstyle as PS      # noqa: E402

OUT = E.OUT
ORDER = PS.ORDER
PCOLOR = PS.PCOLOR
PAPER_TITLE = PS.PAPER_TITLE
FOOT = ('Only this reproduction\'s data.  Lines = mean over 3 seeds (100/101/102), bands = +/-1 std across seeds, '
        'CLIPPED at the axis limit exactly as in the paper (paper bands also run off the panel).  '
        'Metric = log.csv avg_ret (the column read by the official scripts/inspect_results.py).  '
        'Paper bands come from 5 independent runs, so band widths are not comparable with ours.')


def draw(ax, meta, d):
    name, env_id, ns, na, nenv, budget, w = meta
    mean_hi = []
    for alg in ORDER:
        if alg not in d:
            continue
        seeds, x, Y = E.stack(d[alg])
        m, sd = E.mean_sd(Y)
        if sd.max() > 1e-9:
            ax.fill_between(x, m - sd, m + sd, color=PCOLOR[alg], alpha=0.18, lw=0, zorder=3)
        ax.plot(x, m, '-', color=PCOLOR[alg], lw=3.4, solid_capstyle='round', zorder=5)
        mean_hi.append(float(m.max()))

    ax.axhline(0, color='0.78', lw=0.7, ls='--', zorder=1)
    ax.set_xlim(0, budget * 1.02)
    # ★ 关键：y 轴只看均值曲线的最高点，不迁就误差带（带溢出即裁掉）
    lo = min(0.0, min(mean_hi))
    hi = max(mean_hi)
    span = max(hi - lo, 1.0)
    ax.set_ylim(lo - 0.03 * span, hi + 0.12 * span)

    fmt = ScalarFormatter(useMathText=False)
    fmt.set_powerlimits((6, 7))
    ax.xaxis.set_major_formatter(fmt)

    ax.set_title(PAPER_TITLE[env_id], fontsize=11.5, pad=6)
    ax.set_xlabel('Steps', fontsize=10.5)
    ax.set_ylabel('Episode Return', fontsize=10.5)
    ax.tick_params(labelsize=9)
    ax.grid(False)


def build(data, fname):
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.1), dpi=200)
    axes = axes.ravel()
    for i, meta in enumerate(E.ENVS):
        draw(axes[i], meta, data[meta[1]])
    handles = [Line2D([0], [0], color=PCOLOR[a], lw=2.6, label=a) for a in ORDER]
    fig.legend(handles=handles, ncol=6, loc='upper center', bbox_to_anchor=(0.5, 0.982),
               frameon=False, fontsize=11.5, handlelength=2.4, columnspacing=2.4)
    fig.text(0.5, 0.012, '\n'.join(textwrap.wrap(FOOT, 168)), ha='center', va='bottom',
             fontsize=8.0, color='0.34', linespacing=1.6)
    fig.subplots_adjust(left=0.052, right=0.986, top=0.885, bottom=0.135, hspace=0.44, wspace=0.24)
    p = os.path.join(OUT, fname)
    fig.savefig(p)
    plt.close(fig)
    print('saved:', p)


def main():
    data, notes = E.load_all()
    if notes:
        print('!! incomplete:', notes)
    build(data, 'fig_6grid_paperlook.png')


if __name__ == '__main__':
    main()
