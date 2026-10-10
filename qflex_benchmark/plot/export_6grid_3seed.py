# -*- coding: utf-8 -*-
"""6 格学习曲线 —— 3 seed（100/101/102）均值 ± seed 间 std 带

相对 export_6grid_figs.py（只画 seed 100）的改动：
  1. 自动发现 run 目录，**只认带 `.formal_done` 的**（排除 4 个残目录）⇒ 不再硬编码目录名；
  2. 每格每算法画 **3 个 seed 的均值线 + ±1 std（跨 seed）带**，并把每个 seed 画成细线；
  3. 额外出 CSV：per-seed 原始点 + per-(env,alg) 的 final/best 均值与跨 seed 标准差；
  4. 排名图带 **跨 seed 误差棒**（论文口径 = best checkpoint）。

面板顺序严格照论文 Fig.2(b)：row1 = SMPL-Jump / H1-Run / H1-Balance（humanoid），
row2 = MyoHand-PenTwirl / MyoLeg-Walk / Ostrich-Run（musculoskeletal）。
依据 PDF p7 caption「Baselines in the second row are run only on musculoskeletal benchmarks」。

数据源：每个 run 的 log.csv（evaluator.py:129-133，5 回合的 mean/std）——官方 inspect_results.py 读的同一列。
运行：
    QFLEX_LOGS=~/work/Qflex/logs QFLEX_OUT=./figs python export_6grid_3seed.py
    # 需要 matplotlib + numpy；QFLEX_LOGS 也可换成一个本地目录（里面按 <env_id>/<alg>_<ts>_<sX>_<sX>/log.csv 排布）
"""
import os
import csv
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---- 路径：均可用环境变量覆盖，默认值指向"脚本同级目录"下的常见位置 ----
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('QFLEX_LOGS', os.path.expanduser('~/work/Qflex/logs'))
OUT = os.environ.get('QFLEX_OUT', os.path.abspath(os.path.join(HERE, os.pardir, 'figs')))
PAPER = os.environ.get('QFLEX_PAPER', '')   # 可选：论文像素量测 CSV 所在目录；不提供则跳过对比图
os.makedirs(OUT, exist_ok=True)

ALGS = ['qflex', 'crossq', 'dacer', 'sac', 'qsm', 'sdac']
COLORS = {'qflex': '#d62728', 'crossq': '#1f77b4', 'dacer': '#2ca02c',
          'sac': '#9467bd', 'qsm': '#ff7f0e', 'sdac': '#8c564b'}
ALG_ORDER = {a: i for i, a in enumerate(ALGS)}
SEEDS = ['s100', 's101', 's102']

# (显示名, env_id, |S|, |A|, nenv, 预算步数, workers)
ENVS = [
    ('SMPL-Jump',        'SMPLHumanoidJump-v1',      358,  69, 80,  5_000_000, 20),
    ('H1-Run',           'h1-run-v0',                 51,  19, 70,  5_000_000, 14),
    ('H1-Balance',       'h1-balance_simple-v0',      64,  19, 70, 20_000_000, 14),
    ('MyoHand-PenTwirl', 'myoHandPenTwirlRandom-v0',  83,  39, 80,  5_000_000, 20),
    ('MyoLeg-Walk',      'myoLegWalk-v0',            403,  80, 80, 20_000_000, 20),
    ('Ostrich-Run',      'OstrichRun-v1',            596, 120, 80, 10_000_000,  2),
]

PAPER_MYOLEG_QFlex = 1.3e4
PAPER_OSTRICH_END = 1.274e4

FOOT = (
    "Data: per-run log.csv avg_ret = mean over the 5 evaluation episodes of a checkpoint - the exact column the official "
    "plot script reads (scripts/inspect_results.py).  Bands = +/-1 std ACROSS THE 3 SEEDS (100/101/102); thin lines = individual seeds.\n"
    "Paper Fig.2(b) bands are over 5 independent runs, so band WIDTHS are NOT comparable (ours is a 3-seed band).  Separately, "
    "the per-run std over the 5 eval episodes is exactly ZERO for\n"
    "SMPL-Jump / MyoLeg-Walk / Ostrich-Run (their 5 episodes are bit-identical -> effective n=1 per point: a property of the evaluation "
    "protocol, NOT evidence of stability) - so those panels' point-to-point\n"
    "wobble is checkpoint noise, not measured uncertainty.  H1 panels have 19 evaluation points, not 20 (70 envs does not divide the "
    "logging interval - integer divisibility, not a failure).\n"
    "Budgets/workers: 5M@80env w=20 (SMPL, PenTwirl), 5M@70 w=14 (H1-Run), 20M@70 w=14 (H1-Balance), 20M@80 w=20 (MyoLeg), "
    "10M@80 w=2 (Ostrich - official 80 env, only workers lowered to fit the 28 GB WSL memory limit).\n"
    "Only run directories carrying `.formal_done` are used (4 stale/aborted dirs excluded).  Panel order follows Figure 2(b): "
    "row 2 = musculoskeletal benchmarks only.  Dims |S|/|A| from each run's deterministic.txt, matching Figure 2(a)."
)


def read_eval(run_dir):
    rows = []
    with open(os.path.join(run_dir, 'log.csv'), newline='') as f:
        rd = csv.reader(f)
        hdr = next(rd)
        try:
            i_s, i_a, i_d = hdr.index('step'), hdr.index('avg_ret'), hdr.index('std_ret')
        except ValueError:
            i_s, i_a, i_d = 0, 1, 2
        for r in rd:
            if not r or not r[i_s].strip():
                continue
            rows.append((int(float(r[i_s])), float(r[i_a]), float(r[i_d])))
    rows.sort()
    return rows


def discover(env_id):
    """-> {alg: {seed: [(step,avg,std), ...]}}，只取带 .formal_done 的目录。
    整格目录不存在（例如从没跑过 Ostrich）时返回 {}，由 main() 统一打缺格告警，
    不要在这里抛 FileNotFoundError —— 否则 5 环境的日志会让整个出图脚本崩掉。"""
    d = os.path.join(ROOT, env_id)
    res = {}
    if not os.path.isdir(d):
        return res
    for name in sorted(os.listdir(d)):
        m = re.match(r"^(qflex|sac|crossq|qsm|sdac|dacer)_(.+?)_(s\d+)_(s\d+)$", name)
        if not m:
            continue
        algo, ts, s1, s2 = m.groups()
        if s1 != s2:
            continue
        p = os.path.join(d, name)
        if not os.path.exists(os.path.join(p, '.formal_done')):
            continue
        res.setdefault(algo, {})[s1] = read_eval(p)
    return res


def load_all():
    data, notes = {}, []
    for name, env_id, *_ in ENVS:
        data[env_id] = discover(env_id)
        for alg in ALGS:
            got = sorted(data[env_id].get(alg, {}))
            if got != SEEDS:
                notes.append('%s/%s seeds=%s' % (env_id, alg, got))
    return data, notes


def stack(per_seed):
    """per_seed: {seed: [(step,avg,std)]} -> steps, Y[nseed,npts]（要求步点一致）"""
    seeds = sorted(per_seed)
    grids = [tuple(r[0] for r in per_seed[s]) for s in seeds]
    if len(set(grids)) != 1:
        raise SystemExit('step grids differ: ' + str({s: len(g) for s, g in zip(seeds, grids)}))
    steps = np.array(grids[0], float)
    Y = np.array([[r[1] for r in per_seed[s]] for s in seeds], float)
    return seeds, steps, Y


def mean_sd(Y):
    """跨 seed 的 mean / std(ddof=1)；n=1 时 std=0"""
    m = Y.mean(axis=0)
    if Y.shape[0] > 1:
        sd = Y.std(axis=0, ddof=1)
    else:
        sd = np.zeros_like(m)
    return m, sd


def read_paper_ostrich():
    """读论文 Ostrich 面板的像素量测曲线；文件不存在时返回 4 个空数组（调用方跳过对比）。"""
    xs, ym, lo, hi = [], [], [], []
    p = os.path.join(PAPER, 'paper_ostrich_qflex_curve.csv') if PAPER else ''
    if not p or not os.path.isfile(p):
        return (np.array([]),) * 4
    with open(p, newline='') as f:
        for r in csv.DictReader(f):
            try:
                xs.append(float(r['steps']))
                ym.append(float(r['value_median']))
            except (ValueError, KeyError):
                continue
            for key, box in (('band_lo', lo), ('band_hi', hi)):
                v = (r.get(key) or '').strip()
                try:
                    box.append(float(v))
                except ValueError:
                    box.append(np.nan)
    return np.array(xs), np.array(ym), np.array(lo), np.array(hi)


def ma(ys, k=3):
    ys = np.asarray(ys, float)
    if len(ys) < k:
        return ys, ys
    return np.convolve(ys, np.ones(k) / k, mode='valid'), None


def draw_panel(ax, meta, d, smooth=False, paper_line=None, legend=True, show_seeds=True):
    name, env_id, ns, na, nenv, budget, w = meta
    if not d:
        # ★ 整格无数据（最常见：跑批没启用 Ostrich）⇒ 画占位，别让整张 6 格图崩掉。
        #   缺格信息由 main() 里的醒目告警统一打印。
        ax.set_xlim(0, budget / 1e6 * 1.04)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.56, 'NO DATA', ha='center', va='center', transform=ax.transAxes,
                fontsize=13, color='0.55', weight='bold')
        ax.text(0.5, 0.42, 'no run found for this task', ha='center', va='center',
                transform=ax.transAxes, fontsize=7.5, color='0.6')
        ax.set_title('%s  (%s)\n|S|=%d |A|=%d   %gM steps @%d env  w=%d   pts=-   seeds=-'
                     % (name, env_id, ns, na, budget / 1e6, nenv, w), fontsize=8.0, pad=5)
        ax.set_xlabel('training steps (millions)', fontsize=8)
        ax.set_ylabel('episode return', fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.28)
        return
    vlo, vhi = [], []
    for alg in sorted(d, key=lambda a: ALG_ORDER[a]):
        seeds, x, Y = stack(d[alg])
        m, sd = mean_sd(Y)
        xm = x / 1e6
        vlo.append(float(m.min()))
        vhi.append(float(m.max()))
        is_q = (alg == 'qflex')
        lw, z = (2.5, 6) if is_q else (1.6, 3)
        if sd.max() > 1e-9:
            ax.fill_between(xm, m - sd, m + sd, color=COLORS[alg], alpha=0.14, lw=0, zorder=z - 1)
        if show_seeds and Y.shape[0] > 1:
            for k in range(Y.shape[0]):
                ax.plot(xm, Y[k], '-', color=COLORS[alg], lw=0.7, alpha=0.30, zorder=z - 1)
        if smooth:
            ys, _ = ma(m, 3)
            ax.plot(xm[2:], ys, color=COLORS[alg], lw=lw, zorder=z, label=alg)
            ax.plot(xm, m, '.', color=COLORS[alg], ms=2.1, zorder=z - 1)
        else:
            ax.plot(xm, m, '-o', color=COLORS[alg], lw=lw,
                    ms=(3.8 if is_q else 2.8), zorder=z, label=alg)
    if paper_line is not None:
        ax.axhline(paper_line, color='0.25', lw=1.2, ls=':', zorder=2,
                   label='paper Qflex $\\approx$%.1fe4 (pixel-measured)' % (paper_line / 1e4))
    ax.axhline(0, color='0.8', lw=0.8, ls='--', zorder=1)
    ax.set_xlim(0, budget / 1e6 * 1.04)
    vlo_min = min(vlo)
    vhi_max = max(vhi)
    if paper_line is not None:
        vhi_max = max(vhi_max, paper_line)
    span = max(vhi_max - min(vlo_min, 0.0), 1.0)
    ax.set_ylim(min(vlo_min, 0.0) - 0.10 * span, vhi_max + 0.12 * span)
    ax.grid(alpha=0.28)
    npts = len(x)
    ax.set_title('%s  (%s)\n|S|=%d |A|=%d   %gM steps @%d env  w=%d   pts=%d   seeds=3'
                 % (name, env_id, ns, na, budget / 1e6, nenv, w, npts), fontsize=8.0, pad=5)
    ax.set_xlabel('training steps (millions)', fontsize=8)
    ax.set_ylabel('episode return', fontsize=8)
    ax.tick_params(labelsize=7.5)
    if legend:
        ax.legend(fontsize=6.6, loc='best', framealpha=0.92, ncol=1)


def main():
    data, notes = load_all()
    if notes:
        print('!! incomplete:', notes)
    print('envs loaded:', list(data))

    # ---------- ★ 整格缺失告警（最常见原因：跑批时没启用 Ostrich）----------
    # 不报错退出（有人只想要 5 格），但必须**醒目**——否则会静默出一张"看起来是 6 格"的图。
    empty = [e[1] for e in ENVS if not data[e[1]]]
    if empty:
        print('\n' + '!' * 78)
        print('!! 缺数据的环境：%s' % ', '.join(empty))
        print('!! 这些格子会是空面板，排名图对应位置是 NaN —— 出的图**不是完整的 6 格**。')
        if 'OstrichRun-v1' in empty:
            print('!! Ostrich 是最常见的一个：底层跑批脚本里它默认关闭，')
            print('!! 需要 WITH_OSTRICH=1（见 run/run_envs_formal_v5.sh 文件头）。')
        print('!! 每格需要的 run 目录：%s/<env_id>/<alg>_*_s<seed>_<suffix>/ （要有 .formal_done）'
              % os.environ.get('QFLEX_LOGS', '<QFLEX_LOGS>'))
        print('!' * 78 + '\n')

    # ---------- per-seed 原始 CSV ----------
    with open(os.path.join(OUT, 'six_grids_3seed_raw.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['env', 'env_id', 'alg', 'seed', 'step', 'avg_ret', 'std_ret'])
        for name, env_id, *_ in ENVS:
            for alg in ALGS:
                for s in sorted(data[env_id].get(alg, {})):
                    for st, a, sd in data[env_id][alg][s]:
                        wr.writerow([name, env_id, alg, int(s[1:]), st, a, sd])

    # ---------- 汇总 CSV：均值/跨 seed std ----------
    summ = []
    print('\n%-18s %-7s %6s %11s %10s %11s %10s %10s' %
          ('env', 'alg', 'nseed', 'final_mean', 'final_sd', 'best_mean', 'best_sd', 'maxperstd'))
    for name, env_id, *_ in ENVS:
        for alg in sorted(data[env_id], key=lambda a: ALG_ORDER[a]):
            seeds, x, Y = stack(data[env_id][alg])
            finals = Y[:, -1]
            bests = Y.max(axis=1)
            per_std = max(r[2] for s in seeds for r in data[env_id][alg][s])
            summ.append([name, env_id, alg, len(seeds),
                         float(finals.mean()), float(finals.std(ddof=1)) if len(seeds) > 1 else 0.0,
                         float(bests.mean()), float(bests.std(ddof=1)) if len(seeds) > 1 else 0.0,
                         float(per_std)])
            print('%-18s %-7s %6d %11.1f %10.1f %11.1f %10.1f %10.3g' %
                  (name, alg, len(seeds), finals.mean(),
                   finals.std(ddof=1) if len(seeds) > 1 else 0.0,
                   bests.mean(), bests.std(ddof=1) if len(seeds) > 1 else 0.0, per_std))
    with open(os.path.join(OUT, 'six_grids_3seed_summary.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['env', 'env_id', 'alg', 'n_seeds', 'final_mean', 'final_sd_across_seeds',
                     'best_mean', 'best_sd_across_seeds', 'max_per_run_std_ret'])
        wr.writerows(summ)

    # ---------- 主图 2x3 ----------
    for smooth, fname, supt in [
        (False, 'fig_6grid_3seed_raw.png',
         'Qflex benchmark reproduction  -  all 6 tasks  -  6 algorithms  x  3 seeds (100/101/102)  '
         '-  thick = 3-seed mean, band = +/-1 std across seeds  (official metric: log.csv avg_ret)'),
        (True, 'fig_6grid_3seed_ma3.png',
         'Qflex benchmark reproduction  -  all 6 tasks  -  3 seeds  -  3-point moving average of the 3-seed mean '
         '(display only; raw mean points shown as dots)'),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.8), dpi=200)
        axes = axes.ravel()
        for i, meta in enumerate(ENVS):
            env_id = meta[1]
            draw_panel(axes[i], meta, data[env_id], smooth=smooth,
                       paper_line=(PAPER_MYOLEG_QFlex if env_id == 'myoLegWalk-v0' else None))
        fig.suptitle(supt, fontsize=11.2, y=0.989)
        fig.text(0.006, 0.008, FOOT, va='bottom', ha='left', fontsize=6.2,
                 family='DejaVu Sans Mono', linespacing=1.5)
        fig.tight_layout(rect=(0, 0.125, 1, 0.955))
        p = os.path.join(OUT, fname)
        fig.savefig(p)
        plt.close(fig)
        print('saved:', p)

    # ---------- 排名图（带跨 seed 误差棒）----------
    fig, ax = plt.subplots(figsize=(13.2, 4.9), dpi=200)
    names = [e[0] for e in ENVS]
    width = 0.13
    idx = np.arange(len(names))
    for k, alg in enumerate(ALGS):
        vals, errs = [], []
        for name, env_id, *_ in ENVS:
            if alg not in data[env_id]:
                vals.append(np.nan)
                errs.append(0.0)
                continue
            seeds, x, Y = stack(data[env_id][alg])
            b = Y.max(axis=1)
            ref = data[env_id]['qflex']
            _, _, YQ = stack(ref)
            q = YQ.max(axis=1).mean()
            vals.append(b.mean() / q if abs(q) > 1e-9 else np.nan)
            errs.append((b.std(ddof=1) if len(seeds) > 1 else 0.0) / abs(q) if abs(q) > 1e-9 else 0.0)
        ax.bar(idx + (k - (len(ALGS) - 1) / 2) * width, vals, width, color=COLORS[alg],
               label=alg, edgecolor='white', lw=0.5, zorder=3,
               yerr=errs, error_kw=dict(ecolor='0.25', lw=0.7, capsize=1.6))
        for xi, v in zip(idx + (k - (len(ALGS) - 1) / 2) * width, vals):
            if not np.isnan(v):
                txt = ('%.2f' % v) if v >= 0.01 else ('%.1e' % v)
                ax.text(xi, v * 1.16, txt, ha='center', fontsize=5.6, rotation=90, color='0.3')
    ax.axhline(1.0, color='0.3', lw=1.1, ls='--', zorder=2)
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 20)
    ax.set_xticks(idx)
    ax.set_xticklabels(['%s\n(budget %.0fM)' % (n, b / 1e6) for (n, _, _, _, _, b, _) in ENVS],
                       fontsize=8)
    ax.set_ylabel('best avg_ret  /  qflex best avg_ret   (log)', fontsize=8.5)
    ax.grid(alpha=0.25, axis='y')
    ax.legend(fontsize=7.6, ncol=6, loc='upper center', framealpha=0.93)
    ax.set_title('Relative best performance across all 6 tasks, normalised to Qflex = 1.0  '
                 '(bar = mean over 3 seeds of per-seed best avg_ret, / mean qflex best; error bar = std across seeds)',
                 fontsize=9.6, pad=8)
    fig.text(0.5, 0.015, 'Official benchmark convention: best checkpoint value '
                         '(scripts/inspect_results.py: load_best_results -> avg_ret.idxmax()).  '
                         'N = 3 seeds; error bars are std across seeds.',
             ha='center', fontsize=7.2, color='0.4')
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    p = os.path.join(OUT, 'fig_6grid_3seed_rank.png')
    fig.savefig(p)
    plt.close(fig)
    print('saved:', p)

    # ---------- Ostrich：我们 3 seed vs 论文 5 run ----------
    ox, oym, olo, ohi = read_paper_ostrich()
    d = data['OstrichRun-v1']
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.6, 5.5), dpi=200)
    for ax, symlog in ((axL, False), (axR, True)):
        if not d:
            ax.text(0.5, 0.5, 'NO DATA — no Ostrich runs found', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, color='0.55', weight='bold')
        for alg in sorted(d, key=lambda a: ALG_ORDER[a]):
            seeds, x, Y = stack(d[alg])
            m, sd = mean_sd(Y)
            xm = x / 1e6
            is_q = (alg == 'qflex')
            if sd.max() > 1e-9:
                ax.fill_between(xm, m - sd, m + sd, color=COLORS[alg], alpha=0.14, lw=0, zorder=2)
            if Y.shape[0] > 1:
                for k in range(Y.shape[0]):
                    ax.plot(xm, Y[k], '-', color=COLORS[alg], lw=0.7, alpha=0.30, zorder=3)
            ax.plot(xm, m, '-o', color=COLORS[alg], lw=(2.4 if is_q else 1.4),
                    ms=(4.0 if is_q else 2.6), zorder=(6 if is_q else 4),
                    label=('ours %s (3-seed mean)' % alg) if symlog is False else alg)
        ax.fill_between(ox / 1e6, olo, ohi, color='0.45', alpha=0.16, lw=0, zorder=1)
        ax.plot(ox / 1e6, oym, color='0.25', lw=1.9, ls='--', zorder=6,
                label=('paper QFLEX mean $\\pm$1 std (5 runs)' if symlog is False else 'paper QFLEX mean'))
        ax.axhline(0, color='0.8', lw=0.8, ls='--', zorder=1)
        ax.set_xlim(0, 10.4)
        ax.grid(alpha=0.28, which='both')
        ax.set_xlabel('training steps (millions)', fontsize=9)
        ax.tick_params(labelsize=8)
        if symlog:
            ax.set_yscale('symlog', linthresh=10)
            ax.set_ylabel('episode return  (symlog)', fontsize=9)
            ax.set_title('same data, symlog y  -  all 6 algorithms\n'
                         '(Qflex ~1e4 vs sac/qsm ~1e1: three orders of magnitude)', fontsize=9.4)
            ax.legend(fontsize=7.2, loc='lower right', framealpha=0.93, ncol=2)
        else:
            ax.set_ylim(-1500, 20500)
            ax.set_ylabel('episode return', fontsize=9)
            ax.set_title('Ostrich-Run (OstrichRun-v1, 596 dim obs / 120 dim act)\n'
                         'ours (3-seed mean +/- seed std) vs paper mean of 5 runs', fontsize=9.4)
            ax.legend(fontsize=6.0, loc='center left', ncol=2, framealpha=0.93)
    fig.suptitle('Ostrich-Run: this reproduction vs paper Figure 2(b)  '
                 '(ours = log.csv avg_ret; paper curve pixel-measured from the PDF, ~5%)',
                 fontsize=10.5, y=0.985)
    fig.text(0.5, 0.042,
             'Paper mean & band = 5 independent runs, pixel-measured from Fig.2(b) (~5% accuracy).  '
             'Ours = 3 seeds (100/101/102).', ha='center', fontsize=6.9, color='0.38')
    fig.text(0.5, 0.010,
             'Ostrich has reset_noise_scale=0 -> every evaluation is a single deterministic sample of the current checkpoint; '
             'per-point dips are checkpoint noise, so only the trend / best value is interpretable.',
             ha='center', fontsize=6.9, color='0.38')
    fig.tight_layout(rect=(0, 0.085, 1, 0.94))
    p = os.path.join(OUT, 'fig_ostrich_3seed_vs_paper.png')
    fig.savefig(p)
    plt.close(fig)
    print('saved:', p)

    # ---------- 控制台：best 汇总 + 名次 ----------
    print('\n=== per-env best (mean over 3 seeds) ===')
    ranks = {a: [] for a in ALGS}
    for name, env_id, *_ in ENVS:
        if not data[env_id]:
            print('%-18s (no data — 已跳过，缺格原因见上面告警)' % name)
            continue
        row = []
        for alg in ALGS:
            if alg not in data[env_id]:
                row.append((alg, float('nan'), float('nan')))
                continue
            seeds, x, Y = stack(data[env_id][alg])
            b = Y.max(axis=1)
            row.append((alg, float(b.mean()), float(b.std(ddof=1)) if len(seeds) > 1 else 0.0))
        # NaN（缺算法）排到最后，避免 NaN 参与比较时顺序随机
        order = sorted(row, key=lambda t: (t[1] != t[1], -t[1] if t[1] == t[1] else 0.0))
        for r, (alg, m, s) in enumerate(order, 1):
            ranks[alg].append(r)
        print('%-18s %s' % (name, '  '.join('%s %.0f±%.0f(#%d)' % (a, m, s, r)
                                            for r, (a, m, s) in enumerate(order, 1))))
    ntasks = len(ranks['qflex']) if ranks.get('qflex') else 0
    print('\n=== mean rank over %d task(s) (N=3 seeds) ===' % ntasks)
    for a in sorted(ALGS, key=lambda k: np.mean(ranks[k]) if ranks[k] else 1e9):
        print('  %-7s %.2f   (%s)' % (a, float(np.mean(ranks[a])),
                                      ' '.join(str(v) for v in ranks[a])))

    if len(oym):
        print('\n=== paper reference ===')
        print('paper QFLEX mean (Ostrich, 5-run, pixel-measured): final %.0f  max %.0f  band half-width (median) %.0f'
              % (PAPER_OSTRICH_END, np.nanmax(oym), np.nanmedian((ohi - olo) / 2.0)))
    else:
        print('\n[skip] 未提供 QFLEX_PAPER → 跳过 "Ostrich vs paper" 对比图与论文参考值')


if __name__ == '__main__':
    main()
