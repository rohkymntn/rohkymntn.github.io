"""How hazardous is Countdown? Exact residual oracle over rational multisets + statistics."""
import random, json, math, itertools, functools, time
from fractions import Fraction
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = str(__import__('pathlib').Path(__file__).resolve().parent)
CLAY, INK, MUTED, GREEN, LINE, PAPER = "#BE5B3A", "#23211B", "#807A6B", "#4F7F5B", "#E2DCCC", "#FBF9F3"
BLUE = "#1f3fd0"
plt.rcParams.update({"figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.grid": True, "grid.color": LINE,
    "grid.linewidth": 0.7, "font.family": "DejaVu Sans", "font.size": 10, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False})

OPS = ("+", "-", "*", "/")
def apply(a, b, op):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/": return None if b == 0 else a / b

def legal_actions(state):
    """All (i, j, op, result, next_state) for an ordered pair of distinct elements."""
    s = list(state); out = []
    for i in range(len(s)):
        for j in range(len(s)):
            if i == j: continue
            for op in OPS:
                if op in "+*" and i > j: continue          # commutative: one orientation
                r = apply(s[i], s[j], op)
                if r is None: continue
                rest = [s[k] for k in range(len(s)) if k not in (i, j)]
                out.append((s[i], s[j], op, r, tuple(sorted(rest + [r]))))
    return out

@functools.lru_cache(maxsize=None)
def dist(state, target):
    """shortest residual horizon d_y(s): 0 if target in state, inf if dead."""
    if target in state: return 0
    if len(state) == 1: return math.inf
    best = math.inf
    for *_, nxt in legal_actions(state):
        d = dist(nxt, target)
        if d + 1 < best: best = d + 1
    return best

def instance(rng, n=4, lo=1, hi=99):
    nums = tuple(sorted(Fraction(rng.randint(lo, hi)) for _ in range(n)))
    return nums, Fraction(rng.randint(1, 99))

rng = random.Random(0)
N = 3000
t0 = time.time()
stats = {"solvable": 0, "root_alive_frac": [], "knife_edge": 0, "hazard_by_size": {4: [], 3: []},
         "greedy_dead": 0, "greedy_total": 0, "dist_root": [], "n_states": []}
examples = []
for _ in range(N):
    s0, y = instance(rng)
    d0 = dist(s0, y)
    if d0 == math.inf: continue
    stats["solvable"] += 1; stats["dist_root"].append(d0)
    # walk all alive states (BFS) and record hazard = fraction of legal actions that are dead
    seen = {s0}; frontier = [s0]
    while frontier:
        s = frontier.pop()
        acts = legal_actions(s)
        alive = [a for a in acts if dist(a[4], y) < math.inf]
        if len(s) in stats["hazard_by_size"]:
            stats["hazard_by_size"][len(s)].append(1 - len(alive) / len(acts))
        if s == s0:
            stats["root_alive_frac"].append(len(alive) / len(acts))
            if len(alive) == 1: stats["knife_edge"] += 1
            # greedy heuristic: the legal move whose result is numerically closest to the target
            g = min(acts, key=lambda a: abs(a[3] - y))
            stats["greedy_total"] += 1
            if dist(g[4], y) == math.inf:
                stats["greedy_dead"] += 1
                if len(examples) < 6:
                    examples.append({"numbers": [int(v) for v in s0], "target": int(y), "greedy": f"{g[0]} {g[2]} {g[1]} = {g[3]}",
                                     "alive_moves": [f"{a[0]} {a[2]} {a[1]} = {a[3]}" for a in alive][:4]})
        for a in alive:
            if a[4] not in seen and len(a[4]) > 1:
                seen.add(a[4]); frontier.append(a[4])
    stats["n_states"].append(len(seen))
print(f"{N} instances, {stats['solvable']} solvable, {time.time()-t0:.1f}s")
solv = stats["solvable"]
summary = {
    "n_instances": N, "solvable": solv, "solvable_frac": solv / N,
    "root_alive_frac_mean": float(np.mean(stats["root_alive_frac"])),
    "root_dead_frac_mean": 1 - float(np.mean(stats["root_alive_frac"])),
    "knife_edge_frac": stats["knife_edge"] / solv,
    "hazard_size4_mean": float(np.mean(stats["hazard_by_size"][4])),
    "hazard_size3_mean": float(np.mean(stats["hazard_by_size"][3])),
    "greedy_dead_frac": stats["greedy_dead"] / stats["greedy_total"],
    "mean_alive_states": float(np.mean(stats["n_states"])),
    "dist_root_hist": {int(k): int(v) for k, v in zip(*np.unique(stats["dist_root"], return_counts=True))},
    "examples": examples,
}
json.dump(summary, open(f"{OUT}/hazard_stats.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "examples"}, indent=1))
for e in examples: print(e)

# ── figure 1: hazard distribution ──
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
axs[0].hist(stats["root_alive_frac"], bins=20, color=CLAY, edgecolor=PAPER)
axs[0].set_xlabel("fraction of legal first moves that stay alive"); axs[0].set_ylabel("solvable instances")
axs[0].set_title(f"At the root, {100*summary['root_dead_frac_mean']:.0f} % of legal moves are dead", fontsize=10)
h4, h3 = stats["hazard_by_size"][4], stats["hazard_by_size"][3]
axs[1].hist([h4, h3], bins=15, color=[CLAY, GREEN], label=[f"4 numbers left (mean {np.mean(h4):.2f})", f"3 numbers left (mean {np.mean(h3):.2f})"], edgecolor=PAPER)
axs[1].set_xlabel("fraction of legal moves that are dead"); axs[1].set_ylabel("alive states visited"); axs[1].legend(fontsize=8); axs[1].set_title("Hazard rate over all alive states", fontsize=10)
labels = ["greedy move\n(closest to target)\nis dead", "exactly one\nalive first move", "solvable\ninstances"]
vals = [summary["greedy_dead_frac"], summary["knife_edge_frac"], summary["solvable_frac"]]
axs[2].barh(labels, vals, color=[CLAY, INK, MUTED]); axs[2].set_xlim(0, 1); axs[2].set_xlabel("fraction")
for i, v in enumerate(vals): axs[2].text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
axs[2].set_title("Random 4-number Countdown, targets 1–99", fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_vipo_hazard.png", dpi=170); plt.close()

# ── figure 2: Theorem 3 numerically ──
fig, ax = plt.subplots(figsize=(5.5, 3.5))
H = np.arange(2, 9)
for B, col in ((2, GREEN), (3, CLAY), (4, INK)):
    p = B ** (-(H - 1.0)); N_needed = np.log(1 / 0.05) / -np.log1p(-p)
    ax.semilogy(H, N_needed, "o-", color=col, label=f"terminal reward, B = {B}")
ax.semilogy(H, np.ones_like(H), "--", color=BLUE, label="viability oracle: 1 query")
ax.set_xlabel("depth H"); ax.set_ylabel("rollouts to see one success (95 %)"); ax.set_title("Credit assignment cost on the branching family", fontsize=10); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_vipo_theorem3.png", dpi=170); plt.close()

# ── figure 3: main results re-plotted from Table 1 / Table 2 / §6.4 ──
methods = ["SFT seed", "IPO", "VPO-A", "RLOO", "VPO-R"]
p1 = [0.322, 0.422, 0.486, 0.618, 0.884]; lo = [0.283, 0.380, 0.442, 0.575, 0.853]; hi = [0.364, 0.466, 0.530, 0.660, 0.909]
p8 = [0.804, 0.840, 0.862, 0.876, 0.950]; p16 = [0.888, 0.894, 0.900, 0.900, 0.962]; ms = [0.377, 0.435, 0.534, 0.680, 0.889]
cols = [MUTED, "#C9A227", BLUE, CLAY, GREEN]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
x = np.arange(5)
axs[0].bar(x, p1, color=cols, yerr=[np.subtract(p1, lo), np.subtract(hi, p1)], capsize=3, edgecolor=PAPER)
axs[0].set_xticks(x); axs[0].set_xticklabels(methods, fontsize=8); axs[0].set_ylim(0, 1); axs[0].set_title("Pass@1, fresh n = 500 (95 % Wilson CI)", fontsize=10)
for i, v in enumerate(p1): axs[0].text(i, v + 0.04, f"{v:.3f}", ha="center", fontsize=8)
w = 0.27
axs[1].bar(x - w, p1, w, color=cols, alpha=0.5, label="pass@1"); axs[1].bar(x, p8, w, color=cols, alpha=0.8, label="pass@8"); axs[1].bar(x + w, p16, w, color=cols, label="pass@16")
axs[1].set_xticks(x); axs[1].set_xticklabels(methods, fontsize=8); axs[1].set_ylim(0, 1); axs[1].set_title("Pass@k compresses at k = 16 (0.89–0.96)", fontsize=10)
axs[1].legend(fontsize=8, loc="lower right")
budget = ["RLOO", "VPO-A", "VPO-R"]; per_label = [0.060, 0.076, 0.138]; per_tok = [0.012, 0.316, 0.576]
ax2 = axs[2]; xx = np.arange(3)
ax2.bar(xx - 0.2, per_label, 0.4, color=[CLAY, BLUE, GREEN], alpha=0.55, label="Pass@1 per 10k reward labels")
ax2b = ax2.twinx(); ax2b.bar(xx + 0.2, per_tok, 0.4, color=[CLAY, BLUE, GREEN], label="Pass@1 per M training tokens"); ax2b.grid(False)
ax2.set_xticks(xx); ax2.set_xticklabels(budget); ax2.set_ylabel("per 10k labels"); ax2b.set_ylabel("per M tokens")
ax2.set_title("Efficiency: 102k labels / 50.7M tok (RLOO) vs 64k / 1.54M (VPO)", fontsize=9)
h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels(); ax2.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_vipo_results.png", dpi=170); plt.close()

# ── figure 4: inference ablation (§6.4) + training ablation (Table 3) ──
fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
k = [1, 2, 4, 8, 16]; vr = [0.17, 0.43, 0.63, 0.82, 0.87]
axs[0].plot(k, vr, "o-", color=GREEN, lw=2, label="VPO-R (rerank among candidates)"); axs[0].axhline(0.36, color=BLUE, ls="--", label="VPO-A, first legal of 8")
axs[0].set_xscale("log", base=2); axs[0].set_xticks(k); axs[0].set_xticklabels(k); axs[0].set_xlabel("candidates sampled per state"); axs[0].set_ylabel("Pass@1"); axs[0].set_ylim(0, 1)
axs[0].legend(fontsize=8); axs[0].set_title("Where the test-time gain comes from (n = 500)", fontsize=10)
conds = ["baseline", "T1 verbose prompt", "T3 no oracle fallback", "T4 dead reward ρ=0", "T2 repair=no_alive", "t2 repair=bad"]
va = [0.707, 0.707, 0.780, 0.733, 0.620, 0.193]; vrr = [0.767, 0.813, 0.807, 0.800, 0.820, 0.533]; ill = [0.049, 0.021, 0.045, 0.021, 0.243, 0.688]
y = np.arange(len(conds))
axs[1].barh(y - 0.2, va, 0.4, color=BLUE, label="VPO-A pass@1"); axs[1].barh(y + 0.2, vrr, 0.4, color=GREEN, label="VPO-R pass@1")
for i, v in enumerate(ill): axs[1].text(0.99, i, f"illegal {v:.3f}", va="center", ha="right", fontsize=7, color=CLAY if v > 0.1 else MUTED)
axs[1].set_yticks(y); axs[1].set_yticklabels(conds, fontsize=8); axs[1].invert_yaxis(); axs[1].set_xlim(0, 1); axs[1].set_xlabel("Pass@1 (retrained 200 steps, eval n = 150)")
axs[1].legend(fontsize=8, loc="lower right"); axs[1].set_title("Training ablation: robust except one repair rule", fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_vipo_ablations.png", dpi=170); plt.close()
print("figures done")
