"""Builds every figure/table used in the slides from results/*.json (and one interpolation run)."""
import os, glob, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ppo_numpy as P

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#E3EAEF", "figure.dpi": 150})
INK, DEEP = "#1B2430", "#065A82"
LABELS = {("none", 0.0): "No clipping / penalty", ("clip", 0.1): "Clip eps=0.1", ("clip", 0.2): "Clip eps=0.2",
          ("clip", 0.3): "Clip eps=0.3", ("kl_adaptive", 0.2): "Adaptive KL (d_targ=0.01)", ("kl_fixed", 0.2): "Fixed KL (beta=1)"}
COL = {("none", 0.0): "#C0392B", ("clip", 0.1): "#7FB3CE", ("clip", 0.2): DEEP, ("clip", 0.3): "#1C9AA8",
       ("kl_adaptive", 0.2): "#E08E0B", ("kl_fixed", 0.2): "#7A6FA8"}

def load():
    runs = {}
    for f in glob.glob("results/*.json"):
        d = json.load(open(f)); runs.setdefault((d["setting"], d["eps"] if d["setting"] == "clip" else (0.0 if d["setting"] == "none" else 0.2)), []).append(d)
    return runs

# ---- Fig A: reproduction of paper Fig.1 (L^CLIP vs r), analytic
def fig_clip_function(eps=0.2):
    r = np.linspace(0, 2, 400)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    for a, A, ttl in ((ax[0], 1, "A > 0  (good action)"), (ax[1], -1, "A < 0  (bad action)")):
        a.plot(r, r * A, "--", color="#9AA9B5", label="unclipped  r*A  (L^CPI)")
        a.plot(r, np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A), color=DEEP, lw=2.6, label="L^CLIP")
        a.axvline(1 - eps, color="#B8C4CC", lw=1); a.axvline(1 + eps, color="#B8C4CC", lw=1)
        a.plot([1], [A], "o", color="#C0392B"); a.set_title(ttl, color=INK, fontsize=11); a.set_xlabel("probability ratio r")
    ax[0].set_ylabel("objective value"); ax[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(f"Our re-plot of paper Fig. 1 (eps = {eps}): gradient dies once r leaves [1-eps, 1+eps] in the 'improving' direction", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig("fig_A_clip_function.png"); plt.close(fig)

# ---- Fig B: learning curves (mean +- std over seeds)
def fig_curves(runs):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for k in LABELS:
        ys = np.array([d["ret"] for d in runs[k]]); x = np.array(runs[k][0]["step"])
        m, s = ys.mean(0), ys.std(0)
        ax.plot(x, m, color=COL[k], lw=2.8 if k == ("clip", 0.2) else 1.8, label=LABELS[k])
        ax.fill_between(x, m - s, m + s, color=COL[k], alpha=0.10)
    ax.set_xlabel("environment steps"); ax.set_ylabel("episode return (CartPole-v1, max 500)")
    ax.set_title(f"Learning curves, mean +/- std over {len(runs[('clip',0.2)])} seeds", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right"); fig.tight_layout(); fig.savefig("fig_B_curves.png"); plt.close(fig)

# ---- Fig C: how far each update moves the policy (mean KL) - the trust-region effect
def fig_kl(runs):
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    keys = list(LABELS); vals = [np.array([np.mean(d["kl"]) for d in runs[k]]) for k in keys]
    ax.bar(range(len(keys)), [v.mean() for v in vals], yerr=[v.std() for v in vals], color=[COL[k] for k in keys], capsize=3)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([LABELS[k].replace(" (", "\n(") for k in keys], fontsize=7.5)
    ax.set_yscale("log"); ax.set_ylabel("mean KL(old || new) per update (log scale)")
    for i, v in enumerate(vals): ax.text(i, v.mean() * 1.35, f"{v.mean():.4f}", ha="center", fontsize=8, color=INK)
    ax.set_title("Size of each policy update (averaged over training)", color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig("fig_C_kl.png"); plt.close(fig)

# ---- Fig D: interpolation between old and new parameters (paper Fig. 2 idea, extended beyond alpha=1)
def fig_interp(seed=0, eps=0.2, capture_iter=8):
    P.SNAP.clear(); P.run("clip", eps, seed, 2048 * (capture_iter + 1), capture_iter=capture_iter)
    old, new, B = P.SNAP["old"], P.SNAP["new"], P.SNAP["B"]
    al = np.linspace(0, 8, 81); rows = []
    for a in al:
        th = [o + a * (n - o) for o, n in zip(old, new)]
        rows.append(P.surrogates(th, B["obs"], B["act"], B["adv"], B["logp"], B["probs"], eps)[:3])
    rows = np.array(rows)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.7))
    ax[0].plot(al, rows[:, 0], color="#9AA9B5", lw=2.2, label="L^CPI (no clip)"); ax[0].plot(al, rows[:, 1], color=DEEP, lw=2.8, label="L^CLIP (eps=0.2)")
    ax[1].plot(al, rows[:, 2], color="#E08E0B", lw=2.4)
    for a_ in ax: a_.axvline(1.0, color="#B8C4CC", ls=":"); a_.set_xlabel("alpha  (0 = old, 1 = after update, >1 = extrapolate)")
    ax[0].set_ylabel("surrogate value on the batch"); ax[0].legend(frameon=False, fontsize=9, loc="upper left")
    ax[1].set_ylabel("KL(old || new)"); ax[0].set_title("L^CLIP peaks near the real update, then falls", fontsize=10.5, color=INK); ax[1].set_title("...while the policy keeps drifting", fontsize=10.5, color=INK)
    fig.tight_layout(); fig.savefig("fig_D_interp.png"); plt.close(fig)
    np.savetxt("interp_values.csv", np.c_[al, rows], delimiter=",", header="alpha,L_CPI,L_CLIP,KL", comments="")

# ---- Table (paper Table 1 style)
def table(runs):
    rnd = P.random_baseline()
    sc = {k: np.array([d["score_last100"] for d in runs[k]]) for k in LABELS}
    best = max(v.mean() for v in sc.values())
    auc = {k: np.array([np.mean(d["ret"]) for d in runs[k]]) for k in LABELS}
    lines = ["setting,score_last100_mean,score_last100_std,normalized(random=0;best=1),mean_return_all_training,mean_KL,n_seeds"]
    for k in LABELS:
        norm = (sc[k].mean() - rnd) / (best - rnd)
        lines.append(f"{LABELS[k]},{sc[k].mean():.1f},{sc[k].std():.1f},{norm:.2f},{auc[k].mean():.1f},{np.mean([np.mean(d['kl']) for d in runs[k]]):.4f},{len(sc[k])}")
    open("table_results.csv", "w").write("\n".join(lines) + f"\nrandom_policy_score,{rnd:.1f}\n"); print("\n".join(lines), "\nrandom:", rnd)

if __name__ == "__main__":
    runs = load(); fig_clip_function(); fig_curves(runs); fig_kl(runs); table(runs); fig_interp()
