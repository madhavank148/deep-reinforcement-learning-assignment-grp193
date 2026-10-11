"""
PPO from scratch (NumPy only) - Group 193, DRL Assignment, Problem V
Paper: Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347

What this file does
  1. Implements PPO (GAE + minibatch epochs) with a 64x64 tanh MLP policy and value net,
     with manual back-prop (no PyTorch/TF), so every gradient can be explained.
  2. Lets us swap ONLY the surrogate objective, to repeat the paper's Section 6.1 comparison:
        none        : L^CPI, no clipping / no penalty
        clip        : L^CLIP  (eps = 0.1, 0.2, 0.3)
        kl_adaptive : L^KLPEN with the adaptive beta rule (d_targ = 0.01)
        kl_fixed    : L^KLPEN with fixed beta
  3. Logs, for every iteration: episode return, mean KL(old||new) and clip fraction.

Usage
  python ppo_numpy.py run --setting clip --eps 0.2 --seed 0 --steps 150000
  python ppo_numpy.py ablation --seeds 5 --steps 150000      # all settings, parallel
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json, time
import numpy as np
import gymnasium as gym

# ----------------------------------------------------------------------------- network
def init_mlp(sizes, rng, last_gain):
    p = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        gain = last_gain if i == len(sizes) - 2 else np.sqrt(2.0)
        p += [rng.normal(0, 1, (a, b)) * gain / np.sqrt(a), np.zeros(b)]
    return p

def forward(p, x):
    h1 = np.tanh(x @ p[0] + p[1])
    h2 = np.tanh(h1 @ p[2] + p[3])
    return h2 @ p[4] + p[5], (x, h1, h2)

def backward(p, cache, dout):
    x, h1, h2 = cache
    g = [None] * 6
    g[4], g[5] = h2.T @ dout, dout.sum(0)
    d2 = (dout @ p[4].T) * (1 - h2 ** 2)
    g[2], g[3] = h1.T @ d2, d2.sum(0)
    d1 = (d2 @ p[2].T) * (1 - h1 ** 2)
    g[0], g[1] = x.T @ d1, d1.sum(0)
    return g

class Adam:
    def __init__(self, params, lr):
        self.lr, self.t = lr, 0
        self.m = [np.zeros_like(q) for q in params]
        self.v = [np.zeros_like(q) for q in params]
    def step(self, params, grads, max_norm=0.5):
        norm = np.sqrt(sum((g ** 2).sum() for g in grads))
        if norm > max_norm:
            grads = [g * max_norm / (norm + 1e-8) for g in grads]
        self.t += 1
        for i, g in enumerate(grads):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            params[i] -= self.lr * mh / (np.sqrt(vh) + 1e-8)

def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))

# ----------------------------------------------------------------------------- objectives
def surrogates(pol, obs, act, adv, logp_old, probs_old, eps):
    """Batch means of L^CPI, L^CLIP, KL(old||new) and clip-fraction (used for logging and Fig-2 style plot)."""
    logits, _ = forward(pol, obs)
    lp_all = log_softmax(logits)
    r = np.exp(lp_all[np.arange(len(act)), act] - logp_old)
    l_cpi = (r * adv).mean()
    l_clip = np.minimum(r * adv, np.clip(r, 1 - eps, 1 + eps) * adv).mean()
    kl = (probs_old * (np.log(probs_old + 1e-12) - lp_all)).sum(1).mean()
    clipfrac = (np.abs(r - 1) > eps).mean()
    return l_cpi, l_clip, kl, clipfrac

def ppo_update(pol, vf, opt_p, opt_v, B, cfg, beta, rng):
    obs, act, adv, ret = B["obs"], B["act"], B["adv"], B["ret"]
    lp_old, p_old = B["logp"], B["probs"]
    N, M, mode, eps = len(obs), cfg["minibatch"], cfg["mode"], cfg["eps"]
    for _ in range(cfg["epochs"]):
        perm = rng.permutation(N)
        for s in range(0, N, M):
            mb = perm[s:s + M]
            n = len(mb)
            # ---- policy: gradient of the loss w.r.t. the logits
            logits, c = forward(pol, obs[mb])
            lpa = log_softmax(logits)
            pn = np.exp(lpa)
            r = np.exp(lpa[np.arange(n), act[mb]] - lp_old[mb])
            A = adv[mb]
            if mode == "clip":
                # gradient flows only while the un-clipped term is the active branch of the min()
                active = ((A >= 0) & (r <= 1 + eps)) | ((A < 0) & (r >= 1 - eps))
                coef = -A * r * active
            else:
                coef = -A * r
            onehot = np.zeros_like(pn); onehot[np.arange(n), act[mb]] = 1
            dlog = coef[:, None] * (onehot - pn)             # d(-r*A)/d logits
            if mode in ("kl_adaptive", "kl_fixed"):
                dlog += beta * (pn - p_old[mb])               # d KL(old||new)/d logits
            H = -(pn * lpa).sum(1, keepdims=True)
            dlog += cfg["ent"] * pn * (lpa + H)               # entropy bonus (c2)
            opt_p.step(pol, backward(pol, c, dlog / n))
            # ---- value function: squared error
            v, cv = forward(vf, obs[mb])
            opt_v.step(vf, backward(vf, cv, (v - ret[mb][:, None]) / n))

# ----------------------------------------------------------------------------- training
SNAP = {}   # optional snapshot of (old params, new params, batch) for the Fig-2 style interpolation plot

def run(setting="clip", eps=0.2, seed=0, steps=150000, beta0=1.0, d_targ=0.01, T=2048, verbose=False, capture_iter=None):
    mode = {"none": "none", "clip": "clip", "kl_adaptive": "kl_adaptive", "kl_fixed": "kl_fixed"}[setting]
    cfg = dict(mode=mode, eps=eps, epochs=10, minibatch=64, lr=3e-4, gamma=0.99, lam=0.95, ent=0.01)
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    pol = init_mlp([4, 64, 64, 2], rng, 0.01)
    vf = init_mlp([4, 64, 64, 1], rng, 1.0)
    opt_p, opt_v = Adam(pol, cfg["lr"]), Adam(vf, cfg["lr"])
    beta = beta0
    o, _ = env.reset(seed=seed)
    ep_ret, ep_log = 0.0, []                     # (global step, return)
    log = dict(iter=[], step=[], ret=[], kl=[], clipfrac=[], beta=[])
    gstep, it = 0, 0
    while gstep < steps:
        O = np.zeros((T, 4)); Ac = np.zeros(T, int); R = np.zeros(T); D = np.zeros(T)
        V = np.zeros(T + 1); LP = np.zeros(T); PR = np.zeros((T, 2))
        for t in range(T):
            lg, _ = forward(pol, o[None])
            lpa = log_softmax(lg)[0]; pr = np.exp(lpa)
            a = rng.choice(2, p=pr)
            v = forward(vf, o[None])[0][0, 0]
            O[t], Ac[t], LP[t], PR[t], V[t] = o, a, lpa[a], pr, v
            o2, rew, term, trunc, _ = env.step(a)
            ep_ret += rew
            if trunc and not term:                      # time-limit: bootstrap instead of treating as terminal
                rew += cfg["gamma"] * forward(vf, o2[None])[0][0, 0]
            R[t], D[t] = rew, float(term or trunc)
            if term or trunc:
                ep_log.append((gstep + t + 1, ep_ret)); ep_ret = 0.0
                o, _ = env.reset()
            else:
                o = o2
        V[T] = forward(vf, o[None])[0][0, 0]
        gstep += T
        # ---- GAE(lambda)
        adv = np.zeros(T); last = 0.0
        for t in reversed(range(T)):
            nonterm = 1.0 - D[t]
            delta = R[t] + cfg["gamma"] * V[t + 1] * nonterm - V[t]
            last = delta + cfg["gamma"] * cfg["lam"] * nonterm * last
            adv[t] = last
        ret = adv + V[:T]
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        B = dict(obs=O, act=Ac, adv=adv_n, ret=ret, logp=LP, probs=PR)
        old_pol = [q.copy() for q in pol]
        ppo_update(pol, vf, opt_p, opt_v, B, cfg, beta, rng)
        if capture_iter is not None and it + 1 == capture_iter:
            SNAP.update(old=old_pol, new=[q.copy() for q in pol], B=B)
        _, _, kl, cf = surrogates(pol, O, Ac, adv_n, LP, PR, 0.2 if mode != "clip" else eps)
        if mode == "kl_adaptive":                        # paper's beta rule
            if kl < d_targ / 1.5: beta /= 2
            elif kl > d_targ * 1.5: beta *= 2
        it += 1
        recent = [r_ for s_, r_ in ep_log if s_ > gstep - T] or [np.nan]
        log["iter"].append(it); log["step"].append(gstep); log["ret"].append(float(np.mean(recent)))
        log["kl"].append(float(kl)); log["clipfrac"].append(float(cf)); log["beta"].append(float(beta))
        if verbose: print(f"it {it:3d} step {gstep:6d} ret {log['ret'][-1]:7.1f} kl {kl:.4f} clipfrac {cf:.2f} beta {beta:.3f}")
    last100 = [r_ for _, r_ in ep_log][-100:]
    log.update(setting=setting, eps=eps, seed=seed, steps=steps, score_last100=float(np.mean(last100)),
               episode_returns=[r_ for _, r_ in ep_log], episode_steps=[s_ for s_, _ in ep_log])
    return log

def random_baseline(n=200, seed=0):
    env = gym.make("CartPole-v1"); rng = np.random.default_rng(seed); out = []
    for i in range(n):
        env.reset(seed=seed + i); tot, done = 0.0, False
        while not done:
            _, r, te, tr, _ = env.step(int(rng.integers(2))); tot += r; done = te or tr
        out.append(tot)
    return float(np.mean(out))

SETTINGS = [("none", 0.0, "No clipping or penalty"), ("clip", 0.1, "Clipping, eps=0.1"),
            ("clip", 0.2, "Clipping, eps=0.2"), ("clip", 0.3, "Clipping, eps=0.3"),
            ("kl_adaptive", 0.2, "Adaptive KL, d_targ=0.01"), ("kl_fixed", 0.2, "Fixed KL, beta=1")]

def _job(args):
    setting, eps, seed, steps = args
    t0 = time.time(); out = run(setting, eps, seed, steps)
    fn = f"results/{setting}_{eps}_{seed}.json"
    json.dump(out, open(fn, "w"))
    print(f"done {setting:12s} eps={eps} seed={seed} score={out['score_last100']:.1f} ({time.time()-t0:.0f}s)", flush=True)
    return fn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "ablation"])
    ap.add_argument("--setting", default="clip"); ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=150000); ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "run":
        o = run(a.setting, a.eps, a.seed, a.steps, verbose=True); print("last-100-episode score:", o["score_last100"])
    else:
        from multiprocessing import Pool
        jobs = [(s, e, sd, a.steps) for sd in range(a.seeds) for s, e, _ in SETTINGS]
        with Pool(a.procs) as p: p.map(_job, jobs, chunksize=1)
