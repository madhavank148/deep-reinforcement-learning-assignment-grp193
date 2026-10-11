# Proximal Policy Optimization: A Small-Scale Reproduction

**Deep Reinforcement Learning Assignment, Group 193**  
**Problem V: Schulman et al., "Proximal Policy Optimization Algorithms"**  
Paper: <https://arxiv.org/abs/1707.06347>

## Abstract

This project implements Proximal Policy Optimization (PPO) from scratch using
NumPy and compares six update objectives on CartPole-v1. The comparison covers
the unclipped objective, three clipping ranges, adaptive KL control, and fixed
KL control. Ten random seeds are used for each setting, for 60 runs in total.
The trust-region variants are consistently strong: adaptive KL has the highest
mean final return (477.4 +/- 7.5), while clipping with epsilon 0.2 or 0.3 is a
close alternative (473.9 +/- 9.0 and 474.5 +/- 11.5).

## 1. Objective

PPO improves a policy while limiting how far the new policy moves from the old
one. This report studies the practical effect of that constraint by comparing
the following surrogate objectives:

- **No clipping / penalty:** the unconstrained CPI-style surrogate.
- **Clipped PPO:** the probability ratio is clipped to `[1-epsilon, 1+epsilon]`
  for epsilon 0.1, 0.2, and 0.3.
- **Adaptive KL:** a KL penalty whose coefficient is adjusted toward a target
  KL of 0.01.
- **Fixed KL:** a KL penalty with coefficient beta = 1.

The implementation includes generalized advantage estimation (GAE), multiple
minibatch epochs, and manual NumPy back-propagation. The environment is
CartPole-v1 and each run uses 61,440 environment steps.

## 2. Experimental Design

Each of the six settings is evaluated with ten seeds. Training logs are stored
in [`results/`](results/), one JSON file per setting and seed. The complete
workflow is available in [`PPO_Group193_Colab.ipynb`](PPO_Group193_Colab.ipynb).
The local reproduction commands are:

```bash
pip install numpy gymnasium matplotlib
python ppo_numpy.py ablation --seeds 10 --steps 61440
python make_plots.py
```

The plotting script regenerates all figures and [`table_results.csv`](table_results.csv)
from the JSON logs.

## 3. Results

The primary metric is the mean return over the final 100 episodes of each run.
The normalized score uses the random-policy score (22.7) as zero and the best
observed mean as one.

| Setting | Final return mean +/- std | Normalized | Mean return over training | Mean KL | Seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| No clipping / penalty | 276.1 +/- 96.8 | 0.56 | 365.6 | 0.6296 | 10 |
| Clip epsilon = 0.1 | 440.5 +/- 11.5 | 0.92 | 355.0 | 0.0020 | 10 |
| Clip epsilon = 0.2 | 473.9 +/- 9.0 | 0.99 | 402.9 | 0.0055 | 10 |
| Clip epsilon = 0.3 | 474.5 +/- 11.5 | 0.99 | 416.1 | 0.0106 | 10 |
| Adaptive KL, target = 0.01 | **477.4 +/- 7.5** | **1.00** | 406.2 | 0.0167 | 10 |
| Fixed KL, beta = 1 | 476.5 +/- 10.6 | 1.00 | 408.9 | 0.0045 | 10 |

### Interpretation

1. Trust-region control is important. Removing clipping and KL control produces
   a much larger average KL (0.6296) and a lower, more variable final return.
2. Epsilon 0.2 and 0.3 provide nearly saturated CartPole performance. The
   looser epsilon 0.3 also gives the highest mean return across all training
   steps among the clipping settings.
3. Adaptive KL obtains the best final mean and the smallest final-return
   standard deviation in this experiment. The difference from fixed KL and
   clipping is small, so it should not be treated as a broad superiority claim.

## 4. Figures

The generated figures are:

- [Figure A: clipped surrogate function](fig_A_clip_function.png)
- [Figure B: learning curves](fig_B_curves.png)
- [Figure C: mean policy-update KL](fig_C_kl.png)
- [Figure D: interpolation between old and new parameters](fig_D_interp.png)

Figure A shows why clipping removes the incentive to move farther in the
improving direction once the ratio leaves the permitted interval. Figure B
shows the mean and standard deviation across seeds. Figure C compares average
policy movement, and Figure D illustrates the clipped surrogate and KL as the
new parameters are interpolated beyond the learned update.

## 5. Limitations and Reproducibility Notes

This is a scaled-down reproduction of the comparison described around Section
6.1 of the PPO paper. It uses CartPole-v1 and 61k steps, not the paper's full
MuJoCo or Atari workloads. Results therefore demonstrate the behavior of the
implemented objectives on this task and budget only. Exact values can vary with
package versions and environment behavior; the committed JSON logs provide the
data used for the report figures and table.

## 6. Conclusion

On this CartPole ablation, PPO's clipped and KL-regularized objectives all
outperform the unconstrained objective. Adaptive KL is the best setting by the
reported final-return mean, but clipping with epsilon 0.2 or 0.3 achieves
almost the same performance with a simpler and widely used objective. The
results reinforce the central PPO design choice: constrain policy updates so
that optimization does not destabilize learning.