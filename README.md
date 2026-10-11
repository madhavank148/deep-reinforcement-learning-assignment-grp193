# PPO from scratch (NumPy) - Group 193
DRL Assignment, Problem V: Schulman et al., *Proximal Policy Optimization Algorithms* (arXiv:1707.06347).

- `ppo_numpy.py` : PPO (GAE, minibatch epochs, manual back-prop) with swappable objective: none / clip / adaptive KL / fixed KL
- `make_plots.py`: builds the figures and `table_results.csv` from `results/*.json`
- `PPO_Group193_Colab.ipynb`: runs everything end to end
- `results/` : the 60 run logs (6 settings x 10 seeds) behind the slides

Run: `pip install numpy gymnasium matplotlib` then `python ppo_numpy.py ablation --seeds 10 --steps 61440` then `python make_plots.py`.

Scope note: scaled-down check of paper Sec. 6.1 (CartPole-v1, 61k steps) - not the paper's MuJoCo / Atari experiments.
