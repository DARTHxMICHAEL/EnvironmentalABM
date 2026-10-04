# Environmental ABM

A grid-based agent-based model where agents forage, fight and reproduce. The model measures how sensitive the dynamics are to small perturbations, using a finite-time Lyapunov exponent, Shannon entropy and population regime statistics. Agents move either at random or with a small neural network that evolves through basic crossover and mutation.

## Model

- **World:** a `width × height` grid with walls (green), apples (red, $+5$ energy), oranges (orange, $+10$ energy) and agents (blue, with sex $\in \{0, 1\}$).
- **Each tick:** every living agent acts once, in random order, unless another agent removes it before its turn. Each agent pays a metabolic cost $c$ and dies when $E \le 0$. It then tries to move one cell in one of 8 directions. Walls and the grid edges block movement, and an agent only moves into a cell that is empty after the interaction (food eaten, fight loser removed). Children act from the next tick on.
- **Food:** stepping onto food adds its energy. Each tick, $\lfloor \rho \cdot N_{\text{empty}} \rfloor$ new food items spawn: 60% apples, 40% oranges.
- **Same-sex encounter (fight):** the agent with more energy wins and takes the loser's energy, $E_w \leftarrow E_w + E_l$. The loser is removed.
- **Opposite-sex encounter (mating):** each parent pays the reproduction cost $c_r$. The remaining energy is split evenly among the children, who are placed on free cells within radius 2 of the parents' midpoint. The parents are removed:

$$
n = \min\left(n_{\text{free}},\ \left\lfloor \frac{E_1 + E_2 - 2c_r}{E_{\min}} \right\rfloor\right), \qquad E_{\text{child}} = \frac{E_1 + E_2 - 2c_r}{n}
$$

  If no child fits ($n = 0$, from too little energy or no free cells), the mating fails. The parents stay in place with $E_i - c_r$, and a parent with $E_i - c_r \le 0$ dies.

**Energy budget.** Fights and successful matings only move energy between agents, so each tick's change in total agent energy is exact:

$$
\Delta E_{\text{tot}} = F - c \cdot A - c_r \cdot P - R
$$

where $F$ is food eaten, $A$ the number of agents that acted, $P$ the number of parents that mated (including failed matings), and $R \le 0$ the leftover energy of agents removed after running out. `Grid.stats` records every term for the last tick.

## Neural agents (`use_nn=True`)

- **Vision:** 8 rays, each with range 8. Each ray returns $[R, G, B, r/8]$ for the first object it hits, giving a 32-value input vector.
- **Network:** $32 \to 16 \to 8$ with
  $$\mathbf{y} = W_2 \tanh(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2, \qquad \text{move} = \arg\max_k y_k$$
- **Inheritance:** each weight comes from either parent with probability 0.5 (uniform crossover). Then each weight mutates with probability 0.1: $w \leftarrow w + \mathcal{N}(0, 0.05^2)$.

## Analysis

Each trial builds two identical grids from the same seed, then perturbs $k$ agents in the second grid by adding $E_{\min}$ to their energy. Both grids then run in lockstep.

**Phase-space distance** (normalized over all cells):

$$
d(t) = \frac{1}{WH} \sum_{i,j} \delta_{ij}, \qquad
\delta_{ij} =
\begin{cases}
1 & \text{cell types differ} \\
\tfrac12 \mathbb{1}[s_1 \ne s_2] + \tfrac12 \dfrac{|E_1 - E_2|}{\max(|E_1|, |E_2|, 1)} & \text{both cells hold agents} \\
0 & \text{otherwise}
\end{cases}
$$

**Lyapunov exponent:** a linear fit over the first $\max(10,\ \text{cutoff} \cdot T)$ ticks:

$$
d(t) \approx d_0 e^{\lambda t} \quad\Rightarrow\quad \log d(t) \approx \log d_0 + \lambda t
$$

**Shannon entropy:** computed over coarse-grained cell states (empty, wall, apple, orange, and agent × sex × high/low energy). The model reports $\Delta H = |H_2 - H_1|$ at the final tick.

$$
H = -\sum_i p_i \log_2 p_i
$$

**Regime metrics**, computed on grid 1 after the first 30% of ticks (burn-in):
- growth rate $r$, the slope of $\log N(t)$
- coefficient of variation $\mathrm{CV} = \sigma_N / \mu_N$
- viability: both sexes are present at the end
- preservation: the final population is at least the initial population

Results are averaged over `num_runs` trials. Before the trials start, a determinism check confirms that two runs from the same seed give identical grids.

## Usage

```bash
sudo apt install python3.12-venv     # Ubuntu/Debian only, if venv is missing
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest -v                # run the tests
python environmental_abm.py          # run the experiments
```

Tested with Python 3.12, numpy 2.5.3 and matplotlib 3.11.2.

The script runs three experiments one after another:
1. Random agents in the near-critical regime
2. Neural agents in the near-critical regime
3. Random agents under the same constraints as experiment 2

Settings are at the bottom of the file: `grid_params`, `num_runs`, `num_ticks`, `num_prtrb_agents`, `init_seed` and `cutoff`.

## Viewing plots

**Jupyter (recommended).** Plots appear in the browser, under the cell that made them.

```bash
pip install notebook
jupyter notebook --ServerApp.use_redirect_file=False
```

If no browser opens, copy the `http://localhost:8888/tree?token=...` link from the terminal. In a new notebook, run all experiments with:

```python
%run environmental_abm.py
```

Or run a single setup. With `final_render=False, lyapunov_final_render=False`, you get only the divergence plot for each run:

```python
from environmental_abm import main_simulation

main_simulation(
    num_runs=5, num_ticks=2000, num_prtrb_agents=2, init_seed=123, cutoff=0.015,
    final_render=False, lyapunov_final_render=False,
    width=100, height=100, metabolic_cost=0.9, min_child_energy=7, reproduction_cost=8,
    food_respawn_rate=0.012, num_agents=80, num_apples=900, num_oranges=900,
    num_walls=60, use_nn=False,
)
```

**Plot windows.** Running `python environmental_abm.py` opens each plot in its own window. This needs Tk: install it with `sudo apt install python3-tk` on Ubuntu/Debian. Without Tk, matplotlib shows nothing.
