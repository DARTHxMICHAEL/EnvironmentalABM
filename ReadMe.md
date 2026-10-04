# Environmental ABM

A grid-based agent-based model where agents forage, fight and reproduce. It continues the [thesis version](https://github.com/DARTHxMICHAEL/EvolutionaryABM) with the model bugs fixed (see [Changes from the thesis version](#changes-from-the-thesis-version)). The model measures how sensitive the dynamics are to small perturbations, using a finite-time Lyapunov exponent, Shannon entropy and population regime statistics. Agents move either at random or with a small neural network that evolves through basic crossover and mutation.

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
pip install numpy matplotlib
python environmental_abm.py           # run the experiments
python -m unittest -v                # run the tests
```

The script runs three experiments one after another:
1. Random agents in the near-critical regime
2. Neural agents in the near-critical regime
3. Random agents under the same constraints as experiment 2

Settings are at the bottom of the file: `grid_params`, `num_runs`, `num_ticks`, `num_prtrb_agents`, `init_seed` and `cutoff`. The defaults (20 runs × 15,000 ticks each) open many plot windows. The parameter sets were tuned with the thesis version and need recalibrating for the fixed model.

## Changes from the thesis version

The thesis results come from [EvolutionaryABM](https://github.com/DARTHxMICHAEL/EvolutionaryABM), which had these bugs:

1. **Most agents skipped most ticks.** The tick loop stopped (`return`) at the first agent that another agent had already removed in that tick. In dense populations only about 1–5% of agents moved and paid metabolism each tick. The saturated grids and the energy levels reported in the thesis come mainly from this bug.
2. **Agents vanished from the grid.** After a failed mating, the moving agent stepped onto its partner's cell. The partner stayed in the agent list but disappeared from the grid until it moved again, so population counts and grid metrics (entropy, $d(t)$) disagreed.
3. **The reproduction cost did not match the description.** Children split the parents' energy from *before* the cost. The thesis (Section 3.3) describes the energy *after* the cost, which is what the model does now.
4. **Agents could live with negative energy.** Parents could survive a failed mating with $E \le 0$, and a fight against such an agent reduced the winner's energy.

The tests in `test_environmental_abm.py` check the properties these bugs broke: every agent acts exactly once per tick, the agent and food lists match the grid, no living agent has $E \le 0$, the energy budget balances every tick, and the mating rules hold.
