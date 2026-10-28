# Environmental ABM

A grid-based agent-based model where agents forage, fight and reproduce. The model measures how sensitive the dynamics are to small perturbations, using a finite-time Lyapunov exponent, Shannon entropy and population regime statistics. Agents move either at random or with a small neural network that evolves through basic crossover and mutation.

## Model

- **World:** a `width × height` grid with walls (green), apples (red, $+5$ energy), oranges (orange, $+10$ energy) and agents (blue, with sex $\in \{0, 1\}$).
- **Each tick:** every living agent acts once, in random order, unless another agent removes it before its turn. Each agent pays a metabolic cost $c$ and dies when $E \le 0$. It then tries to move one cell in one of 8 directions. Walls and the grid edges block movement, and an agent only moves into a cell that is empty after the interaction (food eaten, fight loser removed). Children act from the next tick on.
- **Food:** stepping onto food adds its energy. Each tick, new food items spawn on random empty cells: 60% apples, 40% oranges. How many depends on `regrowth`:
  - `"empty"` (default): $\lfloor \rho \cdot N_{\text{empty}} \rfloor$. Eating food frees cells, so grazing the stock down *raises* supply.
  - `"logistic"`: $\rho F (1 - F/K)$, with $F$ the food items on the grid and $K$ = `food_capacity`, rounded stochastically. Grazing below $K/2$ *lowers* supply, and $F = 0$ is permanent.
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
- **Inheritance:** each weight comes from either parent with probability 0.5 (uniform crossover). Then each weight mutates with probability 0.1: $w \leftarrow w + \mathcal{N}(0, 0.05^2)$. Mutations draw from the grid's RNG, so siblings mutate independently. `inheritance="no_mutation"` keeps only the crossover, and `inheritance="none"` gives every child a fresh random network, so nothing can adapt.
- **Move noise:** with probability `move_noise` (default 0), a neural agent makes a random move instead of its network's choice. A network is deterministic and has no memory, so in a world that does not change it can lock into a loop, such as pushing against the border; noise breaks such loops.

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

**Divergence timing.** Each grid draws all of its randomness (move order, random moves, food spawns, fight ties) from a single RNG. The perturbation stays latent, only visible in the perturbed agents' energy, until the first event it changes: a fight, a death, or a different number of children. From that tick on, the two grids consume random numbers differently, food spawns and moves differ across the whole grid, and $d(t)$ jumps by orders of magnitude in one tick. It then rises to its saturation level instead of growing exponentially. Each run reports:
- the *desync tick*, the first tick at which at least 10 cells differ
- the *half-saturation tick* $t_{1/2}$, the first tick with $d(t) \ge d_\infty / 2$, where $d_\infty$ is the baseline pair's mean distance after burn-in
- the finite-time Lyapunov exponent over the rise only:

$$
\lambda = \frac{1}{t_{1/2}} \log \frac{d(t_{1/2})}{d(0)}
$$

  Because of the jump, $\lambda$ mostly reflects how long the perturbation stays latent, not an exponential growth rate.

**Baseline (null model).** With `baseline=True`, each run also simulates a third grid. It starts identical to grid 1 but draws from a different RNG stream. Its distance to grid 1, $d_{13}(t)$, is what two runs that share only their initial state look like. If the perturbed pair ends as far apart as the baseline pair, and grid 1 / grid 2 outcomes split about as often as independent fates would, then the perturbation has the same effect as re-seeding.

**Shannon entropy:** computed over coarse-grained cell states (empty, wall, apple, orange, and agent × sex × high/low energy). The model reports $\Delta H = |H_2 - H_1|$ at the final tick.

$$
H = -\sum_i p_i \log_2 p_i
$$

**Regime metrics**, computed on grid 1 after the first 30% of ticks (burn-in):
- growth rate $r$, the slope of $\log N(t)$
- coefficient of variation $\mathrm{CV} = \sigma_N / \mu_N$
- viability: both sexes are present at the end
- preservation: the final population is at least the initial population

**Population summary**, for both grids. Each grid's outcome is decided at the final tick:
- *established*: $N \ge N_0$ (the same as preservation)
- *declined*: $0 < N < N_0$
- *extinct*: $N = 0$, reported with the extinction tick

The summary counts outcomes per grid and how often grid 1 and grid 2 (and grid 1 and the baseline) end with the same outcome. It also prints the split rate expected if the two grids' fates were independent: $1 - \sum_o p_1(o)\, p_2(o)$, with $p_i(o)$ the share of runs where grid $i$ ends with outcome $o$. Averages of population, total energy $E$ and energy per agent $E/N$ (over the same window as $r$ and CV) pool the established grids of both grids. A table lists every run and grid with final and peak $N$ and $E$.

**Event rates**, per agent and tick over the same window: food energy eaten, fights, matings, failed matings, births and starvation deaths. Each rate is total events divided by total agent-ticks. Fights, matings (2 parents each) and starvation are the only ways agents die, so they show where the population's turnover comes from.

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

Tested with Python 3.12, numpy 2.5.3, matplotlib 3.11.2 and notebook 7.6.3.

The script runs three experiments one after another:
1. Random agents in the near-critical regime
2. Neural agents in the near-critical regime
3. Random agents under the same constraints as experiment 2

Settings are at the bottom of the file: `grid_params`, `num_runs`, `num_ticks`, `num_prtrb_agents`, `init_seed` and `baseline`. The baseline grid adds about 50% to the run time; `baseline=False` skips it.

Each experiment saves its output to `results/<timestamp>_<name>/`:

| | |
|---|---|
| `summary.txt` | everything printed: parameters, checks and summaries |
| `params.json` | experiment parameters |
| `runs.csv` | per-run, per-grid stats: outcome, λ, desync and half-saturation ticks, late distances, ΔH, population, energy, event rates, baseline outcome |
| `timeseries.npz` | per tick (index = tick): `d`; for both grids, shape (runs, 2, ticks + 1): `population`, `energy`, `sex1` (agents of sex 1), `food_items` and the event counters `acted`, `starved`, `fights`, `matings`, `failed_matings`, `births`, `food`, `metabolism`, `reproduction`, `residual`; with the baseline also `baseline_d` and `baseline_population` |
| `snapshots.npz` | only with `snapshot_every`: agent positions, sex, energy and networks (up to 100 per grid) every `snapshot_every` ticks, plus all agents every 100 ticks while an established population is below 20 agents (see `Snapshots`) |
| `plots/` | $d(t)$ plot for each run (perturbed and baseline), with both populations |
| `states/` | initial and final grid snapshots of every run, always saved |

Pass `results_dir=None` to `main_simulation` to save nothing.

### Regrowth pilot

`pilot_regrowth.py` asks whether adaptation helps or hurts the population depending on how food regrows. It compares world A (`regrowth="empty"`) with world B (`regrowth="logistic"`, $K = 5000$, $\rho = 0.065$). The two are calibrated so that random agents settle at the same population (about 550) and food stock (about 2480) in both. In world B, that stock is the logistic peak. Both worlds start pre-stocked with 8000 food items, so the start-up phase is the same. Conditions: random agents, evolving networks, crossover only (B only) and no inheritance. Conditions are named `<world>_<agents>[_K<capacity>]`. Two batches are defined:
- `pilot`: random, evolving and no-inheritance agents in A and B, plus crossover-only in B (seeds 123–142)
- `snapshots`: evolving and crossover-only agents in A and B with agent snapshots (seeds 143–162), plus a K sweep in world B for agents that do not adapt (random and no inheritance)
- `noise`: the same with `move_noise` of 1% and 5% (suffix `_n1`, `_n5`), same seeds as `snapshots`

- `sweep`: the main K sweep, about 750 CPU hours and 11 GB (see `--dry-run`). Evolving agents with 5% noise over K = 2000–10000 for 150k ticks, with snapshots every 5000 ticks; world A for 150k ticks as the stable reference; crossover-only (4 K values, 150k ticks); evolving agents without noise (5 K values, 50k ticks); and agents that do not adapt: no-inheritance with noise over the same K, random agents over K = 2000–15000 (30k ticks). 20 fresh seeds each (301–320), in jobs of 2 seeds.

Each of the first three batches takes about 6 h on 6 cores / 12 threads, and every batch writes `report.txt`. For long batches, `--dry-run` lists the jobs with rough CPU hours and storage, `--resume <folder>` continues a stopped batch (finished jobs are skipped), a failing job is logged to `errors.log` instead of stopping the batch, and `PILOT_WORKERS` sets the number of parallel processes. The report merges several folders, for example the pilot's K = 5000 runs with the sweep:

```bash
nohup python pilot_regrowth.py snapshots > snapshots.log 2>&1 &
python pilot_regrowth.py sweep --dry-run
nohup python pilot_regrowth.py sweep > sweep.log 2>&1 &
nohup python pilot_regrowth.py sweep --resume results/sweep_<timestamp> >> sweep.log 2>&1 &
python pilot_regrowth.py --report results/pilot_<timestamp> results/snapshots_<timestamp>
```

### Foraging probe

`probe.py` measures how well agents forage at a given food level, which is their functional response. It takes frozen networks from `snapshots.npz` and places them in an arena where the food stock is held fixed: eaten food is replaced on random empty cells every tick. Agents never starve, fight or mate, so only foraging is measured. The result is the intake per agent and tick at each food level, and $R^*$, the food level at which intake covers the cost per agent and tick (1.0 by default: metabolism 0.9 plus reproduction costs). Random agents are probed as a calibration: their arena intake at $F \approx 2486$ matches the full model within about 6%.

```bash
python probe.py results/snapshots_<timestamp> results/noise_<timestamp>   # writes results/probe_<timestamp>/
python probe.py --summary results/probe_<timestamp>
```

## Viewing plots

**Jupyter (recommended).** Plots appear in the browser, under the cell that made them. With the venv active:

```bash
source .venv/bin/activate # if not used before
jupyter notebook --ServerApp.use_redirect_file=False
```

If no browser opens, copy the `http://localhost:8888/tree?token=...` link from the terminal. In a new notebook, run all experiments with:

```python
%run environmental_abm.py
```

Or run a single setup. With `final_render=False, lyapunov_final_render=False`, only the divergence plots are displayed. The grid snapshots are still saved to `states/`:

```python
from environmental_abm import main_simulation

main_simulation(
    num_runs=10, num_ticks=4000, num_prtrb_agents=2, init_seed=123, baseline=True,
    final_render=False, lyapunov_final_render=False,
    width=100, height=100, metabolic_cost=0.9, min_child_energy=7, reproduction_cost=2,
    food_respawn_rate=0.012, num_agents=120, num_apples=900, num_oranges=900,
    num_walls=60, use_nn=False,
)
```

**Plot windows.** Running `python environmental_abm.py` opens each plot in its own window. This needs Tk: install it with `sudo apt install python3-tk` on Ubuntu/Debian. Without Tk, no windows open, but the plots are still saved to the output folder.
