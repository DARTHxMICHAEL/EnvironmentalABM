"""
Probe: how well do agents forage at a given food level?

Frozen networks from snapshots (see environmental_abm.Snapshots) are placed in an arena where the
food stock is held fixed: eaten food is replaced on random empty cells every tick. Agents never
starve, fight or mate - when two meet, nothing happens - so only foraging is measured. Each run
gives the intake per agent and tick at each food level (the functional response), and R*, the
food level at which intake covers the cost per agent and tick.

Random agents are probed as well. Their R* is known from the full model (about 2480 food items in
world B, K = 5000), so they check whether the arena reproduces what the full model does.

	python probe.py <results folder> [...]   # probe every snapshots.npz found, then summarize
	python probe.py --summary <probe folder>  # summarize a finished probe again

Options (environment): PROBE_GRIDS grids per condition and tick (default 10), PROBE_AGENTS agents
per arena (300), PROBE_TICKS measured ticks per food level (150), PROBE_SNAPSHOT_TICKS which
snapshots to probe (default 0,2500,5000,10000,20000,30000; the sweep saves every 5000 up to 150000).
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool

import numpy as np

from environmental_abm import Agent, Apple, Grid, Orange, SimpleNN

FOOD_LEVELS = [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6500, 8000]
SNAPSHOT_TICKS = [int(t) for t in os.environ.get("PROBE_SNAPSHOT_TICKS", "0,2500,5000,10000,20000,30000").split(",")]
GRIDS = int(os.environ.get("PROBE_GRIDS", 10))
AGENTS = int(os.environ.get("PROBE_AGENTS", 300))
TICKS = int(os.environ.get("PROBE_TICKS", 150))
WARMUP = 50
RANDOM_REPEATS = 10
# energy cost per agent and tick that intake must cover: metabolism 0.9 plus the reproduction
# costs, about 1.0 to 1.06 in the full runs (see the energy budget in ReadMe)
COST = 1.0

ARENA = dict(
	width=100, height=100, metabolic_cost=0.9, min_child_energy=7, reproduction_cost=2,
	food_respawn_rate=0.0, num_agents=0, num_apples=0, num_oranges=0, num_walls=60,
)


class ProbeGrid(Grid):
	"""A Grid where encounters do nothing: agents only block each other."""

	def meet(self, agent1, agent2):
		pass

	def top_up(self, level):
		missing = level - len(self.food_items)
		if missing <= 0:
			return
		empty = self.get_empty_positions()
		for x, y in self.rng.sample(empty, min(missing, len(empty))):
			food = Apple(x, y) if self.rng.random() < 0.6 else Orange(x, y)
			self.place_object(food)
			self.food_items.append(food)


def arena(networks, food_level, move_noise=0.0, seed=0, agents=AGENTS, ticks=TICKS, warmup=WARMUP):
	"""
	Intake of `agents` agents at a fixed food level. networks: flat SimpleNN vectors, used in turn
	(None for random agents). Returns per agent and tick: food items eaten, food energy, blocked moves.
	"""
	use_nn = networks is not None
	g = ProbeGrid(**ARENA, use_nn=use_nn, move_noise=move_noise, seed=seed)
	free = g.rng.sample(g.get_empty_positions(), agents)
	for i, (x, y) in enumerate(free):
		sex = g.rng.randint(0, 1)
		agent = Agent(x, y, sex, (0, 0, 1) if sex == 0 else (0, 0, 0.5), energy=1e12)  # never starves
		if use_nn:
			agent.nn = SimpleNN.from_flat(networks[i % len(networks)], g.rng)
		g.place_object(agent)
		g.agents.append(agent)
	g.top_up(food_level)

	totals = defaultdict(float)
	for t in range(warmup + ticks):
		g.move_agent()
		if t >= warmup:
			for key in ("acted", "eaten", "food", "blocked"):
				totals[key] += g.stats[key]
		g.top_up(food_level)
	acted = totals["acted"]
	return totals["eaten"] / acted, totals["food"] / acted, totals["blocked"] / acted


def r_star(levels, intake, cost=COST):
	"""Lowest food level where intake reaches cost (linear interpolation), nan if it never does."""
	for i in range(1, len(levels)):
		if intake[i - 1] < cost <= intake[i]:
			return levels[i - 1] + (cost - intake[i - 1]) * (levels[i] - levels[i - 1]) / (intake[i] - intake[i - 1])
	return float(levels[0]) if intake[0] >= cost else float("nan")


def find_jobs(roots):
	"""One job per (condition, run, grid, snapshot tick), from every snapshots.npz under roots."""
	jobs, counts = [], defaultdict(int)
	for root in roots:
		for folder in sorted(os.listdir(root)):
			path = os.path.join(root, folder)
			if not os.path.exists(os.path.join(path, "snapshots.npz")):
				continue
			with open(os.path.join(path, "params.json")) as f:
				params = json.load(f)
			condition = params["name"].split("__")[0]
			s = np.load(os.path.join(path, "snapshots.npz"))
			meta = s["snap_meta"]
			for row in np.flatnonzero(meta[:, 3] == 0):  # periodic snapshots
				run, grid, tick = (int(v) for v in meta[row, :3])
				if tick not in SNAPSHOT_TICKS or counts[(condition, tick)] >= GRIDS:
					continue
				nets = s["agent_network"][s["agent_data"][:, 0] == row]
				nets = s["networks"][nets[nets >= 0]]
				if len(nets) < 20:  # too few agents left to say much
					continue
				counts[(condition, tick)] += 1
				jobs.append(dict(condition=condition, source=path, run=run, grid=grid, tick=tick,
								 population=int(meta[row, 4]), move_noise=params.get("move_noise", 0.0), networks=nets))
	for repeat in range(RANDOM_REPEATS):
		jobs.append(dict(condition="random", source="", run=repeat + 1, grid=0, tick=0, population=0,
						 move_noise=0.0, networks=None))
	return jobs


def work(args):
	index, job = args
	rows = []
	for level in FOOD_LEVELS:
		eaten, energy, blocked = arena(job["networks"], level, job["move_noise"], seed=10_000 + index)
		rows.append({k: v for k, v in job.items() if k != "networks"} | dict(
			food_level=level, eaten=eaten, intake=energy, blocked=blocked))
	return rows


def summary(folder):
	with open(os.path.join(folder, "probe.csv")) as f:
		rows = list(csv.DictReader(f))
	curves = defaultdict(dict)  # (condition, tick) -> {(source, run, grid): {level: row}}
	for r in rows:
		curves[(r["condition"], int(r["tick"]))].setdefault((r["source"], r["run"], r["grid"]), {})[int(r["food_level"])] = r

	shown = [500, 1000, 2000, 2500, 3000, 5000]
	lines = [f"Probe - {folder}  ({AGENTS} agents, cost {COST} per agent-tick, medians over grids)\n",
			 f"{'condition':<20} {'tick':>6} {'grids':>5} | intake per agent-tick at food level"
			 + " " * 14 + f"| {'R*':>13} | blocked at 5000",
			 f"{'':<20} {'':>6} {'':>5} | " + " ".join(f"{lvl:>6}" for lvl in shown) + f" | {'median (IQR)':>13} |"]
	order = sorted(curves, key=lambda k: (k[0] != "random", k[0], k[1]))
	for key in order:
		grids = curves[key].values()
		intake = np.array([[float(g[lvl]["intake"]) for lvl in FOOD_LEVELS] for g in grids])
		stars = np.array([r_star(FOOD_LEVELS, curve) for curve in intake])
		blocked = np.median([float(g[5000]["blocked"]) for g in grids])
		med = np.median(intake, axis=0)
		at = {lvl: med[FOOD_LEVELS.index(lvl)] for lvl in shown}
		ok = stars[np.isfinite(stars)]
		star = f"{np.median(ok):5.0f} ({np.percentile(ok, 25):.0f}-{np.percentile(ok, 75):.0f})" if len(ok) else "never"
		lines.append(f"{key[0]:<20} {key[1]:>6} {len(intake):>5} | " + " ".join(f"{at[lvl]:>6.3f}" for lvl in shown)
					 + f" | {star:>13} | {blocked:.3f}")
	text = "\n".join(lines)
	print(text)
	with open(os.path.join(folder, "summary.txt"), "w") as f:
		f.write(text + "\n")


if __name__ == "__main__":
	if len(sys.argv) == 3 and sys.argv[1] == "--summary":
		summary(sys.argv[2])
		sys.exit()

	jobs = find_jobs(sys.argv[1:])
	folder = os.path.join("results", f"probe_{datetime.now():%Y-%m-%d_%H%M%S}")
	os.makedirs(folder)
	print(f"probe: {len(jobs)} arenas x {len(FOOD_LEVELS)} food levels -> {folder}", flush=True)
	with Pool(os.cpu_count()) as pool:
		results = pool.map(work, list(enumerate(jobs)), chunksize=1)
	rows = [row for job_rows in results for row in job_rows]
	with open(os.path.join(folder, "probe.csv"), "w", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=list(rows[0]))
		writer.writeheader()
		writer.writerows(rows)
	summary(folder)
