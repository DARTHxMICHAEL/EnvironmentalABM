"""
Regrowth experiments: does adaptation help or hurt the population, depending on how food regrows?

World A - food spawns in proportion to empty cells: grazing the stock down raises supply.
World B - logistic regrowth r * F * (1 - F / K): grazing below K / 2 lowers supply.
K and r are calibrated so random agents settle at the same N and food stock in both worlds
(N ≈ 550, F ≈ 2480, which is the logistic peak in B). Both worlds start pre-stocked with food,
so the start-up phase, when the initial networks barely forage, is the same in both.

Agents: random, neural with evolving networks, neural with crossover only, and neural with no
inheritance (every child gets a fresh random network, so nothing adapts).

Conditions are named <world>_<agents>[_K<capacity>][_n<move noise in %>], e.g. B_nn_evolve,
B_random_K7000 or B_nn_evolve_n5.

Batches:
	pilot     - first go/no-go run: random, evolve and none in A and B, crossover-only in B
	snapshots - evolve and crossover-only in A and B with agent snapshots (see Snapshots), plus
	            the K sweep for agents that do not adapt (random, none) in B
	noise     - neural agents that make a random move with probability 1% or 5% (move_noise): do
	            populations still collapse when agents cannot get stuck? Same seeds as snapshots

	python pilot_regrowth.py [batch]                 # run a batch in parallel, then report
	python pilot_regrowth.py --report <folder> ...   # report on finished batches (merged)
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # worker processes must never open plot windows

import contextlib
import csv
import json
import re
import sys
from datetime import datetime
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from environmental_abm import main_simulation

# PILOT_TICKS / PILOT_RUNS override these for quick tests
TICKS = int(os.environ.get("PILOT_TICKS", 30000))
RUNS_PER_JOB = int(os.environ.get("PILOT_RUNS", 10))
# first seed of each batch: the snapshots batch continues after the pilot's 20 seeds
SEEDS = {"pilot": 123, "snapshots": 143, "noise": 143}

COMMON = dict(
	width=100, height=100, metabolic_cost=0.9, min_child_energy=7, reproduction_cost=2,
	num_agents=120, num_apples=4000, num_oranges=4000, num_walls=60,
)
WORLDS = {
	"A": dict(regrowth="empty", food_respawn_rate=0.012),
	"B": dict(regrowth="logistic", food_respawn_rate=0.065, food_capacity=5000),
}
AGENTS = {
	"random": dict(use_nn=False),
	"nn_evolve": dict(use_nn=True, inheritance="evolve"),
	"nn_no_mutation": dict(use_nn=True, inheritance="no_mutation"),
	"nn_none": dict(use_nn=True, inheritance="none"),
}

# agents that do not adapt settle within a few thousand ticks
SWEEP = dict(ticks=TICKS // 2)
# (condition, number of RUNS_PER_JOB-run jobs, options)
BATCHES = {
	"pilot": [
		("A_random", 2, {}), ("B_random", 2, {}),
		("A_nn_evolve", 2, {}), ("B_nn_evolve", 2, {}),
		("A_nn_none", 1, {}), ("B_nn_none", 1, {}),
		("B_nn_no_mutation", 2, {}),
	],
	"snapshots": [
		("A_nn_evolve", 2, dict(snapshot_every=2500)), ("A_nn_no_mutation", 2, dict(snapshot_every=2500)),
		("B_nn_evolve", 2, dict(snapshot_every=2500)), ("B_nn_no_mutation", 2, dict(snapshot_every=2500)),
		*[(f"B_nn_none_K{k}", 1, SWEEP) for k in (3000, 4000, 7000, 10000)],
		*[(f"B_random_K{k}", 1, SWEEP) for k in (3000, 4000, 6000, 7000, 8000, 10000)],
	],
	"noise": [
		("B_nn_evolve_n5", 2, dict(snapshot_every=2500)), ("B_nn_no_mutation_n5", 2, {}),
		("B_nn_evolve_n1", 2, {}), ("B_nn_no_mutation_n1", 2, {}),
		("A_nn_evolve_n5", 2, {}), ("B_nn_none_n5", 1, {}),
	],
}

WINDOWS = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 10000), (10000, 15000),
		   (15000, 20000), (20000, 25000), (25000, 30001)]


def params(condition):
	match = re.fullmatch(r"([AB])_(.+?)(?:_K(\d+))?(?:_n(\d+))?", condition)
	world, agents, capacity, noise = match.groups()
	p = {**COMMON, **WORLDS[world], **AGENTS[agents]}
	if capacity:
		p["food_capacity"] = int(capacity)
	if noise:
		p["move_noise"] = int(noise) / 100
	return p


def work(job):
	condition, seed, options, root = job
	# under %run in Jupyter matplotlib is already set to the inline backend, and forked workers
	# must not push plots into the notebook - plots are only saved
	plt.switch_backend("Agg")
	with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
		results = main_simulation(
			num_runs=RUNS_PER_JOB, num_ticks=options.get("ticks", TICKS), num_prtrb_agents=2, init_seed=seed,
			baseline=False, snapshot_every=options.get("snapshot_every"),
			name=f"{condition}__s{seed}", results_dir=root, **params(condition)
		)
	print(f"[{datetime.now():%H:%M}] done: {condition} seeds {seed}-{seed + RUNS_PER_JOB - 1}", flush=True)
	return results.dir


def load(roots):
	"""Merge the jobs of each condition: {condition: {"rows": [...], "series": {key: (grids, ticks + 1)}}}"""
	data = {}
	for root in roots:
		for folder in sorted(os.listdir(root)):
			path = os.path.join(root, folder)
			if not os.path.exists(os.path.join(path, "timeseries.npz")):
				continue
			with open(os.path.join(path, "params.json")) as f:
				condition = json.load(f)["name"].split("__")[0]
			with open(os.path.join(path, "runs.csv")) as f:
				rows = list(csv.DictReader(f))
			s = np.load(os.path.join(path, "timeseries.npz"))
			entry = data.setdefault(condition, {"rows": [], "series": {}})
			if entry["series"] and entry["series"]["population"].shape[1] != s["population"].shape[-1]:
				print(f"skipped {path}: run length differs from earlier {condition} runs", file=sys.stderr)
				continue
			entry["rows"] += rows
			for key in ("population", "sex1", "food_items", "fights", "matings", "acted"):
				grids = s[key].reshape(-1, s[key].shape[-1])  # (runs * 2 grids, ticks + 1), same order as rows
				entry["series"][key] = np.concatenate([entry["series"][key], grids]) if key in entry["series"] else grids
	return data


def capacity(condition):
	return params(condition).get("food_capacity")


def report(roots):
	data = load(roots)
	lines = []
	log = lines.append
	log(f"Regrowth report - {', '.join(roots)}\n")

	for condition in sorted(data, key=lambda c: (c[0], capacity(c) or 0, c)):
		d = data[condition]
		P, F = d["series"]["population"], d["series"]["food_items"]
		n0 = COMMON["num_agents"]
		established_once = (P > 2 * n0).any(axis=1)
		extinct = P[:, -1] == 0
		ext_tick = np.where(extinct, np.argmax(P == 0, axis=1), -1)
		late = extinct & established_once
		# both sexes left at the end - a lone survivor or a single-sex remnant cannot reproduce
		viable = (d["series"]["sex1"][:, -1] > 0) & (d["series"]["sex1"][:, -1] < P[:, -1])

		log(f"=== {condition}  ({len(P)} grids, pairs share initial networks)")
		log(f"  alive at end {np.sum(~extinct)}/{len(P)}, viable (both sexes) {np.sum(viable)}"
			f" | early extinctions (never > {2 * n0}) {np.sum(extinct & ~established_once)}"
			f" | LATE extinctions (after establishing) {np.sum(late)}"
			+ (f", ticks {sorted(ext_tick[late].tolist())}" if late.any() else ""))
		log(f"  {'window':>13} {'alive':>6} {'N mean':>8} {'N sd':>6} {'CV N':>5} {'food':>7} {'encounters/agent':>17}")
		for a, b in WINDOWS:
			if b - P.shape[1] > (b - a) // 2:  # less than half of the window was simulated
				break
			alive = P[:, min(b, P.shape[1]) - 1] > 0
			if not alive.any():
				log(f"  {a:>6}-{b:<6} {0:>6}")
				continue
			n = P[alive, a:b]
			cv = np.mean(n.std(axis=1) / np.maximum(n.mean(axis=1), 1e-9))  # within-grid swings
			enc = (d["series"]["fights"][alive, a:b].sum() + d["series"]["matings"][alive, a:b].sum()) \
				/ max(d["series"]["acted"][alive, a:b].sum(), 1)
			log(f"  {a:>6}-{b:<6} {alive.sum():>6} {n.mean():>8.1f} {n.mean(axis=1).std():>6.1f} {cv:>5.2f} "
				f"{F[alive, a:b].mean():>7.0f} {enc:>17.4f}")
		log("")

	# one line per condition: how many populations made it, and how they died
	log("=== Survival overview")
	log(f"  {'condition':<22} {'grids':>5} {'viable':>6} {'start-up deaths':>15} {'late deaths':>11} {'CV N, last 5000':>16}")
	for condition in sorted(data):
		P, sex1 = data[condition]["series"]["population"], data[condition]["series"]["sex1"]
		viable = (sex1[:, -1] > 0) & (sex1[:, -1] < P[:, -1])
		extinct = P[:, -1] == 0
		established_once = (P > 2 * COMMON["num_agents"]).any(axis=1)
		late = P[viable, -5000:]
		cv = np.mean(late.std(axis=1) / late.mean(axis=1)) if viable.any() else float("nan")
		log(f"  {condition:<22} {len(P):>5} {viable.sum():>6} {np.sum(extinct & ~established_once):>15} "
			f"{np.sum(extinct & established_once):>11} {cv:>16.2f}")
	log("")

	# evolving populations relative to random agents in each world (at the pilot's K)
	log("=== Adaptation summary (last 5000 ticks, grids alive at the end)")
	for world in "AB":
		late_n = {}
		for agents in ("random", "nn_evolve", "nn_no_mutation", "nn_none"):
			c = f"{world}_{agents}"
			if c in data:
				P = data[c]["series"]["population"]
				alive = P[:, -1] > 0
				late_n[agents] = P[alive, -5000:].mean() if alive.any() else 0.0
		if late_n:
			ratio = f"N evolve / N random = {late_n['nn_evolve'] / late_n['random']:.2f}  " \
				if late_n.get("random") and "nn_evolve" in late_n else ""
			log(f"  world {world}: {ratio}" + ", ".join(f"{k} {v:.0f}" for k, v in late_n.items()))

	# K sweep: food level R* (where agents that do not adapt settle) against stability
	sweep = sorted(
		(c for c in data if c.startswith("B_") and c.split("_K")[0] in ("B_random", "B_nn_none")),
		key=lambda c: (c.split("_K")[0], capacity(c)),
	)
	if sweep:
		log("\n=== K sweep, world B (last third of each run; R* = mean food stock)")
		log(f"  {'condition':<22} {'K':>6} {'viable':>7} {'N mean':>7} {'CV N':>5} {'R*':>6} {'R*/K':>5} {'food p5-p95':>13}")
		for c in sweep:
			P, F = data[c]["series"]["population"], data[c]["series"]["food_items"]
			start = 2 * P.shape[1] // 3
			sex1 = data[c]["series"]["sex1"][:, -1]
			alive = (sex1 > 0) & (sex1 < P[:, -1])  # viable: both sexes left
			k = capacity(c)
			if not alive.any():
				log(f"  {c:<22} {k:>6} {0:>3}/{len(P):<3}")
				continue
			n, f = P[alive, start:], F[alive, start:]
			cv = np.mean(n.std(axis=1) / n.mean(axis=1))
			log(f"  {c:<22} {k:>6} {alive.sum():>3}/{len(P):<3} {n.mean():>7.1f} {cv:>5.2f} {f.mean():>6.0f} "
				f"{f.mean() / k:>5.2f} {np.percentile(f, 5):>6.0f}-{np.percentile(f, 95):<6.0f}")

	text = "\n".join(lines)
	print(text)
	with open(os.path.join(roots[-1], "report.txt"), "w") as f:
		f.write(text + "\n")


if __name__ == "__main__":
	if len(sys.argv) >= 3 and sys.argv[1] == "--report":
		report(sys.argv[2:])
		sys.exit()

	batch = sys.argv[1] if len(sys.argv) > 1 else "pilot"
	root = os.path.join("results", f"{batch}_{datetime.now():%Y-%m-%d_%H%M%S}")
	os.makedirs(root)
	# longest jobs first, so the short ones fill the threads that free up
	jobs = sorted(
		((c, SEEDS[batch] + i * RUNS_PER_JOB, options, root) for c, n, options in BATCHES[batch] for i in range(n)),
		key=lambda job: -job[2].get("ticks", TICKS),
	)
	print(f"{batch}: {len(jobs)} jobs x {RUNS_PER_JOB} runs x 2 grids -> {root}", flush=True)
	with Pool(min(len(jobs), os.cpu_count())) as pool:
		pool.map(work, jobs, chunksize=1)
	report([root])
