"""
Pilot: does adaptation help or hurt the population, depending on how food regrows?

World A - food spawns in proportion to empty cells: grazing the stock down raises supply.
World B - logistic regrowth r * F * (1 - F / K): grazing below K / 2 lowers supply.
K and r are calibrated so random agents settle at the same N and food stock in both worlds
(N ≈ 550, F ≈ 2480, which is the logistic peak in B). Both worlds start pre-stocked with food,
so the start-up phase, when the initial networks barely forage, is the same in both.

Agents: random, neural with evolving networks, neural with crossover only (B only), and neural
with no inheritance (every child gets a fresh random network, so nothing adapts).

	python pilot_regrowth.py                   # run all conditions in parallel, then report
	python pilot_regrowth.py --report <folder> # report on a finished pilot
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # worker processes must never open plot windows

import contextlib
import csv
import json
import sys
from datetime import datetime
from multiprocessing import Pool

import numpy as np

from environmental_abm import main_simulation

# PILOT_TICKS / PILOT_RUNS override these for quick tests
TICKS = int(os.environ.get("PILOT_TICKS", 30000))
RUNS_PER_JOB = int(os.environ.get("PILOT_RUNS", 10))
SEED = 123

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
# (condition, number of 10-run jobs): 12 jobs for 12 hardware threads
CONDITIONS = [
	("A_random", 2), ("B_random", 2),
	("A_nn_evolve", 2), ("B_nn_evolve", 2),
	("A_nn_none", 1), ("B_nn_none", 1),
	("B_nn_no_mutation", 2),
]

WINDOWS = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 10000), (10000, 15000),
		   (15000, 20000), (20000, 25000), (25000, 30001)]


def params(condition):
	world, agents = condition.split("_", 1)
	return {**COMMON, **WORLDS[world], **AGENTS[agents]}


def work(job):
	condition, seed, root = job
	with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
		results = main_simulation(
			num_runs=RUNS_PER_JOB, num_ticks=TICKS, num_prtrb_agents=2, init_seed=seed, baseline=False,
			name=f"{condition}__s{seed}", results_dir=root, **params(condition)
		)
	print(f"[{datetime.now():%H:%M}] done: {condition} seeds {seed}-{seed + RUNS_PER_JOB - 1}", flush=True)
	return results.dir


def load(root):
	"""Merge the jobs of each condition: {condition: {"rows": [...], "series": {key: (grids, ticks + 1)}}}"""
	data = {}
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
		entry["rows"] += rows
		for key in ("population", "food_items", "fights", "matings", "acted"):
			grids = s[key].reshape(-1, s[key].shape[-1])  # (runs * 2 grids, ticks + 1), same order as rows
			entry["series"][key] = np.concatenate([entry["series"][key], grids]) if key in entry["series"] else grids
	return data


def report(root):
	data = load(root)
	lines = []
	log = lines.append
	log(f"Pilot report - {root}\n")

	for condition, _ in CONDITIONS:
		if condition not in data:
			continue
		d = data[condition]
		P, F = d["series"]["population"], d["series"]["food_items"]
		n0 = COMMON["num_agents"]
		established_once = (P > 2 * n0).any(axis=1)
		extinct = P[:, -1] == 0
		ext_tick = np.where(extinct, np.argmax(P == 0, axis=1), -1)
		late = extinct & established_once

		log(f"=== {condition}  ({len(P)} grids, pairs share initial networks)")
		log(f"  alive at end {np.sum(~extinct)}/{len(P)} | early extinctions (never > {2 * n0}) {np.sum(extinct & ~established_once)}"
			f" | LATE extinctions (after establishing) {np.sum(late)}"
			+ (f", ticks {sorted(ext_tick[late].tolist())}" if late.any() else ""))
		log(f"  {'window':>13} {'alive':>6} {'N mean':>8} {'N sd':>6} {'food':>7} {'encounters/agent':>17}")
		for a, b in WINDOWS:
			if a >= P.shape[1]:
				break
			alive = P[:, min(b, P.shape[1]) - 1] > 0
			if not alive.any():
				log(f"  {a:>6}-{b:<6} {0:>6}")
				continue
			n = P[alive, a:b]
			enc = (d["series"]["fights"][alive, a:b].sum() + d["series"]["matings"][alive, a:b].sum()) \
				/ max(d["series"]["acted"][alive, a:b].sum(), 1)
			log(f"  {a:>6}-{b:<6} {alive.sum():>6} {n.mean():>8.1f} {n.mean(axis=1).std():>6.1f} "
				f"{F[alive, a:b].mean():>7.0f} {enc:>17.4f}")
		log("")

	# decision: how evolving populations end up relative to random agents in each world
	log("=== Decision summary (late window, grids alive at the end)")
	for world in "AB":
		late_n = {}
		for agents in ("random", "nn_evolve", "nn_none"):
			c = f"{world}_{agents}"
			if c in data:
				P = data[c]["series"]["population"]
				alive = P[:, -1] > 0
				late_n[agents] = P[alive, -5000:].mean() if alive.any() else 0.0
		if "random" in late_n and "nn_evolve" in late_n and late_n["random"]:
			log(f"  world {world}: N evolve / N random = {late_n['nn_evolve'] / late_n['random']:.2f}  "
				+ ", ".join(f"{k} {v:.0f}" for k, v in late_n.items()))
	log("  A hit: ratio > 1 in A and < 1 in B, with N in B_nn_evolve falling over time,")
	log("  and B_nn_none not showing the same fall.")

	text = "\n".join(lines)
	print(text)
	with open(os.path.join(root, "report.txt"), "w") as f:
		f.write(text + "\n")


if __name__ == "__main__":
	if len(sys.argv) == 3 and sys.argv[1] == "--report":
		report(sys.argv[2])
		sys.exit()

	root = os.path.join("results", f"pilot_{datetime.now():%Y-%m-%d_%H%M%S}")
	os.makedirs(root)
	jobs = [(c, SEED + i * RUNS_PER_JOB, root) for c, n in CONDITIONS for i in range(n)]
	print(f"Pilot: {len(jobs)} jobs x {RUNS_PER_JOB} runs x 2 grids x {TICKS} ticks -> {root}", flush=True)
	with Pool(min(len(jobs), os.cpu_count())) as pool:
		pool.map(work, jobs, chunksize=1)
	report(root)
