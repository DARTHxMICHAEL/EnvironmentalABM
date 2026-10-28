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
	sweep     - the main K sweep in world B, about a week of CPU on a mid-size server (see
	            --dry-run): evolving agents with 5% noise over K = 2000-10000 for 150k ticks, plus
	            crossover-only, no-noise and no-inheritance controls, random agents over K = 2000-15000,
	            and world A for 150k ticks as the stable reference

	python pilot_regrowth.py [batch]                       # run a batch in parallel, then report
	python pilot_regrowth.py [batch] --dry-run             # list jobs with rough CPU time and storage
	python pilot_regrowth.py [batch] --resume <folder>     # continue a stopped batch, skipping finished jobs
	python pilot_regrowth.py --report <folder> ...         # report on finished batches (merged)

Environment: PILOT_WORKERS parallel processes (default: all CPUs), PILOT_TICKS / PILOT_RUNS /
PILOT_MAX_SEEDS / PILOT_TICKS_CAP shrink batches for quick tests.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # worker processes must never open plot windows

import contextlib
import csv
import json
import math
import re
import sys
import traceback
from datetime import datetime
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np

from environmental_abm import main_simulation

# PILOT_TICKS / PILOT_RUNS / PILOT_MAX_SEEDS override these for quick tests
TICKS = int(os.environ.get("PILOT_TICKS", 30000))
RUNS_PER_JOB = int(os.environ.get("PILOT_RUNS", 10))
MAX_SEEDS = int(os.environ.get("PILOT_MAX_SEEDS", 10**6))
TICKS_CAP = int(os.environ.get("PILOT_TICKS_CAP", 10**9))  # caps every job's ticks, for quick tests
WORKERS = int(os.environ.get("PILOT_WORKERS", os.cpu_count()))
# first seed of each batch: the snapshots batch continues after the pilot's 20 seeds, the sweep
# uses fresh seeds
SEEDS = {"pilot": 123, "snapshots": 143, "noise": 143, "sweep": 301}

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
SHORT = dict(ticks=TICKS // 2)
# main sweep: long runs (does falling R* destabilize every K eventually?), small jobs for many cores
SWEEP_SEEDS = 20
LONG = dict(ticks=150_000, snapshot_every=5000, runs=2, determinism_ticks=2000)
MEDIUM = dict(ticks=50_000, snapshot_every=5000, runs=2, determinism_ticks=2000)
STATIONARY = dict(ticks=30_000, runs=2, determinism_ticks=2000)
K_EVOLVE = (2000, 2500, 3000, 3500, 4000, 5000, 6000, 8000, 10000)
K_RANDOM = (2000, 2250, 2500, 2750, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 10000, 12500, 15000)

# (condition, number of seeds, options: ticks, snapshot_every, runs per job, determinism_ticks)
BATCHES = {
	"pilot": [
		("A_random", 20, {}), ("B_random", 20, {}),
		("A_nn_evolve", 20, {}), ("B_nn_evolve", 20, {}),
		("A_nn_none", 10, {}), ("B_nn_none", 10, {}),
		("B_nn_no_mutation", 20, {}),
	],
	"snapshots": [
		("A_nn_evolve", 20, dict(snapshot_every=2500)), ("A_nn_no_mutation", 20, dict(snapshot_every=2500)),
		("B_nn_evolve", 20, dict(snapshot_every=2500)), ("B_nn_no_mutation", 20, dict(snapshot_every=2500)),
		*[(f"B_nn_none_K{k}", 10, SHORT) for k in (3000, 4000, 7000, 10000)],
		*[(f"B_random_K{k}", 10, SHORT) for k in (3000, 4000, 6000, 7000, 8000, 10000)],
	],
	"noise": [
		("B_nn_evolve_n5", 20, dict(snapshot_every=2500)), ("B_nn_no_mutation_n5", 20, {}),
		("B_nn_evolve_n1", 20, {}), ("B_nn_no_mutation_n1", 20, {}),
		("A_nn_evolve_n5", 20, {}), ("B_nn_none_n5", 10, {}),
	],
	"sweep": [
		# main result: evolving agents (5% noise, so no trap) across K
		*[(f"B_nn_evolve_K{k}_n5", SWEEP_SEEDS, LONG) for k in K_EVOLVE],
		# reference: world A, where falling R* should never destabilize
		("A_nn_evolve_n5", SWEEP_SEEDS, LONG),
		# ratchet: does running out of variation stop the destabilization?
		*[(f"B_nn_no_mutation_K{k}_n5", SWEEP_SEEDS, LONG) for k in (2500, 3500, 5000, 8000)],
		# with the trap: cycles turn into extinction
		*[(f"B_nn_evolve_K{k}", SWEEP_SEEDS, MEDIUM) for k in (2500, 3500, 5000, 6000, 8000)],
		# no adaptation: the textbook baseline at the same K
		*[(f"B_nn_none_K{k}_n5", SWEEP_SEEDS, STATIONARY) for k in K_EVOLVE],
		*[(f"B_random_K{k}", SWEEP_SEEDS, STATIONARY) for k in K_RANDOM],
	],
}

WINDOWS = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 10000), (10000, 15000),
		   (15000, 20000), (20000, 25000), (25000, 30001), (30001, 50001), (50001, 75000),
		   (75000, 100000), (100000, 125000), (125000, 150001)]


def params(condition):
	match = re.fullmatch(r"([AB])_(.+?)(?:_K(\d+))?(?:_n(\d+))?", condition)
	world, agents, capacity, noise = match.groups()
	p = {**COMMON, **WORLDS[world], **AGENTS[agents]}
	if capacity:
		p["food_capacity"] = int(capacity)
	if noise:
		p["move_noise"] = int(noise) / 100
	return p


def jobs_of(batch, root):
	"""(condition, first seed, runs, options, root) for every job of a batch."""
	jobs = []
	for condition, seeds, options in BATCHES[batch]:
		seeds = min(seeds, MAX_SEEDS)
		runs = min(options.get("runs", RUNS_PER_JOB), RUNS_PER_JOB, seeds)
		for i in range(math.ceil(seeds / runs)):
			jobs.append((condition, SEEDS[batch] + i * runs, min(runs, seeds - i * runs), options, root))
	# longest first, so the short ones fill the CPUs that free up
	return sorted(jobs, key=lambda job: -estimate(job)[0])


def finished(root):
	"""Names (<condition>__s<seed>) of the jobs that completed in a batch folder."""
	done = set()
	for folder in os.listdir(root):
		path = os.path.join(root, folder)
		if os.path.exists(os.path.join(path, "timeseries.npz")):
			with open(os.path.join(path, "params.json")) as f:
				done.add(json.load(f)["name"])
	return done


def estimate(job):
	"""Rough CPU hours and GB of one job, from benchmarks on a desktop CPU (i5-12400F)."""
	condition, seed, runs, options, root = job
	p = params(condition)
	ticks = min(options.get("ticks", TICKS), TICKS_CAP)
	agents = min(900, 200 + 0.08 * p.get("food_capacity", 5000))  # rough population size
	per_agent = 40e-6 if p["use_nn"] else 4e-6                       # seconds per agent and tick
	grid_ticks = 2 * runs * ticks + 2 * min(options.get("determinism_ticks", ticks), ticks)
	hours = grid_ticks * (agents * per_agent + 1.5e-3) / 3600
	every = options.get("snapshot_every")
	gb = 2 * runs * ticks * 16 * 4 * 0.6 / 1e9                        # time series, compressed ~60%
	if every:
		gb += 2 * runs * (ticks / every + 1) * (100 * 664 * 4 + agents * 40) / 1e9
		if not p.get("move_noise"):
			gb += 2 * runs * 2.7e6 / 1e9                                  # endgame snapshots (trap)
	return hours, gb + runs * 0.3e-3                                     # + plots and grid images


def work(job):
	condition, seed, runs, options, root = job
	# under %run in Jupyter matplotlib is already set to the inline backend, and forked workers
	# must not push plots into the notebook - plots are only saved
	plt.switch_backend("Agg")
	try:
		with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
			results = main_simulation(
				num_runs=runs, num_ticks=min(options.get("ticks", TICKS), TICKS_CAP), num_prtrb_agents=2,
				init_seed=seed, baseline=False, snapshot_every=options.get("snapshot_every"),
				determinism_ticks=min(options.get("determinism_ticks", TICKS_CAP), TICKS_CAP),
				name=f"{condition}__s{seed}", results_dir=root, **params(condition)
			)
		return results.dir
	except Exception:
		# one failing job must not stop a week-long batch: log it, the others keep running
		with open(os.path.join(root, "errors.log"), "a") as f:
			f.write(f"[{datetime.now():%Y-%m-%d %H:%M}] {condition} seed {seed}\n{traceback.format_exc()}\n")
		return None


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


def instability_onset(P, start=3000, block=1000, threshold=0.4):
	"""Per grid: first tick of a block (after `start`) where N swings (CV > threshold) or hits 0; nan if never."""
	blocks = (P.shape[1] - start) // block
	if blocks <= 0:
		return np.full(len(P), np.nan)
	b = P[:, start:start + blocks * block].reshape(len(P), blocks, block).astype(float)
	unstable = (b.std(axis=2) > threshold * b.mean(axis=2)) | (b.min(axis=2) == 0)
	return np.where(unstable.any(axis=1), start + block * unstable.argmax(axis=1), np.nan)


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

	# K sweeps: every world-B family run at more than one K (e.g. B_nn_evolve_n5 at K2000..K10000)
	families = {}
	for c in data:
		if c.startswith("B_"):
			families.setdefault(re.sub(r"_K\d+", "", c), []).append(c)
	families = {f: sorted(cs, key=capacity) for f, cs in sorted(families.items()) if len(cs) > 1}
	if families:
		log("\n=== K sweeps, world B (last third of each run, viable grids)")
		log("  food/K is R*/K only where the population is stable (low CV); onset = first 1000-tick block")
		log("  after tick 3000 with CV of N > 0.4 or a collapse to 0, median over grids that have one")
		log(f"  {'condition':<26} {'K':>6} {'viable':>7} {'N mean':>7} {'CV N':>5} {'food':>6} {'food/K':>6} "
			f"{'p5-p95 food':>12} {'unstable':>8} {'onset':>7}")
		for family, conditions in families.items():
			for c in conditions:
				P, F = data[c]["series"]["population"], data[c]["series"]["food_items"]
				sex1 = data[c]["series"]["sex1"][:, -1]
				viable = (sex1 > 0) & (sex1 < P[:, -1])  # both sexes left
				onset = instability_onset(P)
				unstable = np.isfinite(onset)
				when = f"{np.median(onset[unstable]):>7.0f}" if unstable.any() else f"{'-':>7}"
				k = capacity(c)
				head = f"  {c:<26} {k:>6} {viable.sum():>3}/{len(P):<3}"
				if not viable.any():
					log(f"{head} {'':>7} {'':>5} {'':>6} {'':>6} {'':>12} {unstable.sum():>4}/{len(P):<3} {when}")
					continue
				start = 2 * P.shape[1] // 3
				n, f = P[viable, start:], F[viable, start:]
				cv = np.mean(n.std(axis=1) / n.mean(axis=1))
				log(f"{head} {n.mean():>7.1f} {cv:>5.2f} {f.mean():>6.0f} {f.mean() / k:>6.2f} "
					f"{np.percentile(f, 5):>5.0f}-{np.percentile(f, 95):<6.0f} {unstable.sum():>4}/{len(P):<3} {when}")
			log("")

	text = "\n".join(lines)
	print(text)
	with open(os.path.join(roots[-1], "report.txt"), "w") as f:
		f.write(text + "\n")


if __name__ == "__main__":
	if len(sys.argv) >= 3 and sys.argv[1] == "--report":
		report(sys.argv[2:])
		sys.exit()

	args = sys.argv[1:]
	batch = args[0] if args and not args[0].startswith("--") else "pilot"
	if "--resume" in args:
		root = args[args.index("--resume") + 1]
	else:
		root = os.path.join("results", f"{batch}_{datetime.now():%Y-%m-%d_%H%M%S}")
	jobs = jobs_of(batch, root)

	if "--dry-run" in args:
		total_h = total_gb = 0.0
		rows = {}
		for job in jobs:
			h, gb = estimate(job)
			row = rows.setdefault(job[0], [0, 0, 0.0, 0.0, job[3].get("ticks", TICKS)])
			row[0] += 1; row[1] += job[2]; row[2] += h; row[3] += gb
			total_h += h; total_gb += gb
		print(f"{batch}: {len(jobs)} jobs (rough estimates, desktop CPU speed)")
		print(f"  {'condition':<26} {'jobs':>5} {'seeds':>6} {'ticks':>8} {'CPU h':>7} {'GB':>6}")
		for c, (n, seeds, h, gb, ticks) in rows.items():
			print(f"  {c:<26} {n:>5} {seeds:>6} {ticks:>8} {h:>7.1f} {gb:>6.2f}")
		print(f"  total: {total_h:.0f} CPU hours, {total_gb:.1f} GB")
		print(f"  with {WORKERS} workers: about {total_h / WORKERS:.0f} h ({total_h / WORKERS / 24:.1f} days), "
			  f"if each worker gets a full core; the longest job alone takes {estimate(jobs[0])[0]:.0f} h")
		sys.exit()

	os.makedirs(root, exist_ok=True)
	done = finished(root)
	todo = [job for job in jobs if f"{job[0]}__s{job[1]}" not in done]
	print(f"{batch}: {len(todo)} of {len(jobs)} jobs to run, {WORKERS} workers -> {root}", flush=True)
	with Pool(min(len(todo), WORKERS) or 1) as pool:
		for i, result in enumerate(pool.imap_unordered(work, todo), start=1):
			status = "done" if result else "FAILED (see errors.log)"
			print(f"[{datetime.now():%Y-%m-%d %H:%M}] {i}/{len(todo)} {status}: {os.path.basename(result or '')}", flush=True)
	report([root])
