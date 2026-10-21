import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import numpy as np

from environmental_abm import (
	BASELINE_SEED_OFFSET, SERIES, Agent, Food, Grid, SimpleNN, Wall, divergence_stats,
	event_rates, grid_difference, main_simulation, population_stats, split_rate_if_independent
)


def empty_grid(width=10, height=10, **overrides):
	params = dict(
		width=width, height=height, metabolic_cost=0.9, min_child_energy=7, reproduction_cost=8,
		food_respawn_rate=0.0, num_agents=0, num_apples=0, num_oranges=0, num_walls=0,
		use_nn=False, seed=1
	)
	params.update(overrides)
	return Grid(**params)


def add_agent(grid, x, y, sex, energy):
	agent = Agent(x, y, sex, (0, 0, 1) if sex == 0 else (0, 0, 0.5), energy=energy)
	if grid.use_nn:
		agent.nn = SimpleNN(grid.rng)
	grid.place_object(agent)
	grid.agents.append(agent)
	return agent


def total_energy(grid):
	return sum(a.energy for a in grid.agents)


def assert_consistent(test, grid):
	"""Agent and food lists must match the grid exactly, and no agent may live with E <= 0."""
	cells = [obj for row in grid.grid for obj in row if obj is not None]
	agents_on_grid = [obj for obj in cells if isinstance(obj, Agent)]
	food_on_grid = [obj for obj in cells if isinstance(obj, Food)]

	test.assertEqual(len(agents_on_grid), len(grid.agents))
	test.assertEqual(len(food_on_grid), len(grid.food_items))
	test.assertEqual(len(set(map(id, grid.agents))), len(grid.agents))
	for agent in grid.agents:
		test.assertTrue(agent.alive)
		test.assertIs(grid.grid[agent.x][agent.y], agent)
		test.assertGreater(agent.energy, 0)
	for food in grid.food_items:
		test.assertIs(grid.grid[food.x][food.y], food)


class TickTest(unittest.TestCase):

	def test_every_agent_acts_once_per_tick(self):
		# a fully packed grid of one sex: every move is a fight, and many agents get
		# killed before their turn. The thesis version stopped the tick at the first one.
		grid = empty_grid()
		for x in range(10):
			for y in range(10):
				add_agent(grid, x, y, sex=0, energy=10)

		grid.move_agent()

		self.assertGreater(grid.stats["fights"], 0)
		# an agent that skipped its turn would still have exactly 10
		for agent in grid.agents:
			self.assertNotAlmostEqual(agent.energy, 10)
		self.assertGreaterEqual(grid.stats["acted"], len(grid.agents))
		self.assertAlmostEqual(grid.stats["metabolism"], 0.9 * grid.stats["acted"])
		assert_consistent(self, grid)

	def test_energy_budget_and_consistency(self):
		for use_nn in (False, True):
			with self.subTest(use_nn=use_nn):
				grid = Grid(
					width=30, height=30, metabolic_cost=0.9, min_child_energy=7, reproduction_cost=8,
					food_respawn_rate=0.012, num_agents=60, num_apples=150, num_oranges=150,
					num_walls=10, use_nn=use_nn, seed=7
				)
				events = dict.fromkeys(("fights", "matings", "failed_matings", "starved"), 0)

				for _ in range(300):
					before = total_energy(grid)
					grid.move_agent()
					grid.respawn_food()
					s = grid.stats
					expected = s["food"] - s["metabolism"] - s["reproduction"] - s["residual"]
					self.assertAlmostEqual(total_energy(grid) - before, expected, delta=1e-6)
					assert_consistent(self, grid)
					for key in events:
						events[key] += s[key]
					if not grid.agents:
						break

				if not use_nn:
					# make sure the random run actually exercised every interaction path
					for key, count in events.items():
						self.assertGreater(count, 0, key)

	def test_determinism(self):
		for use_nn in (False, True):
			with self.subTest(use_nn=use_nn):
				params = dict(
					width=30, height=30, metabolic_cost=0.9, min_child_energy=7, reproduction_cost=8,
					food_respawn_rate=0.012, num_agents=60, num_apples=150, num_oranges=150,
					num_walls=10, use_nn=use_nn, seed=3
				)
				g1, g2 = Grid(**params), Grid(**params)
				for _ in range(150):
					for g in (g1, g2):
						g.move_agent()
						g.respawn_food()
				self.assertEqual(grid_difference(g1, g2), 0)


class MatingTest(unittest.TestCase):

	def test_children_split_energy_left_after_cost(self):
		grid = empty_grid()
		a = add_agent(grid, 5, 5, sex=0, energy=20)
		b = add_agent(grid, 5, 6, sex=1, energy=20)

		grid.meet(a, b)

		# (20 - 8) + (20 - 8) = 24 -> floor(24 / 7) = 3 children with 8 each
		children = [x for x in grid.agents if x.alive]
		self.assertFalse(a.alive or b.alive)
		self.assertEqual(len(children), 3)
		for child in children:
			self.assertAlmostEqual(child.energy, 8)
		self.assertEqual(grid.stats["births"], 3)
		self.assertEqual(grid.stats["reproduction"], 16)

	def test_failed_mating_kills_exhausted_parent(self):
		grid = empty_grid()
		a = add_agent(grid, 5, 5, sex=0, energy=5)
		b = add_agent(grid, 5, 6, sex=1, energy=9)

		grid.meet(a, b)

		# -3 + 1 < 7 -> no child; a ran out of energy, b survives with 1
		self.assertFalse(a.alive)
		self.assertTrue(b.alive)
		self.assertAlmostEqual(b.energy, 1)
		self.assertEqual(grid.stats["failed_matings"], 1)
		self.assertAlmostEqual(grid.stats["residual"], -3)
		self.assertIsNone(grid.grid[5][5])

	def test_failed_mating_without_space_keeps_partner_on_grid(self):
		# 3x3 grid walled off except for the two parents - there is never room for a child,
		# and the only possible move is into the partner
		grid = empty_grid(width=3, height=3)
		for x in range(3):
			for y in range(3):
				if (x, y) not in ((1, 1), (1, 2)):
					wall = Wall(x, y)
					grid.place_object(wall)
					grid.walls.append(wall)
		add_agent(grid, 1, 1, sex=0, energy=100)
		add_agent(grid, 1, 2, sex=1, energy=100)

		failed = 0
		for _ in range(200):
			grid.move_agent()
			failed += grid.stats["failed_matings"]
			assert_consistent(self, grid)
			if len(grid.agents) < 2:
				break

		self.assertGreater(failed, 0)
		self.assertEqual(grid.stats["births"], 0)


class InheritanceTest(unittest.TestCase):

	def test_siblings_get_different_mutations(self):
		# networks used to deep-copy their parent's RNG, so from the second generation on
		# all children of one parent mutated identically
		grid = empty_grid(use_nn=True)
		a = add_agent(grid, 5, 5, sex=0, energy=30)
		b = add_agent(grid, 5, 6, sex=1, energy=30)
		parent = grid.cross_mutate(a.nn, b.nn)
		self.assertIs(parent.rng, grid.rng)

		# crossing a network with itself leaves only the mutation
		children = [grid.cross_mutate(parent, parent) for _ in range(3)]
		self.assertFalse(np.array_equal(children[0].W1, children[1].W1))
		self.assertFalse(np.array_equal(children[1].W1, children[2].W1))


	def test_inheritance_modes(self):
		for inheritance, same_as_parents in (("evolve", False), ("no_mutation", True), ("none", False)):
			with self.subTest(inheritance=inheritance):
				grid = empty_grid(use_nn=True, inheritance=inheritance)
				a = add_agent(grid, 5, 5, sex=0, energy=30)
				b = add_agent(grid, 5, 6, sex=1, energy=30)
				b.nn.W1[:], b.nn.W2[:] = a.nn.W1, a.nn.W2  # identical parents: crossover is a no-op

				grid.meet(a, b)
				child = next(x for x in grid.agents if x.alive)
				self.assertEqual(np.array_equal(child.nn.W1, a.nn.W1), same_as_parents)

	def test_invalid_options(self):
		for overrides in ({"inheritance": "lamarck"}, {"regrowth": "linear"}, {"regrowth": "logistic"}):
			with self.subTest(**overrides), self.assertRaises(ValueError):
				empty_grid(**overrides)


class RegrowthTest(unittest.TestCase):

	def test_logistic_regrowth(self):
		# r * F * (1 - F / K) = 0.5 * 20 * (1 - 20 / 100) = 8 new items
		grid = empty_grid(regrowth="logistic", food_capacity=100, food_respawn_rate=0.5, num_apples=20)
		grid.respawn_food()
		self.assertEqual(len(grid.food_items), 28)
		assert_consistent(self, grid)

	def test_logistic_regrowth_bounds(self):
		# no food left -> none ever comes back; at or above capacity -> no growth
		for apples, expected in ((0, 0), (100, 100)):
			with self.subTest(apples=apples):
				grid = empty_grid(
					width=20, height=20, regrowth="logistic", food_capacity=100, food_respawn_rate=0.5, num_apples=apples
				)
				grid.respawn_food()
				self.assertEqual(len(grid.food_items), expected)

	def test_logistic_regrowth_rounds_stochastically(self):
		# expected 0.5 * 1 * (1 - 1 / 100) = 0.495 items per tick, so a single item can still regrow
		grid = empty_grid(regrowth="logistic", food_capacity=100, food_respawn_rate=0.5, num_apples=1)
		for _ in range(10):
			grid.respawn_food()
		self.assertGreater(len(grid.food_items), 1)


class DivergenceTest(unittest.TestCase):

	def test_divergence_stats(self):
		# latent for 3 ticks, then a jump (desync) and a rise to the 0.4 plateau
		d = [1e-4, 1e-4, 1e-4, 1e-4, 0.05, 0.1, 0.25, 0.4, 0.4]
		s = divergence_stats(d, d_ref=0.4, cells=1000)
		self.assertEqual(s["desync_tick"], 4)  # first d >= 10 cells / 1000
		self.assertEqual(s["half_tick"], 6)    # first d >= 0.2
		self.assertAlmostEqual(s["lyapunov"], np.log(0.25 / 1e-4) / 6)

	def test_divergence_never_spreads(self):
		s = divergence_stats([1e-4, 2e-4, 1e-4], d_ref=0.4, cells=100)
		self.assertIsNone(s["desync_tick"])
		self.assertIsNone(s["half_tick"])
		self.assertAlmostEqual(s["lyapunov"], 0.0)  # over the whole run

	def test_split_rate_if_independent(self):
		self.assertAlmostEqual(split_rate_if_independent(["extinct"] * 4, ["extinct"] * 4), 0.0)
		# half / half on both sides -> split half of the time
		a = ["extinct", "extinct", "established", "established"]
		self.assertAlmostEqual(split_rate_if_independent(a, a[::-1]), 0.5)

	def test_baseline_grid_starts_identical(self):
		for use_nn in (False, True):
			with self.subTest(use_nn=use_nn):
				params = dict(
					width=30, height=30, metabolic_cost=0.9, min_child_energy=7, reproduction_cost=8,
					food_respawn_rate=0.012, num_agents=60, num_apples=150, num_oranges=150,
					num_walls=10, use_nn=use_nn, seed=3
				)
				g1, g3 = Grid(**params), Grid(**params)
				g3.rng.seed(BASELINE_SEED_OFFSET + 3)
				self.assertEqual(grid_difference(g1, g3), 0)
				if use_nn:
					# initial networks are identical and follow the reseeded RNG
					self.assertTrue(np.array_equal(g1.agents[0].nn.W1, g3.agents[0].nn.W1))
					self.assertIs(g3.agents[0].nn.rng, g3.rng)
				for _ in range(30):
					for g in (g1, g3):
						g.move_agent()
						g.respawn_food()
				self.assertGreater(grid_difference(g1, g3), 0)


class PopulationStatsTest(unittest.TestCase):

	def test_established(self):
		pop = [10, 12, 20, 15, 14]
		energy = [100, 150, 300, 330, 280]
		s = population_stats(pop, energy, initial_agents=10, start=2)

		self.assertEqual(s["outcome"], "established")
		self.assertIsNone(s["extinction_tick"])
		self.assertEqual((s["peak_population"], s["peak_population_tick"]), (20, 2))
		self.assertEqual((s["peak_energy"], s["peak_energy_tick"]), (330, 3))
		self.assertAlmostEqual(s["mean_population"], (20 + 15 + 14) / 3)
		self.assertAlmostEqual(s["mean_energy_per_agent"], (300 / 20 + 330 / 15 + 280 / 14) / 3)
		self.assertAlmostEqual(s["final_energy_per_agent"], 20)

	def test_declined_and_extinct(self):
		self.assertEqual(population_stats([10, 5, 3], [100, 60, 900], 10, 1)["outcome"], "declined")

		s = population_stats([10, 4, 0, 0], [100, 30, 0, 0], initial_agents=10, start=1)
		self.assertEqual(s["outcome"], "extinct")
		self.assertEqual(s["extinction_tick"], 2)
		self.assertTrue(np.isnan(s["final_energy_per_agent"]))
		# ticks without agents are left out of the energy-per-agent mean
		self.assertAlmostEqual(s["mean_energy_per_agent"], 30 / 4)

	def test_event_rates(self):
		series = {key: [0, 0, 0] for key in SERIES}
		series["acted"] = [0, 10, 30]
		series["fights"] = [0, 2, 6]
		series["food"] = [0, 50, 150]
		rates = event_rates(series, start=1)
		# total events / total agent-ticks
		self.assertAlmostEqual(rates["rate_fights"], 8 / 40)
		self.assertAlmostEqual(rates["rate_food"], 200 / 40)
		self.assertAlmostEqual(rates["rate_births"], 0)
		self.assertTrue(np.isnan(event_rates(series, start=3)["rate_fights"]))  # no agent-ticks


class ResultsTest(unittest.TestCase):

	def test_main_simulation_saves_output(self):
		params = dict(
			width=20, height=20, metabolic_cost=0.9, min_child_energy=7, reproduction_cost=2,
			food_respawn_rate=0.012, num_agents=15, num_apples=30, num_oranges=30,
			num_walls=5, use_nn=False
		)
		runs, ticks = 2, 40
		# render flags off: grid snapshots are saved but not displayed, only d(t) plots are shown
		with tempfile.TemporaryDirectory() as root, contextlib.redirect_stdout(io.StringIO()), \
				mock.patch("environmental_abm.plt.get_backend", return_value="inline"), \
				mock.patch("environmental_abm.plt.show") as show:
			results = main_simulation(
				num_runs=runs, num_ticks=ticks, num_prtrb_agents=2, init_seed=5,
				final_render=False, lyapunov_final_render=False, name="test", results_dir=root, **params
			)
			self.assertEqual(show.call_count, runs)
			out = results.dir
			self.assertTrue(os.path.basename(out).endswith("_test"))

			with open(os.path.join(out, "params.json")) as f:
				self.assertEqual(json.load(f)["num_ticks"], ticks)
			with open(os.path.join(out, "summary.txt")) as f:
				summary = f.read()
			for section in ("PARAMETERS LIST", "DETERMINISTIC CHECK", "DIVERGENCE COMPARISON", "REGIME SUMMARY",
							"POPULATION SUMMARY", "Outcome agreement (grid 1 / baseline)"):
				self.assertIn(section, summary)

			with open(os.path.join(out, "runs.csv")) as f:
				rows = list(csv.DictReader(f))
			self.assertEqual([(r["run"], r["grid"]) for r in rows], [("1", "1"), ("1", "2"), ("2", "1"), ("2", "2")])
			for key in ("desync_tick", "baseline_d_late", "rate_fights", "baseline_outcome"):
				self.assertIn(key, rows[0])

			series = np.load(os.path.join(out, "timeseries.npz"))
			self.assertEqual(series["d"].shape, (runs, ticks + 1))
			self.assertEqual(series["population"].shape, (runs, 2, ticks + 1))
			self.assertEqual(series["energy"].shape, (runs, 2, ticks + 1))
			for key in SERIES:
				self.assertEqual(series[key].shape, (runs, 2, ticks + 1), key)
			self.assertEqual(series["baseline_d"].shape, (runs, ticks + 1))
			self.assertEqual(series["baseline_population"].shape, (runs, ticks + 1))
			# tick 0 has no events, and the saved counters keep the energy budget
			self.assertEqual(series["acted"][:, :, 0].sum(), 0)
			budget = series["food"] - series["metabolism"] - series["reproduction"] - series["residual"]
			np.testing.assert_allclose(np.diff(series["energy"], axis=2), budget[:, :, 1:], atol=1e-2)
			# the saved series must agree with the per-run table
			for r in rows:
				self.assertEqual(series["population"][int(r["run"]) - 1, int(r["grid"]) - 1, -1], int(r["final_population"]))

			self.assertEqual(sorted(os.listdir(os.path.join(out, "plots"))), ["divergence_run01.png", "divergence_run02.png"])
			# 4 determinism renders + 4 per run (2 grids, initial and final)
			self.assertEqual(len(os.listdir(os.path.join(out, "states"))), 4 + 4 * runs)


if __name__ == "__main__":
	unittest.main()
