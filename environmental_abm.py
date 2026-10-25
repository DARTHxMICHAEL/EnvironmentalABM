import random
import matplotlib.pyplot as plt
import numpy as np
import copy
import csv
import json
import os
import math
from collections import Counter
from datetime import datetime

directions = [
	(-1, 0), (1, 0), (0, -1), (0, 1),  # left, right, down, up
	(-1, -1), (-1, 1), (1, -1), (1, 1)  # diagonals
]

# per-tick event and energy-flow counters (see Grid.move_agent)
TICK_STATS = (
	"acted", "starved", "fights", "matings", "failed_matings", "births",
	"food", "metabolism", "reproduction", "residual",
	"blocked", "eaten"  # moves into an edge or wall, food items eaten
)

# counters reported as per-agent rates, see event_rates
RATE_STATS = ("food", "fights", "matings", "failed_matings", "births", "starved")

# per-tick series saved for both grids of every run, see record
SERIES = ("population", "energy", "sex1", "food_items") + TICK_STATS

# final-tick outcome of a grid, see population_stats
OUTCOMES = ("established", "declined", "extinct")

# RNG seed offset of the baseline grid, see compare_grids
BASELINE_SEED_OFFSET = 1_000_003


class Results:
	"""
	Output of one experiment. Everything printed through log() and every figure passed to
	show() is also saved under <root>/<timestamp>_<name>/:

		summary.txt     printed parameters, checks and summaries
		params.json     experiment parameters
		runs.csv        population/energy stats, one row per run and grid
		timeseries.npz  d(t), population, energy and event counters per tick for every run
		plots/          d(t) divergence plots
		states/         rendered grid states

	With root=None nothing is saved.
	"""

	def __init__(self, name, root="results"):
		self.dir = None
		if root is not None:
			stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
			self.dir = os.path.join(root, f"{stamp}_{name}")
			for subdir in ("plots", "states"):
				os.makedirs(os.path.join(self.dir, subdir), exist_ok=True)

	def path(self, *parts):
		return os.path.join(self.dir, *parts)

	def log(self, text=""):
		print(text)
		if self.dir:
			with open(self.path("summary.txt"), "a") as f:
				f.write(text + "\n")

	def show(self, fig, subdir=None, name=None, display=True):
		# save (with a name and an output folder), then show unless display=False
		if self.dir and name:
			fig.savefig(self.path(subdir, name + ".png"), dpi=100, bbox_inches="tight")
		if display and plt.get_backend().lower() != "agg":  # agg has no display, the saved file is enough
			plt.show()
		plt.close(fig)

	def save_json(self, name, data):
		if self.dir:
			with open(self.path(name), "w") as f:
				json.dump(data, f, indent=2)

	def save_csv(self, name, rows):
		if self.dir and rows:
			with open(self.path(name), "w", newline="") as f:
				writer = csv.DictWriter(f, fieldnames=list(rows[0]))
				writer.writeheader()
				writer.writerows(rows)

	def save_arrays(self, name, **arrays):
		if self.dir:
			np.savez_compressed(self.path(name), **arrays)

class Agent:
	def __init__(self, x, y, sex, color, energy=10):
		self.x = x
		self.y = y
		self.sex = sex
		self.color = color
		self.energy = energy
		self.alive = True

	def nn_output_to_move(self):
		logits = self.nn.forward(self.vision)
		idx = int(np.argmax(logits))
		return directions[idx]

class Food:
	def __init__(self, x, y, color, energy):
		self.x = x
		self.y = y
		self.color = color
		self.energy = energy

class Apple(Food):
	def __init__(self, x, y):
		super().__init__(x, y, color=(1, 0, 0), energy=5)  # red

class Orange(Food):
	def __init__(self, x, y):
		super().__init__(x, y, color=(1, 0.5, 0), energy=10)  # orange

class Wall:
	def __init__(self, x, y, color=(0, 1, 0)):  # green
		self.x = x
		self.y = y
		self.color = color

class SimpleNN:
	def __init__(self, rng, input_size=32, hidden_size=16, output_size=8, activation="tanh", mutation_std=0.05):

		self.activation_name = activation
		self.mutation_std = mutation_std
		self.rng = rng  # local RNG

		# xavier-style initialization
		limit1 = 1 / math.sqrt(input_size)
		limit2 = 1 / math.sqrt(hidden_size)

		self.W1 = np.array([[rng.uniform(-limit1, limit1)
							 for _ in range(input_size)]
							for _ in range(hidden_size)])

		self.b1 = np.zeros(hidden_size)

		self.W2 = np.array([[rng.uniform(-limit2, limit2)
							 for _ in range(hidden_size)]
							for _ in range(output_size)])

		self.b2 = np.zeros(output_size)

	def activation(self, x):
		if self.activation_name == "tanh":
			return np.tanh(x)
		elif self.activation_name == "sigmoid":
			return 1 / (1 + np.exp(-x))
		else:
			raise ValueError("Unsupported activation.")

	def forward(self, x):
		h = self.activation(self.W1 @ x + self.b1)
		out = self.W2 @ h + self.b2
		return out  # raw logits

	def mutate(self):
		def mutate_matrix(M):
			for i in range(M.shape[0]):
				for j in range(M.shape[1]):
					if self.rng.random() < 0.1:
						M[i, j] += self.rng.gauss(0, self.mutation_std)

		mutate_matrix(self.W1)
		mutate_matrix(self.W2)

	def flat(self):
		"""All weights as one float32 vector: W1, b1, W2, b2 (see from_flat)."""
		return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2]).astype(np.float32)

	@classmethod
	def from_flat(cls, flat, rng, input_size=32, hidden_size=16, output_size=8):
		"""Rebuild a network saved with flat(), e.g. from a snapshot."""
		nn = cls(rng, input_size, hidden_size, output_size)
		sizes = [hidden_size * input_size, hidden_size, output_size * hidden_size, output_size]
		parts = np.split(np.asarray(flat, dtype=float), np.cumsum(sizes)[:-1])
		nn.W1 = parts[0].reshape(hidden_size, input_size)
		nn.b1 = parts[1]
		nn.W2 = parts[2].reshape(output_size, hidden_size)
		nn.b2 = parts[3]
		return nn

class Grid:
	def __init__(self, width, height, metabolic_cost, min_child_energy, reproduction_cost, food_respawn_rate,
	num_agents, num_apples, num_oranges, num_walls=30, use_nn=False, regrowth="empty", food_capacity=None,
	inheritance="evolve", move_noise=0.0, seed=123):
		"""
		regrowth      : "empty"    - floor(food_respawn_rate * empty cells) food items spawn per tick
		                "logistic" - r * F * (1 - F / food_capacity) items, F = food items on the grid and
		                             r = food_respawn_rate (stochastically rounded, so small stocks can recover)
		inheritance   : how children of neural agents get their network
		                "evolve"      - uniform crossover of the parents, then mutation
		                "no_mutation" - crossover only, selection acts on the initial variation
		                "none"        - a fresh random network, no heritability and no adaptation
		move_noise    : chance that a neural agent makes a random move instead of its network's choice
		"""
		if regrowth not in ("empty", "logistic"):
			raise ValueError(f"unknown regrowth rule: {regrowth}")
		if regrowth == "logistic" and not food_capacity:
			raise ValueError("logistic regrowth needs food_capacity")
		if inheritance not in ("evolve", "no_mutation", "none"):
			raise ValueError(f"unknown inheritance: {inheritance}")
		if not 0 <= move_noise <= 1:
			raise ValueError(f"move_noise must be in [0, 1]: {move_noise}")

		self.width = width
		self.height = height
		self.metabolic_cost = metabolic_cost
		self.min_child_energy = min_child_energy
		self.reproduction_cost = reproduction_cost
		self.food_respawn_rate = food_respawn_rate
		self.grid = [[None for _ in range(width)] for _ in range(height)]
		self.agents = []
		self.food_items = []
		self.walls = []
		self.use_nn = use_nn
		self.regrowth = regrowth
		self.food_capacity = food_capacity
		self.inheritance = inheritance
		self.move_noise = move_noise
		self.rng = random.Random(seed)
		self.stats = dict.fromkeys(TICK_STATS, 0)

		self.populate(num_agents, num_apples, num_oranges, num_walls)

	def child_network(self, nn1, nn2):
		if self.inheritance == "none":
			return SimpleNN(self.rng)
		return self.cross_mutate(nn1, nn2, mutate=self.inheritance == "evolve")

	def cross_mutate(self, nn1, nn2, mutate=True):
		child = copy.deepcopy(nn1)
		# deepcopy also copied nn1's RNG - without this every child of nn1 would mutate
		# from the same frozen RNG state, and siblings would get identical mutations
		child.rng = self.rng

		for attr in ["W1", "W2"]:
			parent_matrix = getattr(nn2, attr)
			child_matrix = getattr(child, attr)

			mask = np.array([[self.rng.random() < 0.5
							for _ in range(child_matrix.shape[1])]
							for _ in range(child_matrix.shape[0])])

			child_matrix[mask] = parent_matrix[mask]

		if mutate:
			child.mutate()
		return child

	def get_empty_positions(self):
		return [(i, j) for i in range(self.height) for j in range(self.width) if self.grid[i][j] is None]

	def is_empty(self, x, y):
		return (0 <= x < self.height) and (0 <= y < self.width) and self.grid[x][y] is None

	def place_object(self, obj):
		self.grid[obj.x][obj.y] = obj

	def remove_object(self, obj):
		if self.grid[obj.x][obj.y] is obj:
			self.grid[obj.x][obj.y] = None

	def remove_agent(self, agent):
		# dead agents are dropped from self.agents at the end of the tick
		agent.alive = False
		self.remove_object(agent)

	def exhaust_agent(self, agent):
		# remove an agent whose energy ran out; its (non-positive) residual leaves the system
		self.stats["residual"] += agent.energy
		self.remove_agent(agent)

	def place_wall(self):
		directions = [(1, 0), (0, 1)]  # vertical, horizontal
		start_positions = self.get_empty_positions()
		if not start_positions:
			return

		start_x, start_y = self.rng.choice(start_positions)
		dir_x, dir_y = self.rng.choice(directions)
		length = self.rng.randint(2, 6)

		for i in range(length):
			nx, ny = start_x + i * dir_x, start_y + i * dir_y
			if not self.is_empty(nx, ny):
				break
			wall = Wall(nx, ny)
			self.place_object(wall)
			self.walls.append(wall)

	def populate(self, num_agents, num_apples, num_oranges, num_walls):
		for _ in range(num_walls):
			self.place_wall()

		total_needed = num_agents + num_apples + num_oranges
		empty_positions = self.get_empty_positions()

		if total_needed > len(empty_positions):
			raise ValueError("Not enough space to place all entities.")

		selected_positions = self.rng.sample(empty_positions, total_needed)
		index = 0

		for _ in range(num_agents):
			x, y = selected_positions[index]
			sex = self.rng.randint(0, 1)
			color = (0, 0, 1) if sex == 0 else (0, 0, 0.5)
			agent = Agent(x, y, sex, color)

			# artificial NN use
			if self.use_nn:
				agent.nn = SimpleNN(self.rng)

			self.place_object(agent)
			self.agents.append(agent)
			index += 1

		for _ in range(num_apples):
			x, y = selected_positions[index]
			apple = Apple(x, y)
			self.place_object(apple)
			self.food_items.append(apple)
			index += 1

		for _ in range(num_oranges):
			x, y = selected_positions[index]
			orange = Orange(x, y)
			self.place_object(orange)
			self.food_items.append(orange)
			index += 1

	def meet(self, agent1, agent2):
		if not (agent1.alive and agent2.alive):
			return

		# fighting scenario
		if agent1.sex == agent2.sex:
			if agent1.energy > agent2.energy:
				winner, loser = agent1, agent2
			elif agent1.energy < agent2.energy:
				winner, loser = agent2, agent1
			else:
				winner, loser = (agent1, agent2) if self.rng.random() < 0.5 else (agent2, agent1)

			winner.energy += loser.energy
			self.remove_agent(loser)
			self.stats["fights"] += 1
			return

		# mating scenario - both parents pay the reproduction cost,
		# children split the energy that is left
		agent1.energy -= self.reproduction_cost
		agent2.energy -= self.reproduction_cost
		self.stats["reproduction"] += 2 * self.reproduction_cost

		total_energy = agent1.energy + agent2.energy
		max_children = max(0, int(total_energy // self.min_child_energy))

		if max_children == 0:
			self.fail_mating(agent1, agent2)
			return

		# center around mid-point between parents
		cx = (agent1.x + agent2.x) // 2
		cy = (agent1.y + agent2.y) // 2

		R = 2  # reproduction radius
		possible_spots = [
			(cx + dx, cy + dy)
			for dx in range(-R, R + 1)
			for dy in range(-R, R + 1)
			if not (dx == 0 and dy == 0)
			and self.is_empty(cx + dx, cy + dy)
		]
		self.rng.shuffle(possible_spots)

		num_children = min(len(possible_spots), max_children)

		if num_children == 0:
			self.fail_mating(agent1, agent2)
			return

		energy_per_child = total_energy / num_children

		children_positions = possible_spots[:num_children]

		for pos in children_positions:
			x, y = pos
			sex = self.rng.randint(0, 1)
			color = (0, 0, 1) if sex == 0 else (0, 0, 0.5)
			child = Agent(x, y, sex, color, energy=energy_per_child)

			# artificial NN use
			if self.use_nn:
				child.nn = self.child_network(agent1.nn, agent2.nn)

			self.place_object(child)
			self.agents.append(child)

		# remove parents
		for parent in [agent1, agent2]:
			self.remove_agent(parent)

		self.stats["matings"] += 1
		self.stats["births"] += num_children

	def fail_mating(self, agent1, agent2):
		# no child fits (energy or space) - parents keep their reduced energy, or die if it ran out
		self.stats["failed_matings"] += 1
		for parent in [agent1, agent2]:
			if parent.energy <= 0:
				self.exhaust_agent(parent)

	def get_agent_vision(self, agent, vision_range=8):
		"""
		Ray-cast vision:
		For each of 8 directions:
			- Find first object within vision_range
			- Encode as [R, G, B, normalized_distance]

		If nothing is detected within range:
			return background white (1,1,1) with distance = 1.0

		Returns:
			np.ndarray of shape (8, 4)
		"""
		vision = []

		for dx, dy in directions:

			detected = False

			for r in range(1, vision_range + 1):
				nx = agent.x + dx * r
				ny = agent.y + dy * r

				# keep within bounds
				if not (0 <= nx < self.height and 0 <= ny < self.width):
					vision.append([0, 0, 0, r / vision_range])
					detected = True
					break

				obj = self.grid[nx][ny]

				if obj is not None:
					# object encoding
					R, G, B = obj.color
					vision.append([R, G, B, r / vision_range])
					detected = True
					break

			if not detected:
				# detected nothing within range
				vision.append([1, 1, 1, 1.0])

		return np.array(vision).flatten()

	def move_agent(self):
		"""
		Advance all agents by one tick.

		Every agent alive at the start of the tick acts exactly once - pays the metabolic
		cost, then moves - unless another agent removes it (fight or mating) before its turn.
		Children born during the tick act from the next tick on.

		self.stats holds this tick's counters. Agent energy changes only through them:

			ΔE_total = food - metabolism - reproduction - residual

		where residual is the (non-positive) energy of agents removed after running out.
		Fights and successful matings only move energy between agents.
		"""
		self.stats = dict.fromkeys(TICK_STATS, 0)
		self.rng.shuffle(self.agents)  # random move order

		for agent in list(self.agents):  # copy to avoid iteration issues
			if not agent.alive:
				continue  # removed earlier in this tick

			self.stats["acted"] += 1

			# decay mechanism
			agent.energy -= self.metabolic_cost
			self.stats["metabolism"] += self.metabolic_cost
			if agent.energy <= 0:
				self.stats["starved"] += 1
				self.exhaust_agent(agent)
				continue

			# artificial NN use - with probability move_noise a random move instead (the draw is
			# skipped when move_noise is 0, so runs without noise keep their random number sequence)
			if self.use_nn and not (self.move_noise and self.rng.random() < self.move_noise):
				vision = self.get_agent_vision(agent)
				agent.vision = vision
				dx, dy = agent.nn_output_to_move()
			else:
				# random movement (default behavior)
				dx, dy = self.rng.choice(directions)

			nx, ny = agent.x + dx, agent.y + dy

			# check bounds
			if not (0 <= nx < self.height and 0 <= ny < self.width):
				self.stats["blocked"] += 1
				continue

			# blocked by wall
			if isinstance(self.grid[nx][ny], Wall):
				self.stats["blocked"] += 1
				continue

			target = self.grid[nx][ny]

			# eat food
			if isinstance(target, Food):
				agent.energy += target.energy
				self.stats["food"] += target.energy
				self.stats["eaten"] += 1
				self.remove_object(target)

			# meet with another agent
			elif isinstance(target, Agent):
				self.meet(agent, target)

			# move only if alive and the target cell is free now (after a failed
			# mating the partner still occupies it)
			if agent.alive and self.grid[nx][ny] is None:
				self.remove_object(agent)
				agent.x, agent.y = nx, ny
				self.place_object(agent)

		self.agents = [a for a in self.agents if a.alive]
		self.food_items = [f for f in self.food_items if self.grid[f.x][f.y] is f]

	def respawn_food(self):
		empty = self.get_empty_positions()
		if not empty:
			return

		if self.regrowth == "logistic":
			stock = len(self.food_items)
			expected = self.food_respawn_rate * stock * (1 - stock / self.food_capacity)
			n = int(expected) + (self.rng.random() < expected - int(expected)) if expected > 0 else 0
			n = min(n, len(empty))
		else:
			n = int(len(empty) * self.food_respawn_rate)

		for x, y in self.rng.sample(empty, n):
			if self.rng.random() < 0.6:
				food = Apple(x, y)
			else:
				food = Orange(x, y)

			self.place_object(food)
			self.food_items.append(food)

	def total_energy(self):
		return sum(agent.energy for agent in self.agents)

	def render(self, label=None):
		"""Draw the grid and return the figure (shown/saved by the caller)."""
		color_grid = np.ones((self.height, self.width, 3))  # white background

		for wall in self.walls:
			color_grid[wall.x][wall.y] = wall.color

		for agent in self.agents:
			color_grid[agent.x][agent.y] = agent.color

		for food in self.food_items:
			color_grid[food.x][food.y] = food.color

		num_apples = sum(isinstance(f, Apple) for f in self.food_items)
		num_oranges = sum(isinstance(f, Orange) for f in self.food_items)

		title = (
			f"Agents: {len(self.agents)} | "
			f"Total agents energy: {self.total_energy():.2f} | "
			f"Apples: {num_apples} | "
			f"Oranges: {num_oranges} | "
			f"Walls: {len(self.walls)}"
		)
		if label:
			title = f"{label}\n{title}"

		fig, ax = plt.subplots(figsize=(10, 10))
		ax.imshow(color_grid, interpolation='nearest')
		ax.set_title(title)
		ax.axis('off')
		return fig


def shannon_entropy(grid, energy_eps=6):
	"""
	Compute the coarse-grained Shannon entropy of the grid state.

	This entropy is a macroscopic, discretized classification of the system.
	Agents are grouped into categorical states based on sex and an energy
	threshold (HighE / LowE). Food types, walls, and empty cells are classified discretely.

	Shannon entropy measures the structural diversity at a coarse ecological level, while the Lyapunov
	exponent measures microscopic trajectory sensitivity in continuous state space.

	The entropy is computed as:

		H = -Σ (p_i * log2(p_i))

	where p_i is the probability of each discrete cell state.

	Returns
	-------
	float
		Shannon entropy of the discretized grid configuration.
	"""
	states = []
	for row in grid.grid:
		for cell in row:
			if cell is None:
				states.append("Empty")
			elif isinstance(cell, Agent):
				sex = "M" if cell.sex == 0 else "F"
				energy_class = "HighE" if cell.energy >= energy_eps else "LowE"

				states.append(f"Agent_{sex}_{energy_class}")
			elif isinstance(cell, Apple):
				states.append("Apple")
			elif isinstance(cell, Orange):
				states.append("Orange")
			elif isinstance(cell, Wall):
				states.append("Wall")

	counts = Counter(states)
	total = sum(counts.values())
	probs = [c / total for c in counts.values()]
	entropy = -sum(p * math.log2(p) for p in probs if p > 0)
	return entropy


def grid_difference(g1, g2):
	"""
	Compute the normalized continuous structural difference between two grids.

	This metric operates in continuous phase space:
	- Agent energy differences are treated continuously (normalized magnitude).
	- Sex mismatch contributes discretely.
	- Entity type mismatch contributes discretely.

	This is intentionally different from Shannon entropy as:

	- Lyapunov exponent → continuous phase-space metric
	- Shannon entropy → discretized macroscopic classification

	The returned value represents a normalized phase-space distance
	between two system configurations.

	Returns
	-------
	float
		Normalized difference in the range [0, 1] (approximately).
	"""
	diff = 0.0
	total = g1.width * g1.height

	for i in range(g1.height):
		for j in range(g1.width):
			c1, c2 = g1.grid[i][j], g2.grid[i][j]

			if (c1 is None) != (c2 is None):
				diff += 1.0

			elif (c1 is not None) and (c2 is not None):

				if type(c1) != type(c2):
					diff += 1.0

				elif isinstance(c1, Agent):
					# continuous energy difference
					energy_scale = max(abs(c1.energy), abs(c2.energy), 1.0)
					energy_term = abs(c1.energy - c2.energy) / energy_scale

					sex_term = 0.0 if c1.sex == c2.sex else 1.0

					diff += 0.5 * sex_term + 0.5 * energy_term

	return diff / total


def population_stats(pop, energy, initial_agents, start):
	"""
	Outcome and population/energy statistics of one grid.

	Parameters
	----------
	pop, energy : sequence
		Population size and total agent energy at ticks 0..T (index = tick).
	initial_agents : int
		Starting population.
	start : int
		First tick of the post burn-in window used for the mean values.

	The outcome is decided at the final tick: extinct (N = 0), declined (0 < N < initial)
	or established (N >= initial, the same as "population preserved").
	"""
	pop = np.asarray(pop)
	energy = np.asarray(energy, dtype=float)
	final_population = int(pop[-1])

	if final_population == 0:
		outcome = "extinct"
	elif final_population < initial_agents:
		outcome = "declined"
	else:
		outcome = "established"

	late_pop, late_energy = pop[start:], energy[start:]
	alive = late_pop > 0

	return {
		"outcome": outcome,
		"extinction_tick": int(np.argmax(pop == 0)) if outcome == "extinct" else None,
		"final_population": final_population,
		"final_energy": float(energy[-1]),
		"final_energy_per_agent": float(energy[-1] / final_population) if final_population else float("nan"),
		"peak_population": int(pop.max()),
		"peak_population_tick": int(pop.argmax()),
		"peak_energy": float(energy.max()),
		"peak_energy_tick": int(energy.argmax()),
		"mean_population": float(late_pop.mean()),
		"mean_energy": float(late_energy.mean()),
		"mean_energy_per_agent": float(np.mean(late_energy[alive] / late_pop[alive])) if alive.any() else float("nan"),
	}


def divergence_stats(d, d_ref, cells):
	"""
	Timing of the divergence d(t) between two grids.

	d     : d(t) at ticks 0..T
	d_ref : saturation level - the late distance of two independently driven grids
	cells : number of grid cells

	Each grid draws all randomness from one RNG, so the first event the perturbation changes
	(a fight, a death, a different number of children) shifts every later draw. From that tick on
	food spawns and moves differ across the whole grid, and d(t) jumps by orders of magnitude in
	a single tick instead of growing exponentially.

	desync_tick : first tick with at least 10 cells differing (None if it never happens)
	half_tick   : first tick with d >= d_ref / 2 (None if it never happens)
	lyapunov    : finite-time exponent ln(d(t_end) / d(0)) / t_end over the rise only,
	              t_end = half_tick, or T if d never gets there
	"""
	d = np.asarray(d)
	desync = np.flatnonzero(d >= 10 / cells)
	half = np.flatnonzero(d >= d_ref / 2) if d_ref > 0 else []
	desync_tick = int(desync[0]) if len(desync) else None
	half_tick = int(half[0]) if len(half) and half[0] > 0 else None
	t_end = half_tick or len(d) - 1
	return {
		"desync_tick": desync_tick,
		"half_tick": half_tick,
		"lyapunov": float(np.log(d[t_end] / d[0]) / t_end),
	}


def event_rates(series, start):
	"""
	Events per agent and tick over ticks start..T, from the per-tick counters of one grid.
	Ratio of sums: total events / total agent-ticks ("acted"), so busy ticks weigh more.
	food is energy eaten per agent and tick, the others are counts.
	"""
	acted = np.sum(series["acted"][start:])
	return {
		f"rate_{key}": float(np.sum(series[key][start:]) / acted) if acted else float("nan")
		for key in RATE_STATS
	}


class Snapshots:
	"""
	Agent snapshots taken during compare_grids runs, saved to snapshots.npz.

	periodic : every `every` ticks, starting at tick 0 - positions, sex and energy of all agents,
	           networks of the first `max_networks` (the tick shuffles the agent list, so these
	           are a random sample)
	endgame  : once a grid has reached `established` agents and then falls below
	           ENDGAME_POPULATION, everything about every agent, every ENDGAME_EVERY ticks until it
	           recovers to RECOVERED_POPULATION or dies out (at most MAX_ENDGAME per grid)

	Arrays (one row per snapshot / per agent / per saved network):
	snap_meta     : run, grid, tick, kind (0 periodic, 1 endgame), population, food items
	agent_data    : snapshot row, x, y, sex, energy
	agent_network : row in networks, -1 if not saved
	networks      : SimpleNN.flat() vectors
	"""
	PERIODIC, ENDGAME = 0, 1
	ENDGAME_POPULATION = 20
	RECOVERED_POPULATION = 50
	ENDGAME_EVERY = 100
	MAX_ENDGAME = 100

	def __init__(self, every, max_networks, established):
		self.every = every
		self.max_networks = max_networks
		self.established = established
		self.run = 0
		self.meta, self.agents, self.agent_net, self.networks = [], [], [], []
		self.state = {}

	def start_run(self, run):
		self.run = run
		self.state = {}  # grid -> {"established", "endgame_since", "endgames"}

	def take(self, g, grid, tick, kind, max_networks):
		snap = len(self.meta)
		self.meta.append((self.run, grid, tick, kind, len(g.agents), len(g.food_items)))
		for i, agent in enumerate(g.agents):
			nn = getattr(agent, "nn", None)
			net = -1
			if nn is not None and i < max_networks:
				net = len(self.networks)
				self.networks.append(nn.flat())
			self.agents.append((snap, agent.x, agent.y, agent.sex, agent.energy))
			self.agent_net.append(net)

	def update(self, g, grid, tick):
		n = len(g.agents)
		st = self.state.setdefault(grid, {"established": False, "endgame_since": None, "endgames": 0})
		st["established"] |= n >= self.established

		if tick % self.every == 0:
			self.take(g, grid, tick, self.PERIODIC, self.max_networks)

		if st["endgame_since"] is None:
			if st["established"] and 0 < n < self.ENDGAME_POPULATION:
				st["endgame_since"] = tick
		elif n == 0 or n >= self.RECOVERED_POPULATION:
			st["endgame_since"] = None
		if (st["endgame_since"] is not None and (tick - st["endgame_since"]) % self.ENDGAME_EVERY == 0
				and st["endgames"] < self.MAX_ENDGAME):
			self.take(g, grid, tick, self.ENDGAME, n)
			st["endgames"] += 1

	def arrays(self):
		return {
			"snap_meta": np.array(self.meta, dtype=np.int64).reshape(-1, 6),
			"agent_data": np.array(self.agents, dtype=np.float64).reshape(-1, 5),
			"agent_network": np.array(self.agent_net, dtype=np.int32),
			# random agents have no networks: keep the 2-D shape of SimpleNN.flat() rows
			"networks": np.array(self.networks, dtype=np.float32) if self.networks else np.zeros((0, 664), np.float32),
		}


def new_series():
	return {key: [] for key in SERIES}


def record(g, series):
	series["population"].append(len(g.agents))
	series["energy"].append(g.total_energy())
	series["sex1"].append(sum(a.sex == 1 for a in g.agents))
	series["food_items"].append(len(g.food_items))
	for key in TICK_STATS:
		series[key].append(g.stats[key])


def main_analysis(g1, g2, num_ticks=50, render=False, trial=1, g3=None, results=None, snapshots=None, **grid_params):
	"""
	Run two nearly identical grids in lockstep and measure how they diverge, compute the Shannon
	entropy difference, and calculate basic regime stability metrics.

	Methodology
	-----------
	- Divergence: d(t) is the phase-space distance between g1 and g2 (grid_difference).
	  The perturbation stays latent until the first event it changes, then the grids' RNG
	  streams desynchronize and d(t) jumps (see divergence_stats). Reported are the desync tick,
	  the tick at which d(t) reaches half its saturation level, and the finite-time Lyapunov
	  exponent over that rise.

	- Baseline: g3 starts identical to g1 but draws from a different RNG stream. d_13(t) is
	  the divergence of two runs that share nothing but the initial state. If the perturbed pair
	  ends up as far apart as the baseline pair, the perturbation is indistinguishable from
	  re-seeding.

	- Shannon entropy difference ΔH is computed at the final tick as a macroscopic measure
	  of structural divergence between the two grids. Agents are categorized by sex and energy
	  thresholds, food types, walls, and empty cells. The entropy is computed as:
		  H = -Σ (p_i * log2(p_i))
	  where p_i is the probability of each discrete cell state.

	- Population dynamics metrics are recorded to assess regime stability:
		- Mean log-growth rate of population
		- Coefficient of variation (CV)
		- Viability (both sexes present)
		- Population preservation (final population >= initial population)

	Parameters
	----------
	g1, g2 : Grid
		Two Grid instances with nearly identical initial states (g2 can have a small perturbation).
	num_ticks : int, optional
		Number of simulation ticks. Default 50.
	render : bool, optional
		If True, renders initial and final grid states. Default False.
	trial : int, optional
		Trial index for display purposes. Default 1.
	g3 : Grid or None, optional
		Baseline grid: same initial state as g1, different RNG stream. None skips the baseline.
	snapshots : Snapshots or None, optional
		Records agent snapshots of g1 and g2 (grid 1 and 2) during the run.
	**grid_params : dict
		Additional grid parameters required for entropy computation and population metrics.

	Returns
	-------
	dict
		Dictionary containing the following metrics:
		- "lyap" : float
			Finite-time Lyapunov exponent over the rise of d(t).
		- "desync_tick", "half_tick" : int or None
			See divergence_stats.
		- "d_late", "baseline_d_late" : float
			Mean d(t) after burn-in of the perturbed pair and of the baseline pair (nan without g3).
		- "diffs", "baseline_diffs" : list of float
			d(t) of the perturbed pair and of the baseline pair (None without g3).
		- "shannon_diff" : float
			Shannon entropy difference ΔH between grids at final tick.
		- "r" : float
			Estimated mean log-growth rate of population (after burn-in).
		- "cv" : float
			Coefficient of variation of population size (after burn-in).
		- "viable" : bool
			True if both sexes are present in final population.
		- "population_preserved" : bool
			True if final population >= initial population.
		- "final_population" : int
			Population size at the final tick.
		- "grids" : list of dict
			population_stats and event_rates of grid 1 and grid 2.
		- "baseline" : dict or None
			population_stats of g3.
		- "series" : list of dict
			Per-tick series of grid 1 and grid 2 (keys in SERIES) at ticks 0..T.
		- "baseline_population" : list of int or None
			Population of g3 at ticks 0..T.
	"""
	results = results or Results(None, root=None)
	run = f"run{trial+1:02d}"
	cells = g1.width * g1.height

	series = (new_series(), new_series())
	for g, s in zip((g1, g2), series):
		record(g, s)
	if snapshots is not None:
		snapshots.start_run(trial + 1)
		for grid, g in enumerate((g1, g2), start=1):
			snapshots.update(g, grid, 0)
	diffs = [max(grid_difference(g1, g2), 1e-12)]

	if g3 is not None:
		baseline_pop = [len(g3.agents)]
		baseline_energy = [g3.total_energy()]
		baseline_diffs = [max(grid_difference(g1, g3), 1e-12)]

	def render_states(stage):
		# saved whenever there is an output folder, displayed only with render=True
		if not (render or results.dir):
			return
		for i, g in enumerate((g1, g2), start=1):
			fig = g.render(f"Run {trial+1:02d} - grid {i} - {stage}")
			results.show(fig, "states", f"{run}_grid{i}_{stage}", display=render)
			if render:
				results.log(f"Grid {i} - {stage.capitalize()} Shannon entropy: {shannon_entropy(g, grid_params['min_child_energy'])}")

	render_states("initial")

	for t in range(num_ticks):
		for g, s in zip((g1, g2), series):
			g.move_agent()
			g.respawn_food()
			record(g, s)
		if snapshots is not None:
			for grid, g in enumerate((g1, g2), start=1):
				snapshots.update(g, grid, t + 1)
		diffs.append(max(grid_difference(g1, g2), 1e-12))

		if g3 is not None:
			g3.move_agent()
			g3.respawn_food()
			baseline_pop.append(len(g3.agents))
			baseline_energy.append(g3.total_energy())
			baseline_diffs.append(max(grid_difference(g1, g3), 1e-12))

	pops = [s["population"] for s in series]
	pop_history = pops[0][1:]  # grid 1, ticks 1..T

	render_states("final")

	burn_frac = 0.3
	start = int(burn_frac * num_ticks)

	# --- divergence ---
	d_late = float(np.mean(diffs[start + 1:]))
	baseline_d_late = float(np.mean(baseline_diffs[start + 1:])) if g3 is not None else float("nan")
	divergence = divergence_stats(diffs, baseline_d_late if g3 is not None else d_late, cells)

	# --- entropy ---
	g1_H = shannon_entropy(g1, grid_params['min_child_energy'])
	g2_H = shannon_entropy(g2, grid_params['min_child_energy'])
	shannon_diff = abs(g2_H - g1_H)

	# --- regime metrics ---
	pop_history = np.array(pop_history)
	g = g1

	if pop_history[-1] == 0:
		viable = 0
	else:
		viable = any(a.sex == 1 for a in g.agents) and any(a.sex == 0 for a in g.agents)

	population_preserved = pop_history[-1] >= grid_params['num_agents']

	pop_history_safe = np.maximum(pop_history, 1)

	t_vals = np.arange(start, num_ticks)

	logN = np.log(pop_history_safe[start:])

	r = float("nan")
	if len(logN) > 5:
		r, _ = np.polyfit(t_vals, logN, 1)

	meanN = np.mean(pop_history[start:])
	stdN = np.std(pop_history[start:])
	cv = stdN / meanN if meanN > 0 else float("inf")

	# --- population, energy and event stats of all grids (same window as r and CV) ---
	grid_stats = [
		{**population_stats(s["population"], s["energy"], grid_params['num_agents'], start + 1),
		 **event_rates(s, start + 1)}
		for s in series
	]
	baseline_stats = None
	if g3 is not None:
		baseline_stats = population_stats(baseline_pop, baseline_energy, grid_params['num_agents'], start + 1)

	def tick(t):
		return "never" if t is None else str(t)

	results.log(
		f"[Run {trial+1:02d}] "
		f"λ = {divergence['lyapunov']:.5f} | "
		f"desync @ {tick(divergence['desync_tick'])} | "
		f"d½ @ {tick(divergence['half_tick'])} | "
		f"d_late = {d_late:.3f}"
		+ (f" (baseline {baseline_d_late:.3f})" if g3 is not None else "")
		+ f" | ΔH = {shannon_diff:.5f} | "
		f"grid 1: {grid_stats[0]['outcome']}, grid 2: {grid_stats[1]['outcome']}"
		+ (f", baseline: {baseline_stats['outcome']}" if g3 is not None else "")
	)

	# --- divergence trajectory plot, with both populations on a second axis ---
	fig, ax = plt.subplots(figsize=(7, 4))
	ax.plot(range(len(diffs)), diffs, label='d(t) perturbed')
	if g3 is not None:
		ax.plot(range(len(baseline_diffs)), baseline_diffs, color='tab:gray', linewidth=1, label='d(t) baseline (reseeded)')
	if divergence["desync_tick"] is not None:
		ax.axvline(x=divergence["desync_tick"], color='red', linestyle='--', label=f'Desync (tick {divergence["desync_tick"]})')
	ax.set_xlabel('Tick')
	ax.set_ylabel('Phase-space distance d(t)')
	ax.set_title(f'Divergence Run {trial+1}')
	ax.grid(True)

	ax_pop = ax.twinx()
	ax_pop.plot(pops[0], color='tab:green', alpha=0.6, linewidth=1, label='N(t) grid 1')
	ax_pop.plot(pops[1], color='tab:purple', alpha=0.6, linewidth=1, linestyle='--', label='N(t) grid 2')
	ax_pop.set_ylabel('Population N(t)')

	handles, labels = ax.get_legend_handles_labels()
	pop_handles, pop_labels = ax_pop.get_legend_handles_labels()
	ax.legend(handles + pop_handles, labels + pop_labels, fontsize='small')
	results.show(fig, "plots", f"divergence_{run}")

	return {
		"lyap": divergence["lyapunov"],
		"desync_tick": divergence["desync_tick"],
		"half_tick": divergence["half_tick"],
		"d_late": d_late,
		"baseline_d_late": baseline_d_late,
		"diffs": diffs,
		"baseline_diffs": baseline_diffs if g3 is not None else None,
		"shannon_diff": shannon_diff,
		"r": r,
		"cv": cv,
		"viable": viable,
		"population_preserved": population_preserved,
		"final_population": int(pop_history[-1]),
		"grids": grid_stats,
		"baseline": baseline_stats,
		"series": series,
		"baseline_population": baseline_pop if g3 is not None else None,
	}


def split_rate_if_independent(a, b):
	"""Chance that two grids end with different outcomes if their fates were independent."""
	pa, pb = Counter(a), Counter(b)
	return 1 - sum(pa[o] * pb[o] for o in OUTCOMES) / (len(a) * len(b))


def population_summary(runs, results, baseline=None):
	"""
	Print outcome counts, pooled population/energy stats, event rates and a per-run table.

	runs     : list of (run, grid, stats) with stats from population_stats and event_rates.
	baseline : {run: population_stats} of the reseeded baseline grid, or None.
	Means pool the established grids of both grid 1 and grid 2.
	"""
	log = results.log
	log("\n----- POPULATION SUMMARY -----")

	for grid in (1, 2):
		counts = Counter(s["outcome"] for _, g, s in runs if g == grid)
		log(f"{f'Outcomes (grid {grid})':45s}: " + ", ".join(f"{counts[o]} {o}" for o in OUTCOMES))
	if baseline:
		counts = Counter(s["outcome"] for s in baseline.values())
		log(f"{'Outcomes (baseline, reseeded grid 1)':45s}: " + ", ".join(f"{counts[o]} {o}" for o in OUTCOMES))

	outcomes = {}
	for run, grid, s in runs:
		outcomes.setdefault(run, {})[grid] = s["outcome"]
	same = sum(o[1] == o[2] for o in outcomes.values())
	log(f"{'Outcome agreement (grid 1 / grid 2)':45s}: {same} same, {len(outcomes) - same} split")
	# if the perturbed split rate is close to this, the perturbation acts like a new seed
	expected = split_rate_if_independent([o[1] for o in outcomes.values()], [o[2] for o in outcomes.values()])
	log(f"{'  split rate if fates were independent':45s}: {expected:.2f} (observed {(len(outcomes) - same) / len(outcomes):.2f})")
	if baseline:
		same_b = sum(outcomes[run][1] == s["outcome"] for run, s in baseline.items())
		log(f"{'Outcome agreement (grid 1 / baseline)':45s}: {same_b} same, {len(baseline) - same_b} split")

	extinction_ticks = [s["extinction_tick"] for _, _, s in runs if s["outcome"] == "extinct"]
	if extinction_ticks:
		log(f"{'Median extinction tick':45s}: {np.median(extinction_ticks):.0f} ({len(extinction_ticks)} grids)")

	established = [s for _, _, s in runs if s["outcome"] == "established"]
	log(f"Established grids, both grids pooled (n={len(established)}), mean ± sd:" if established
		else "Established grids                            : none")
	for label, key in (
		("Population (after burn-in)", "mean_population"),
		("Total energy (after burn-in)", "mean_energy"),
		("Energy per agent (after burn-in)", "mean_energy_per_agent"),
		("Peak population", "peak_population"),
		("Peak total energy", "peak_energy"),
	):
		values = [s[key] for s in established]
		if values:
			log(f"  {label:43s}: {np.mean(values):.1f} ± {np.std(values):.1f}")
	if established:
		log("  Per agent and tick (after burn-in):")
		for key in RATE_STATS:
			values = [s[f"rate_{key}"] for s in established]
			label = "food energy eaten" if key == "food" else key.replace("_", " ")
			log(f"    {label:41s}: {np.mean(values):.5f} ± {np.std(values):.5f}")

	# per-run table, * marks the largest value of the column (no mark if every row ties)
	maxima = {}
	for key in ("final_population", "peak_population", "final_energy", "peak_energy"):
		values = [s[key] for _, _, s in runs]
		if values.count(max(values)) < len(values):
			maxima[key] = max(values)

	def cell(s, key, tick_key=None, fmt="{:.0f}"):
		text = fmt.format(s[key])
		if tick_key:
			text += f" ({s[tick_key]})"
		return text + ("*" if s[key] == maxima.get(key) else " ")

	log(f"\n{'run':>3} {'grid':>4}  {'outcome':<16}{'N_final':>9}{'N_peak (tick)':>16}{'E_final':>11}{'E_peak (tick)':>18}{'E/N_final':>11}")
	for run, grid, s in runs:
		outcome = f"extinct @{s['extinction_tick']}" if s["outcome"] == "extinct" else s["outcome"]
		per_agent = f"{s['final_energy_per_agent']:.1f}" if s["final_population"] else "-"
		log(
			f"{run:>3} {grid:>4}  {outcome:<16}"
			f"{cell(s, 'final_population'):>9}{cell(s, 'peak_population', 'peak_population_tick'):>16}"
			f"{cell(s, 'final_energy'):>11}{cell(s, 'peak_energy', 'peak_energy_tick'):>18}"
			f"{per_agent:>10}"
		)
	log("* largest value in the column")


def compare_grids(num_ticks=50, num_perturbed_agents=1, seed=123, final_render=True, lyapunov_final_render=True, num_trials=30, baseline=True, results=None, snapshot_every=None, snapshot_networks=100, **grid_params):
	"""
	Compare two nearly identical grid simulations to measure their divergence,
	Shannon entropy and aggregate regime statistics over multiple trials.

	Parameters
	----------
	num_ticks : int, optional
		Number of simulation ticks to run. Default is 50.
	num_perturbed_agents : int, optional
		Number of agents to be perturbated (by adding energy).
	seed : int, optional
		Random seed ensuring deterministic setup. Default is 123.
	baseline : bool, optional
		Also run a third grid per trial, identical to grid 1 but with a different RNG stream
		(see main_analysis). Costs 50% more run time. Default is True.
	snapshot_every : int or None, optional
		Save agent snapshots (positions, networks) every this many ticks, plus endgame
		snapshots of collapsing populations, to snapshots.npz (see Snapshots). None saves none.
	snapshot_networks : int, optional
		Networks saved per periodic snapshot and grid. Default 100.

	Returns
	-------
	mean_lambda : float
		Mean finite-time Lyapunov exponent over the rise of d(t).
	mean_H : float
		Shannon entropy difference between two grids.
	"""
	results = results or Results(None, root=None)
	log = results.log

	run_results = []
	grid_runs = []  # (run, grid, population_stats + event_rates)
	baseline_runs = {}  # run -> population_stats of the baseline grid
	snapshots = Snapshots(snapshot_every, snapshot_networks, 2 * grid_params["num_agents"]) if snapshot_every else None

	for trial in range(num_trials):
		g1 = Grid(**grid_params, seed=seed + trial)
		g2 = Grid(**grid_params, seed=seed + trial)

		if g2.agents:
			local_rng = random.Random(seed + 999 + trial)
			k = min(num_perturbed_agents, len(g2.agents))
			perturbed = local_rng.sample(g2.agents, k)
			for agent in perturbed:
				agent.energy += g2.min_child_energy

		g3 = None
		if baseline:
			# same initial state as g1, then a different RNG stream. Reseeded in place:
			# the initial networks hold a reference to this RNG object
			g3 = Grid(**grid_params, seed=seed + trial)
			g3.rng.seed(BASELINE_SEED_OFFSET + seed + trial)

		result = main_analysis(
			g1, g2, num_ticks, lyapunov_final_render, trial, g3, results=results, snapshots=snapshots, **grid_params
		)
		run_results.append(result)
		for grid, stats in enumerate(result["grids"], start=1):
			grid_runs.append((trial + 1, grid, stats))
		if baseline:
			baseline_runs[trial + 1] = result["baseline"]

	def column(key):
		return [res[key] for res in run_results]

	# --- divergence & entropy ---
	lambdas = column("lyap")
	mean_lambda = np.mean(lambdas)
	mean_H = np.mean(column("shannon_diff"))

	log("\n----- SUMMARY -----")
	log(f"{'Lyapunov exponent (rise of d(t))':37s}: {mean_lambda:.6f} ± {np.std(lambdas):.6f}")
	for label, key in (("Desync tick", "desync_tick"), ("Half-saturation tick", "half_tick")):
		ticks = [t for t in column(key) if t is not None]
		reached = f"{np.median(ticks):.0f} (range {min(ticks)}-{max(ticks)})" if ticks else "-"
		log(f"{label + ', median':37s}: {reached}, reached in {len(ticks)}/{num_trials} runs")
	log(f"{'Late distance d (after burn-in)':37s}: {np.mean(column('d_late')):.4f} ± {np.std(column('d_late')):.4f}")
	if baseline:
		log(f"{'  baseline (reseeded, unperturbed)':37s}: {np.mean(column('baseline_d_late')):.4f} ± {np.std(column('baseline_d_late')):.4f}")
	log(f"{'Shannon entropy ΔH':37s}: {mean_H:.6f} ± {np.std(column('shannon_diff')):.6f}")

	# --- regime stats ---
	r_vals = column("r")
	cv_vals = column("cv")
	valid_cvs = [cv for cv in cv_vals if np.isfinite(cv)]

	log("\n----- REGIME SUMMARY -----")
	log(f"{'Mean log population growth rate':45s}: {np.mean(r_vals):.6f}")
	log(f"{'Standard deviation of log growth rate':45s}: {np.std(r_vals):.6f}")
	# runs extinct before the burn-in ends have no defined CV and are left out
	log(f"{'Mean coefficient of variation':45s}: {np.mean(valid_cvs):.6f} ({len(valid_cvs)}/{len(cv_vals)} runs)")
	log(f"{'Viability preservation probability':45s}: {np.mean(column('viable')):.3f}")
	log(f"{'Population preservation probability':45s}: {np.mean(column('population_preserved')):.3f}")

	population_summary(grid_runs, results, baseline_runs or None)

	run_columns = ("lyap", "desync_tick", "half_tick", "d_late", "baseline_d_late", "shannon_diff")
	results.save_csv("runs.csv", [
		{
			"run": run, "grid": grid,
			**{("lyapunov" if key == "lyap" else key): run_results[run - 1][key] for key in run_columns},
			**stats,
			**({"baseline_outcome": baseline_runs[run]["outcome"]} if baseline else {}),
		}
		for run, grid, stats in grid_runs
	])

	def stacked(key):
		# (runs, 2 grids, ticks + 1); counts as int32, energies as float32 to keep the file small
		values = np.array([[s[key] for s in res["series"]] for res in run_results])
		return values.astype(np.float32 if values.dtype.kind == "f" else np.int32)

	arrays = {key: stacked(key) for key in SERIES}
	arrays["d"] = np.array(column("diffs"))  # (runs, ticks + 1)
	if baseline:
		arrays["baseline_d"] = np.array(column("baseline_diffs"))                 # (runs, ticks + 1)
		arrays["baseline_population"] = np.array(column("baseline_population"))  # (runs, ticks + 1)
	results.save_arrays("timeseries.npz", **arrays)
	if snapshots is not None:
		results.save_arrays("snapshots.npz", **snapshots.arrays())

	return mean_lambda, mean_H


def check_determinism(num_ticks, seed, debug_render=False, final_render=True, results=None, **grid_params):
	"""
	Verify whether the simulation environment is fully deterministic.

	Runs two simulations with identical initial conditions and checks
	whether the final grid states are identical.

	Returns
	-------
	bool
		True if deterministic, False otherwise.
	"""
	results = results or Results(None, root=None)
	log = results.log
	log("\n----- DETERMINISTIC CHECK -----")

	g1 = Grid(**grid_params, seed=seed)
	g2 = Grid(**grid_params, seed=seed)

	def render_states(stage):
		# saved whenever there is an output folder, displayed only with final_render=True
		if not (final_render or results.dir):
			return
		for i, g in enumerate((g1, g2), start=1):
			fig = g.render(f"Determinism check - grid {i} - {stage}")
			results.show(fig, "states", f"determinism_grid{i}_{stage}", display=final_render)
			if final_render:
				log(f"Grid {i} - {stage.capitalize()} Shannon entropy: {shannon_entropy(g, grid_params['min_child_energy'])}")

	render_states("initial")

	for t in range(num_ticks):
		g1.move_agent()
		g1.respawn_food()

		if debug_render == True:
			results.show(g1.render(f"Determinism check - grid 1 - tick {t}"))
			results.show(g2.render(f"Determinism check - grid 2 - tick {t}"))

		g2.move_agent()
		g2.respawn_food()

	render_states("final")

	diff = grid_difference(g1, g2)

	log(
		"Difference between two grids with identical initial conditions "
		f"(expected 0.0): {diff}"
	)

	if diff > 0:
		log("Determinism check FAILED.")
		return False

	log("Determinism check PASSED.")
	log("--------------------------------------")
	return True


def main_simulation(num_runs, num_ticks, num_prtrb_agents, init_seed, baseline=True, snapshot_every=None, debug_render=False, final_render=False, lyapunov_final_render=False, name="experiment", results_dir="results", **grid_params):
	"""
	Execute the main experiment pipeline.

	This function:
	1. Runs a single grid simulation to observe entropy evolution.
	2. Runs a Lyapunov analysis to estimate system sensitivity to perturbations.

	Parameters
	----------
	num_ticks : int, optional
		Number of simulation ticks to run.
	num_prtrb_agents : int, optional
		Number of agents to be perturbated (by adding energy).
	init_seed : int, optional
		Random seed to ensure deterministic behavior.
	baseline : bool, optional
		Run a reseeded, unperturbed copy of grid 1 per run as a null model (see compare_grids).
	snapshot_every : int or None, optional
		Save agent snapshots every this many ticks (see compare_grids). None saves none.
	name : str, optional
		Experiment name, used for the output folder.
	results_dir : str or None, optional
		Output is saved to <results_dir>/<timestamp>_<name>/ (see Results). None saves nothing.

	Returns
	-------
	Results
		Output of the experiment; Results.dir is the output folder.
	"""
	results = Results(name, root=results_dir)
	log = results.log
	results.save_json("params.json", {
		"name": name, "num_runs": num_runs, "num_ticks": num_ticks, "num_prtrb_agents": num_prtrb_agents,
		"init_seed": init_seed, "baseline": baseline, "snapshot_every": snapshot_every, **grid_params
	})

	log("\n----- PARAMETERS LIST -----")
	log(f"Number of runs              : {num_runs}")
	log(f"Simulation ticks            : {num_ticks}")
	log(f"Perturbed agents            : {num_prtrb_agents}")
	log(f"Initial random seed         : {init_seed}")
	log(f"Baseline (reseeded) grid    : {'yes' if baseline else 'no'}")
	log(f"Grid width                  : {grid_params['width']}")
	log(f"Grid height                 : {grid_params['height']}")
	log(f"Metabolic cost              : {grid_params['metabolic_cost']}")
	log(f"Minimum child energy        : {grid_params['min_child_energy']}")
	log(f"Reproduction cost           : {grid_params['reproduction_cost']}")
	log(f"Food respawn rate           : {grid_params['food_respawn_rate']}")
	log(f"Initial number of agents    : {grid_params['num_agents']}")
	log(f"Initial number of apples    : {grid_params['num_apples']}")
	log(f"Initial number of oranges   : {grid_params['num_oranges']}")
	log(f"Number of walls             : {grid_params['num_walls']}")
	log(f"Agent decision mechanism    : {'Neural Network' if grid_params['use_nn'] else 'Random'}")
	regrowth = grid_params.get('regrowth', 'empty')
	log(f"Food regrowth               : {regrowth}"
		+ (f" (capacity {grid_params['food_capacity']})" if regrowth == "logistic" else ""))
	if grid_params['use_nn']:
		log(f"Network inheritance         : {grid_params.get('inheritance', 'evolve')}")
		log(f"Move noise                  : {grid_params.get('move_noise', 0.0)}")
	if results.dir:
		log(f"Output folder               : {results.dir}")
	log("--------------------------------------")

	is_deterministic = check_determinism(num_ticks=num_ticks,seed=init_seed,debug_render=debug_render,final_render=final_render,results=results,**grid_params)

	if not is_deterministic:
		raise RuntimeError("Simulation environment is non-deterministic.")

	log("\n----- DIVERGENCE COMPARISON -----")
	compare_grids(
		num_ticks, num_prtrb_agents, init_seed, final_render, lyapunov_final_render, num_runs, baseline,
		results=results, snapshot_every=snapshot_every, **grid_params
	)
	log("--------------------------------------")
	return results


if __name__ == "__main__":
	num_runs=20
	num_ticks=15000
	num_prtrb_agents=2
	init_seed=123
	baseline=True  # reseeded third grid per run, +50% run time


	"""
	Near-critical ecological growth regime - Random Driven Agents.
	--- ---
	This regime is intentionally near-critical. Small perturbations alter early reproduction timing,
	which cascades via nonlinear reproduction and energy redistribution.

	- Lyapunov exponent measures sensitivity to microscopic perturbations.
	- A marginal growth regime amplifies divergence.
	- Both agent types are tuned to operate in comparable fluctuation-dominated ecological regimes.

	Parameter set characteristics:
		• Avoid trivial extinction.
		• Avoid immediate saturation.
		• Maximize observable dynamical instability.

	The goal is dynamical comparability, not ecological realism.
	"""

	# near-critical ecological growth regime - random agents
	grid_params = {
		"width": 100,
		"height": 100,
		"metabolic_cost":0.9,
		"min_child_energy": 7,
		"reproduction_cost": 2,
		"food_respawn_rate": 0.012,
		"num_agents": 15,
		"num_apples": 25,
		"num_oranges": 25,
		"num_walls": 60,
		"use_nn": False
	}

	main_simulation(
		name="exp1_random_critical",
		num_runs=num_runs,
		num_ticks=num_ticks,
		num_prtrb_agents=num_prtrb_agents,
		init_seed=init_seed,
		baseline=baseline,
		**grid_params
	)


	"""
	Near-critical ecological growth regime - ANN Driven Agents.
	--- ---
	This regime is intentionally near-critical. Small perturbations alter early reproduction timing,
	which cascades via nonlinear reproduction and energy redistribution.

	Parameter set characteristics:
		• Avoid trivial extinction.
		• Avoid immediate saturation.
		• Maximize observable dynamical instability.
	"""

	# near-critical ecological growth regime - nn agents
	grid_params = {
		"width": 100,
		"height": 100,
		"metabolic_cost":0.9,
		"min_child_energy": 7,
		"reproduction_cost": 2,
		"food_respawn_rate": 0.012,
		"num_agents": 120,
		"num_apples": 900,
		"num_oranges": 900,
		"num_walls": 60,
		"use_nn": True
	}

	main_simulation(
		name="exp2_ann_critical",
		num_runs=num_runs,
		num_ticks=num_ticks,
		num_prtrb_agents=num_prtrb_agents,
		init_seed=init_seed,
		baseline=baseline,
		**grid_params
	)


	"""
	Identical ecological constraints scenario.
	--- ---
	In this block both agent types operate under exactly the same
	resource density, initial population size and environmental structure.

	Purpose:
	- Remove regime calibration as a confounding factor.
	- Isolate the effect of decision mechanism alone.
	- Observe whether NN agents shift the effective dynamical phase
	  under identical ecological pressure.

	This is not regime-matched.
	This is mechanism-under-identical-constraints.
	"""

	# identical ecological constraints - random agents
	grid_params = {
		"width": 100,
		"height": 100,
		"metabolic_cost":0.9,
		"min_child_energy": 7,
		"reproduction_cost": 2,
		"food_respawn_rate": 0.012,
		"num_agents": 120,
		"num_apples": 900,
		"num_oranges": 900,
		"num_walls": 60,
		"use_nn": False
	}

	main_simulation(
		name="exp3_random_ann_params",
		num_runs=num_runs,
		num_ticks=num_ticks,
		num_prtrb_agents=num_prtrb_agents,
		init_seed=init_seed,
		baseline=baseline,
		**grid_params
	)