import random
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import math
from collections import Counter

directions = [
	(-1, 0), (1, 0), (0, -1), (0, 1),  # left, right, down, up
	(-1, -1), (-1, 1), (1, -1), (1, 1)  # diagonals
]

# per-tick event and energy-flow counters (see Grid.move_agent)
TICK_STATS = (
	"acted", "starved", "fights", "matings", "failed_matings", "births",
	"food", "metabolism", "reproduction", "residual"
)

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

class Grid:
	def __init__(self, width, height, metabolic_cost, min_child_energy, reproduction_cost, food_respawn_rate,
	num_agents, num_apples, num_oranges, num_walls=30, use_nn=False, seed=123):
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
		self.rng = random.Random(seed)
		self.stats = dict.fromkeys(TICK_STATS, 0)

		self.populate(num_agents, num_apples, num_oranges, num_walls)

	def cross_mutate(self, nn1, nn2):
		child = copy.deepcopy(nn1)

		for attr in ["W1", "W2"]:
			parent_matrix = getattr(nn2, attr)
			child_matrix = getattr(child, attr)

			mask = np.array([[self.rng.random() < 0.5
							for _ in range(child_matrix.shape[1])]
							for _ in range(child_matrix.shape[0])])

			child_matrix[mask] = parent_matrix[mask]

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
				child.nn = self.cross_mutate(agent1.nn, agent2.nn)

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

			# artificial NN use
			if self.use_nn:
				vision = self.get_agent_vision(agent)
				agent.vision = vision
				dx, dy = agent.nn_output_to_move()
			else:
				# random movement (default behavior)
				dx, dy = self.rng.choice(directions)

			nx, ny = agent.x + dx, agent.y + dy

			# check bounds
			if not (0 <= nx < self.height and 0 <= ny < self.width):
				continue

			# blocked by wall
			if isinstance(self.grid[nx][ny], Wall):
				continue

			target = self.grid[nx][ny]

			# eat food
			if isinstance(target, Food):
				agent.energy += target.energy
				self.stats["food"] += target.energy
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

		n = int(len(empty) * self.food_respawn_rate)
		for x, y in self.rng.sample(empty, n):
			if self.rng.random() < 0.6:
				food = Apple(x, y)
			else:
				food = Orange(x, y)

			self.place_object(food)
			self.food_items.append(food)

	def render(self):
		color_grid = np.ones((self.height, self.width, 3))  # white background

		for wall in self.walls:
			color_grid[wall.x][wall.y] = wall.color

		for agent in self.agents:
			color_grid[agent.x][agent.y] = agent.color

		for food in self.food_items:
			color_grid[food.x][food.y] = food.color

		num_apples = sum(isinstance(f, Apple) for f in self.food_items)
		num_oranges = sum(isinstance(f, Orange) for f in self.food_items)

		combined_energy = sum(agent.energy for agent in self.agents)

		title = (
			f"Agents: {len(self.agents)} | "
			f"Total agents energy: {combined_energy:.2f} | "
			f"Apples: {num_apples} | "
			f"Oranges: {num_oranges} | "
			f"Walls: {len(self.walls)}"
		)

		plt.figure(figsize=(10, 10))
		plt.imshow(color_grid, interpolation='nearest')
		plt.title(title)
		plt.axis('off')
		plt.show()


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


def main_analysis(g1, g2, num_ticks=50, render=False, trial=1, cutoff=0.2, **grid_params):
	"""
	Estimate the maximal finite-time Lyapunov exponent (FTLE) via lockstep evolution,
	compute Shannon entropy difference, and calculate basic regime stability metrics.

	Methodology
	-----------
	- Lyapunov exponent λ is estimated from the early exponential divergence:
		d(t) ≈ d0 * exp(λ t)
		log d(t) ≈ log d0 + λ t
	  using a fixed cutoff fraction of total ticks for regression.

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
	cutoff : float, optional
		Fraction of total ticks used for Lyapunov regression. Default 0.2.
	**grid_params : dict
		Additional grid parameters required for entropy computation and population metrics.

	Returns
	-------
	dict
		Dictionary containing the following metrics:
		- "lyap" : float
			Estimated finite-time Lyapunov exponent.
		- "diffs" : list of float
			Phase-space distances d(t) trajectory.
		- "shannon_diff" : float
			Shannon entropy difference ΔH between grids at final tick.
		- "cutoff_pct" : float
			Actual cutoff fraction (%) used for regression.
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
	"""
	diffs = []
	pop_history = []

	d0 = max(grid_difference(g1, g2), 1e-12)
	diffs.append(d0)

	if render:
		g1.render()
		print("Grid 1 - Initial Shannon entropy:", shannon_entropy(g1, grid_params['min_child_energy']))
		g2.render()
		print("Grid 2 - Initial Shannon entropy:", shannon_entropy(g2, grid_params['min_child_energy']))

	for t in range(num_ticks):
		g1.move_agent()
		g1.respawn_food()

		g2.move_agent()
		g2.respawn_food()

		d = max(grid_difference(g1, g2), 1e-12)
		diffs.append(d)
		pop_history.append(len(g1.agents))

	if render:
		g1.render()
		print("Grid 1 - Final Shannon entropy:", shannon_entropy(g1, grid_params['min_child_energy']))
		g2.render()
		print("Grid 2 - Final Shannon entropy:", shannon_entropy(g2, grid_params['min_child_energy']))

	# -- lyap --
	log_diffs = np.log(diffs)
	cutoff_idx = max(10, int(cutoff * num_ticks))
	cutoff_pct = (cutoff_idx / num_ticks) * 100

	x = np.arange(cutoff_idx)
	y = log_diffs[:cutoff_idx]

	slope, _ = np.polyfit(x, y, 1)
	lyap = slope

	# --- entropy ---
	g1_H = shannon_entropy(g1, grid_params['min_child_energy'])
	g2_H = shannon_entropy(g2, grid_params['min_child_energy'])
	shannon_diff = abs(g2_H - g1_H)

	# --- regime metrics ---
	pop_history = np.array(pop_history)
	g = g1
	burn_frac = 0.3

	if pop_history[-1] == 0:
		viable = 0
	else:
		viable = any(a.sex == 1 for a in g.agents) and any(a.sex == 0 for a in g.agents)

	population_preserved = pop_history[-1] >= grid_params['num_agents']

	pop_history_safe = np.maximum(pop_history, 1)

	start = int(burn_frac * num_ticks)
	t_vals = np.arange(start, num_ticks)

	logN = np.log(pop_history_safe[start:])

	r = float("nan")
	if len(logN) > 5:
		r, _ = np.polyfit(t_vals, logN, 1)

	meanN = np.mean(pop_history[start:])
	stdN = np.std(pop_history[start:])
	cv = stdN / meanN if meanN > 0 else float("inf")

	print(
		f"[Run {trial+1:02d}] "
		f"λ = {lyap:.5f} | "
		f"cutoff = {cutoff_pct:.2f}% | "
		f"ΔH = {shannon_diff:.5f}"
	)

	# --- divergence trajectory plot ---
	plt.figure(figsize=(6, 4))
	plt.plot(range(len(diffs)), diffs, label='d(t) trajectory')
	plt.axvline(x=cutoff_idx, color='red', linestyle='--', label=f'Cutoff ({cutoff_pct:.1f}%)')
	plt.xlabel('Tick')
	plt.ylabel('Phase-space distance d(t)')
	plt.title(f'Lyapunov Divergence Run {trial+1}')
	plt.legend()
	plt.grid(True)
	plt.show()

	return {
		"lyap": lyap,
		"diffs": diffs,
		"shannon_diff": shannon_diff,
		"cutoff_pct": cutoff_pct,
		"r": r,
		"cv": cv,
		"viable": viable,
		"population_preserved": population_preserved,
		"final_population": int(pop_history[-1])
	}


def compare_grids(num_ticks=50, num_perturbed_agents=1, seed=123, final_render=True, lyapunov_final_render=True, num_trials=30, cutoff=0.2, **grid_params):
	"""
	Compare two nearly identical grid simulations to estimate the Lyapunov exponent,
	Shannon entropy and aggregate regime statistics over multiple trials.

	Parameters
	----------
	num_ticks : int, optional
		Number of simulation ticks to run. Default is 50.
	num_perturbed_agents : int, optional
		Number of agents to be perturbated (by adding energy).
	seed : int, optional
		Random seed ensuring deterministic setup. Default is 123.
	cutoff : float, optional
		Fraction of ticks used for Lyapunov regression. Default is 0.2.

	Returns
	-------
	mean_lambda : float
		Estimated Lyapunov exponent of the system.
	mean_H : float
		Shannon entropy difference between two grids.
	"""
	lambdas = []
	shannon_diffs = []
	cutoffs = []
	r_vals = []
	cv_vals = []
	viable_flags = []
	pop_preserved_flags = []

	for trial in range(num_trials):
		g1 = Grid(**grid_params, seed=seed + trial)
		g2 = Grid(**grid_params, seed=seed + trial)

		if g2.agents:
			local_rng = random.Random(seed + 999 + trial)
			k = min(num_perturbed_agents, len(g2.agents))
			perturbed = local_rng.sample(g2.agents, k)
			for agent in perturbed:
				agent.energy += g2.min_child_energy

		result = main_analysis(
			g1, g2, num_ticks, lyapunov_final_render, trial, cutoff, **grid_params
		)

		lambdas.append(result["lyap"])
		shannon_diffs.append(result["shannon_diff"])
		cutoffs.append(result["cutoff_pct"])
		r_vals.append(result["r"])
		cv_vals.append(result["cv"])
		viable_flags.append(result["viable"])
		pop_preserved_flags.append(result["population_preserved"])

	# --- lyap & entropy ---
	mean_lambda = np.mean(lambdas)
	std_lambda = np.std(lambdas)
	mean_H = np.mean(shannon_diffs)
	std_H = np.std(shannon_diffs)

	print("\n----- SUMMARY -----")
	print(f"Lyapunov exponent     : {mean_lambda:.6f} ± {std_lambda:.6f}")
	print(f"Shannon entropy ΔH    : {mean_H:.6f} ± {std_H:.6f}")

	# --- regime stats ---
	mean_r = np.mean(r_vals)
	std_r = np.std(r_vals)
	valid_cvs = [cv for cv in cv_vals if np.isfinite(cv)]
	mean_cv = np.mean(valid_cvs)
	viability_rate = np.mean(viable_flags)
	preservation_rate = np.mean(pop_preserved_flags)

	print("\n----- REGIME SUMMARY -----")
	print(f"{'Mean log population growth rate':45s}: {mean_r:.6f}")
	print(f"{'Standard deviation of log growth rate':45s}: {std_r:.6f}")
	# runs extinct before the burn-in ends have no defined CV and are left out
	print(f"{'Mean coefficient of variation':45s}: {mean_cv:.6f} ({len(valid_cvs)}/{len(cv_vals)} runs)")
	print(f"{'Viability preservation probability':45s}: {viability_rate:.3f}")
	print(f"{'Population preservation probability':45s}: {preservation_rate:.3f}")

	return mean_lambda, mean_H


def check_determinism(num_ticks, seed, debug_render=False, final_render=True, **grid_params):
	"""
	Verify whether the simulation environment is fully deterministic.

	Runs two simulations with identical initial conditions and checks
	whether the final grid states are identical.

	Returns
	-------
	bool
		True if deterministic, False otherwise.
	"""
	print("\n----- DETERMINISTIC CHECK -----")

	g1 = Grid(**grid_params, seed=seed)
	g2 = Grid(**grid_params, seed=seed)

	if final_render == True:
		g1.render()
		print("Grid 1 - Initial Shannon entropy:", shannon_entropy(g1, grid_params['min_child_energy']))
		g2.render()
		print("Grid 2 - Initial Shannon entropy:", shannon_entropy(g2, grid_params['min_child_energy']))

	for t in range(num_ticks):
		g1.move_agent()
		g1.respawn_food()

		if debug_render == True:
			g1.render()
			g2.render()

		g2.move_agent()
		g2.respawn_food()

	if final_render == True:
		g1.render()
		print("Grid 1 - Final Shannon entropy:", shannon_entropy(g1, grid_params['min_child_energy']))
		g2.render()
		print("Grid 2 - Final Shannon entropy:", shannon_entropy(g2, grid_params['min_child_energy']))

	diff = grid_difference(g1, g2)

	print(
		"Difference between two grids with identical initial conditions "
		f"(expected 0.0): {diff}"
	)

	if diff > 0:
		print("Determinism check FAILED.")
		return False

	print("Determinism check PASSED.")
	print("--------------------------------------")
	return True


def main_simulation(num_runs, num_ticks, num_prtrb_agents, init_seed, cutoff, debug_render=False, final_render=False, lyapunov_final_render=False, **grid_params):
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

	Returns
	-------
	tuple of (Grid, float)
		Final grid state and estimated Lyapunov exponent.
	"""

	print("\n----- PARAMETERS LIST -----")
	print(f"Number of runs              : {num_runs}")
	print(f"Simulation ticks            : {num_ticks}")
	print(f"Perturbed agents            : {num_prtrb_agents}")
	print(f"Initial random seed         : {init_seed}")
	print(f"Grid width                  : {grid_params['width']}")
	print(f"Grid height                 : {grid_params['height']}")
	print(f"Metabolic cost              : {grid_params['metabolic_cost']}")
	print(f"Minimum child energy        : {grid_params['min_child_energy']}")
	print(f"Reproduction cost           : {grid_params['reproduction_cost']}")
	print(f"Food respawn rate           : {grid_params['food_respawn_rate']}")
	print(f"Initial number of agents    : {grid_params['num_agents']}")
	print(f"Initial number of apples    : {grid_params['num_apples']}")
	print(f"Initial number of oranges   : {grid_params['num_oranges']}")
	print(f"Number of walls             : {grid_params['num_walls']}")
	print(f"Agent decision mechanism    : {'Neural Network' if grid_params['use_nn'] else 'Random'}")
	print("--------------------------------------")

	is_deterministic = check_determinism(num_ticks=num_ticks,seed=init_seed,debug_render=debug_render,final_render=final_render,**grid_params)

	if not is_deterministic:
		raise RuntimeError("Simulation environment is non-deterministic.")

	print("\n----- LYAPUNOV EXPONENT COMPARISON -----")
	lyap = compare_grids(num_ticks, num_prtrb_agents, init_seed, final_render, lyapunov_final_render, num_runs, cutoff, **grid_params)
	print("--------------------------------------")


if __name__ == "__main__":
	num_runs=20
	num_ticks=15000
	num_prtrb_agents=2
	init_seed=123
	cutoff=0.015


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
		"reproduction_cost": 8,
		"food_respawn_rate": 0.012,
		"num_agents": 15,
		"num_apples": 30,
		"num_oranges": 30,
		"num_walls": 60,
		"use_nn": False
	}

	main_simulation(
		num_runs=num_runs,
		num_ticks=num_ticks,
		num_prtrb_agents=num_prtrb_agents,
		init_seed=init_seed,
		cutoff=cutoff,
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
		"reproduction_cost": 8,
		"food_respawn_rate": 0.012,
		"num_agents": 80,
		"num_apples": 900,
		"num_oranges": 900,
		"num_walls": 60,
		"use_nn": True
	}

	main_simulation(
		num_runs=num_runs,
		num_ticks=num_ticks,
		num_prtrb_agents=num_prtrb_agents,
		init_seed=init_seed,
		cutoff=cutoff,
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
		"reproduction_cost": 8,
		"food_respawn_rate": 0.012,
		"num_agents": 80,
		"num_apples": 900,
		"num_oranges": 900,
		"num_walls": 60,
		"use_nn": False
	}

	main_simulation(
		num_runs=num_runs,
		num_ticks=num_ticks,
		num_prtrb_agents=num_prtrb_agents,
		init_seed=init_seed,
		cutoff=cutoff,
		**grid_params
	)