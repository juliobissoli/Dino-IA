# from dinoAIrenderless import KeyClassifier, SmallCactus, LargeCactus, Bird
import numpy as np, pandas as pd

from dinoDefoult import KeyClassifier, SmallCactus, LargeCactus, Bird

from math import tanh, exp

from itertools import count
import time, numpy as np, random
from dinoDefoult import manyPlaysResults
# from dinoOnlyBackgrond import manyPlaysResults


class NeuralClassifier(KeyClassifier):
	def __init__(self, state, input, hidden, output):

		initState = 0
		endState = initState + input*hidden[0]
		self.hidden_weights = [np.array(state[initState:endState]).reshape(input, hidden[0])]
		initState = endState
		endState = initState + hidden[0] 
		self.hidden_bias = [np.array(state[initState:endState]).reshape(1, hidden[0])]
		i = 0
		while i+1 < len(hidden):
			initState = endState
			endState = initState + hidden[i]*hidden[i+1] 
			self.hidden_weights.append(np.array(state[initState:endState]).reshape(hidden[i], hidden[i+1]))
			initState = endState
			endState = initState + hidden[i+1] 
			self.hidden_bias.append(np.array(state[initState:endState]).reshape(1, hidden[i+1]))
			i+=1

		initState = endState
		endState = initState + hidden[i]*output 
		self.output_weights = np.array(state[initState:endState]).reshape(hidden[i], output)
		initState = endState
		endState = initState + output 
		self.output_bias = np.array(state[initState:endState]).reshape(1, output)

	def keySelector(self, speed, obstacles, player):
		obs = []
		for i in range(len(obstacles)):
			distance = obstacles[i].rect.right - player.dino_rect.left - speed
			if distance > 0:
				obs.append(obstacles[i])
				if i + 1 < len(obstacles):
					if not(obstacles[i+1].__class__.__name__ == 'Bird' and obstacles[i+1].getHeight() > 50):
						obs.append(obstacles[i+1])
				break
		
		dino_feet_pos = player.dino_rect.bottom - player.jump_vel
		dino_current_vertical_speed = player.jump_vel - player.jump_grav
		ground_top = player.Y_POS_DUCK+(player.dino_rect.bottom - player.dino_rect.top)
		distance_dino_bottom_ground_top = -(ground_top - dino_feet_pos)
		is_dino_jumping = int(player.dino_jump)

		distance_to_cross_obs = 0
		distance_to_reach_obs = 0
		distance_obs_top_dino_bottom = 0
		distance_to_reach_next_obs = 0
		next_obs_height = 0
		isBirdHigh = 0
		is_obstacle_large_cactus = 0

		if len(obs) > 0:
			isBirdHigh = int(obs[0].__class__.__name__ == 'Bird' and obs[0].getHeight() > 50)
			is_obstacle_large_cactus = int(obs[0].__class__.__name__ == 'LargeCactus')

			distance_to_cross_obs = obs[0].rect.right - player.dino_rect.left - speed
			distance_to_reach_obs = obs[0].rect.left - player.dino_rect.right - speed
			# Distances in the Y axis must be inverted, since
			# the Y axis grows down
			distance_obs_top_dino_bottom = -(obs[0].rect.top - dino_feet_pos)

		if len(obs) > 1:
			distance_to_reach_next_obs = obs[1].rect.left - player.dino_rect.right - speed
			next_obs_height = -(obs[1].rect.top - ground_top)
		time_to_reach_obs = distance_to_reach_obs/speed
		time_to_cross_obs = distance_to_cross_obs/speed
		time_to_be_above_obs = -100
		if distance_obs_top_dino_bottom > 0: 
			time_to_be_above_obs = self.parabola_roots(distance_obs_top_dino_bottom, 17, -1.1)[0]
		time_to_reach_obs_top_holding_down = self.parabola_roots(distance_obs_top_dino_bottom, dino_current_vertical_speed, -4.4)[1]
		time_to_reach_ground_holding_down = self.parabola_roots(distance_dino_bottom_ground_top, dino_current_vertical_speed, -4.4)[1]
		time_to_reach_next_obs = 0
		time_to_be_above_next_obs_from_ground = 0
		time_to_be_above_next_obs_from_here = 0

		if distance_to_reach_next_obs > 0:
			time_to_reach_next_obs = distance_to_reach_next_obs/speed
			time_to_be_above_next_obs_from_ground = self.parabola_roots(next_obs_height, 17, -1.1)[0]
			time_to_be_above_next_obs_from_here = time_to_reach_ground_holding_down + time_to_be_above_next_obs_from_ground

		KEY = self.classify(time_to_cross_obs, isBirdHigh, time_to_reach_next_obs, time_to_be_above_next_obs_from_here, is_obstacle_large_cactus, time_to_reach_obs_top_holding_down, is_dino_jumping, dino_current_vertical_speed, time_to_reach_obs, time_to_be_above_obs)

		return KEY

	def classify(self, time_to_cross_obs, isBirdHigh, time_to_reach_next_obs, time_to_be_above_next_obs_from_here, is_obstacle_large_cactus, time_to_reach_obs_top_holding_down, is_dino_jumping, dino_current_vertical_speed, time_to_reach_obs, time_to_be_above_obs):

		input_layer = np.array([time_to_cross_obs, isBirdHigh, time_to_reach_next_obs, time_to_be_above_next_obs_from_here, is_obstacle_large_cactus, time_to_reach_obs_top_holding_down, is_dino_jumping, dino_current_vertical_speed, time_to_reach_obs, time_to_be_above_obs]).reshape(1, 10)

		hidden_layer = input_layer
		for hw,hb in zip(self.hidden_weights, self.hidden_bias):
			hidden_layer =  np.matmul(hidden_layer, hw)
			hidden_layer += hb 
			hidden_layer =(hidden_layer > 0).astype(float)
		
		output_layer = np.matmul(hidden_layer, self.output_weights)
		output_layer += self.output_bias

		output = output_layer[0][0]

		if output < 0:
			# print('K_DOWN')
			return 'K_DOWN'
		else:
			# print('K_UP')
			return 'K_UP'


	def updateState(self, state):
		self.__init__(state)


	def parabola_roots(self, dy, v, a):
		delta = max((v-a/2)*(v-a/2)+2*dy*a, 0)**0.5
		roots = [(-(v-a/2)-delta)/a,(-(v-a/2)+delta)/a]
		return(min(roots), max(roots), )




# ++++ GA ++++++

def getBestPlayerResult(player_class, state, input, hidden, output):
	return manyPlaysResults(player_class(state, input, hidden, output), 10)[1]

# def generate_states(initial_state, lr=0.01):
# 	return [[e+lr*int(i==j) for i, e in enumerate(initial_state)] for j in range(len(initial_state))] + [[e-lr*int(i==j) for i, e in enumerate(initial_state)] for j in range(len(initial_state))]

# def states_total_value(states):
# 	return sum([max(e[0], 0) for e in states])

def roulette_construction(states):
	# total_value = states_total_value(states)
	total_value = sum([max(e[0], 0) for e in states])
	roulette = [(max(e[0],0)/total_value, e[1]) for e in states]
	for i in range(1, len(roulette)):
		roulette[i] =(roulette[i][0]+roulette[i-1][0], roulette[i][1])
	return roulette

def roulette_run(rounds, roulette):
	return [roulette[np.searchsorted([e[0] for e in roulette], random.uniform(0,1))][1] for _ in range(rounds)]

def selection(value_population, n):
	return roulette_run(n, roulette_construction(value_population))

# def crossover(dad, mom):
# 	r = random.randint(0, len(dad) - 1)
# 	return dad[:r] + mom[r:], mom[:r] + dad[r:]

def crossover2(dad, mom): 
	alfa = np.random.rand()
	dad = np.array(dad)
	mom = np.array(mom)
	child1 = alfa * dad +(1 - alfa) * mom
	child2 = alfa * mom +(1 - alfa) * dad
	return list(child1), list(child2)
	#  .randint(0, len(dad) - 1)

def mutation(indiv, lr=0.01):
	index = random.randint(0, len(indiv) - 1)

	if random.uniform(0, 1) > 0.5:
		indiv[index] += lr
	else:
		indiv[index] -= lr

	return indiv

def initPopulation(n):
	# list(np.random.rand(11*8 + 9*6 + 7*1) * 20 - 10)
	listRes = []
	i = 0
	for _ in range(n):	
		state = list(np.random.rand(149) * 101 - 200)
		# state[8] = 91.5
		listRes.append(state)
	return listRes
	# return [list(np.random.rand(149) * 200 - 100) for _ in range(n)]
	# [list(np.random.rand(-100, 101, 149)) for _ in range(n)]

def convergent(population):
	# print('population => ', population)
	for i in range(len(population) - 1):
		if population[i] != population[i + 1]:
			return False
	return True

def evaluate_population(player_class, population, input, hidden, output):
	list = []
	count = 0
	for state in population:
		evalRes = getBestPlayerResult(player_class, state, input, hidden, output)
		count +=1
		print('evaluate population: \t', count, '\t value', evalRes)
		list.append((evalRes, state))
	return list
	# return [(getBestPlayerResult(player_class, state, input, hidden, output), state) for state in population]

def elitism(val_pop, pct):
	return [s for v, s in sorted(val_pop, key=lambda x: x[0], reverse=True)[:max(pct*len(val_pop)//100, 1)]]

def crossover_step(population, crossover_ratio):
	# print('cros_step => ', population, crossover_ratio)
	new_pop = []
	for _ in range(len(population)//2):
		parent1, parent2 = random.sample(population, 2)
		if random.uniform(0, 1) <= crossover_ratio:
			offspring1, offspring2 = crossover2(parent1, parent2)
			# offspring1, offspring2 = crossover(parent1, parent2)
		else:
			offspring1, offspring2 = parent1, parent2
		new_pop += [offspring1, offspring2]
	return new_pop

def mutation_step(population, mutation_ratio, lr=0.01):
	return [mutation(e, lr) if random.uniform(0, 1) < mutation_ratio else e for e in population]

def genetic(player_class, base_state, pop_size, max_iter, cross_ratio, mut_ratio, max_time, elite_pct, learning_rate=0.01, input=10, hidden=[], output=1):
	print('=========== GA ===============\n\n')
	start = time.time()
	pop = [base_state] + initPopulation(pop_size - 1)
	opt_state = base_state
	opt_value = getBestPlayerResult(player_class, opt_state, input, hidden, output)
	last_change_iter = 0
	last_change_time = time.time()
	i = 0
	try:
		while not convergent(pop) and i < max_iter and time.time() - start <= max_time:
			val_pop = evaluate_population(player_class, pop, input, hidden, output)
			new_pop = elitism(val_pop, elite_pct)
			best = new_pop[0].copy()
			val_best = getBestPlayerResult(player_class, best, input, hidden, output)

			if val_best > opt_value:
				print('New best state found')
				print(best)
				last_change_iter = i
				last_change_time = time.time()
				opt_state = best.copy()
				opt_value = val_best
			
			# if last_change_iter - i > max_iter//5:
			# 	print('More than 20%% of maximum permited iterations without changing best state, breaking earlier.')
			# 	return opt_state, opt_value, i, convergent(pop), time.time() - start <= max_time
			# if last_change_iter - i > 1000:
			# 	print('More than 1000 iterations without changing best state, breaking earlier.')
			# 	return opt_state, opt_value, i, convergent(pop), time.time() - start <= max_time
			# if time.time() - last_change_time > max_time//5:
			# 	print('More than 20%% of maximum permited time without changing best state, breaking earlier.')
			# 	return opt_state, opt_value, i, convergent(pop), time.time() - start <= max_time
			# if time.time() - last_change_time > 8*60*60:
			# 	print('More than 8 hours without changing best state, breaking earlier.')
			# 	return opt_state, opt_value, i, convergent(pop), time.time() - start <= max_time


			selected = selection(val_pop, pop_size - len(new_pop))
			crossed = crossover_step(selected, cross_ratio)
			mutated = mutation_step(crossed, mut_ratio, learning_rate)
			pop = new_pop + mutated

			print('% 15s % 5d done, time elapsed: % 8.1f, score: % 10.3f' %('Iteration', i, time.time() - start, opt_value))
			i += 1
	except KeyboardInterrupt:
		pass

	return opt_state, opt_value, i, convergent(pop), time.time() - start <= max_time

