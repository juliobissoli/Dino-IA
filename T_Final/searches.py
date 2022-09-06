import time, numpy as np, random
from dinoOnlyBackgrond import manyPlaysResults

def evaluate_state (player_class, state):
	return manyPlaysResults (player_class (state), 30)[1]

def generate_states (initial_state, lr=0.01):
	return [[e+lr*int(i==j) for i, e in enumerate (initial_state)] for j in range (len (initial_state))] + [[e-lr*int(i==j) for i, e in enumerate (initial_state)] for j in range (len (initial_state))]

def states_total_value (states):
	return sum ([max(e[0], 0) for e in states])

def roulette_construction (states):
	total_value = states_total_value (states)
	roulette = [(max(e[0],0)/total_value, e[1]) for e in states]
	for i in range (1, len (roulette)):
		roulette[i] = (roulette[i][0]+roulette[i-1][0], roulette[i][1])
	return roulette

def roulette_run (rounds, roulette):
	return [roulette[np.searchsorted ([e[0] for e in roulette], random.uniform (0,1))][1] for _ in range (rounds)]

def selection (value_population, n):
	return roulette_run (n, roulette_construction (value_population))

def crossover (dad, mom):
	r = random.randint (0, len (dad) - 1)
	return dad[:r] + mom[r:], mom[:r] + dad[r:]

def mutation (indiv, lr=0.01):
	index = random.randint (0, len (indiv) - 1)

	if random.uniform (0, 1) > 0.5:
		indiv[index] += lr
	else:
		indiv[index] -= lr

	return indiv

def initial_population (n):
	return [list (np.random.randint (-100, 101, 149)) for _ in range (n)]

def convergent (population):
	for i in range (len (population) - 1):
		if population[i] != population[i + 1]:
			return False
	return True

def evaluate_population (player_class, population):
	return [(evaluate_state (player_class, state), state) for state in population]

def elitism (val_pop, pct):
	return [s for v, s in sorted (val_pop, key=lambda x: x[0], reverse=True)[:max (pct*len (val_pop)//100, 1)]]

def crossover_step (population, crossover_ratio):
	new_pop = []
	for _ in range (len (population)//2):
		parent1, parent2 = random.sample (population, 2)
		if random.uniform (0, 1) <= crossover_ratio:
			offspring1, offspring2 = crossover (parent1, parent2)
		else:
			offspring1, offspring2 = parent1, parent2
		new_pop += [offspring1, offspring2]
	return new_pop

def mutation_step (population, mutation_ratio, lr=0.01):
	return [mutation (e, lr) if random.uniform (0, 1) < mutation_ratio else e for e in population]

def genetic (player_class, base_state, pop_size, max_iter, cross_ratio, mut_ratio, max_time, elite_pct, learning_rate=0.01):
	start = time.time ()
	pop = [base_state] + initial_population (pop_size - 1)
	opt_state = base_state
	opt_value = evaluate_state (player_class, opt_state)
	last_change_iter = 0
	last_change_time = time.time ()
	i = 0
	try:
		while not convergent (pop) and i < max_iter and time.time () - start <= max_time:
			val_pop = evaluate_population (player_class, pop)
			new_pop = elitism (val_pop, elite_pct)
			best = new_pop[0].copy ()
			val_best = evaluate_state (player_class, best)

			if val_best > opt_value:
				print ('New best state found')
				print (best)
				last_change_iter = i
				last_change_time = time.time ()
				opt_state = best.copy ()
				opt_value = val_best
			
			if last_change_iter - i > max_iter//5:
				print ('More than 20%% of maximum permited iterations without changing best state, breaking earlier.')
				return opt_state, opt_value, i, convergent (pop), time.time () - start <= max_time
			if last_change_iter - i > 1000:
				print ('More than 1000 iterations without changing best state, breaking earlier.')
				return opt_state, opt_value, i, convergent (pop), time.time () - start <= max_time
			if time.time () - last_change_time > max_time//5:
				print ('More than 20%% of maximum permited time without changing best state, breaking earlier.')
				return opt_state, opt_value, i, convergent (pop), time.time () - start <= max_time
			if time.time () - last_change_time > 8*60*60:
				print ('More than 8 hours without changing best state, breaking earlier.')
				return opt_state, opt_value, i, convergent (pop), time.time () - start <= max_time


			selected = selection (val_pop, pop_size - len (new_pop))
			crossed = crossover_step (selected, cross_ratio)
			mutated = mutation_step (crossed, mut_ratio, learning_rate)
			pop = new_pop + mutated

			print ('% 15s % 5d done, time elapsed: % 8.1f, score: % 10.3f' % ('Iteration', i, time.time () - start, opt_value))
			i += 1
	except KeyboardInterrupt:
		pass

	return opt_state, opt_value, i, convergent (pop), time.time () - start <= max_time

