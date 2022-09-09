# from dinoAI import *
from dinoDefoult import *
# from dinoJulio import *

# from dinoOnlyBackgrond import *
import numpy as np
# import time, numpy as np, random
import random

# from searches import genetic
from classifiers import *


def main ():
	INITIAL_TIME = time.time()

	best_state = [
		-1.5, 	0.0, 	1.0, 	-0.9, 	-0.5, 	0.0,	-0.5, 	0.0,
		91.5, 	0.0, 	-1.0, 	0.9, 	0.0, 	0.0,	0.4, 	0.5,
		-0.6, 	-1.0, 	0.0, 	0.0, 	0.0, 	0.0,	0.0, 	0.0,
		0.0, 	1.0, 	0.0, 	0.1, 	0.0, 	0.0,	0.5, 	0.0,
		-0.4, 	0.0, 	-0.1, 	2.9, 	0.5, 	0.0,	0.0, 	0.0,
		0.0, 	0.0, 	-1.0, 	0.0, 	0.5, 	0.0,	0.5, 	-0.1,
		0.0, 	-0.5, 	0.0, 	0.0, 	0.0, 	1.0,	0.5, 	-2.0,
		0.0, 	0.0, 	0.0, 	0.0, 	0.1, 	0.0,	1.5, 	0.0,
		-0.1, 	0.0, 	0.0, 	-0.5, 	0.0, 	0.0,	0.1, 	-0.9,
		0.0, 	0.0, 	0.0, 	0.0, 	0.0, 	0.0,	0.0, 	1.5, # Hidden layer 0 weights
		
		0.0, 	-0.5, 	0.0, 	11.6, 	0.0, 	-0.5, 	0.5, 	2.1, # Hidden layer 0 biases
		
		1.0, 	0.0, 	0.0,	 0.0,	 -0.5,	0.5,
		-0.5, 	0.5,	0.5,	 0.5,	 0.1,	0.0,
		0.0, 	-0.4, 	0.5,	 -0.5,	 0.0,	0.0,
		0.0, 	-0.5, 	1.0,	 0.0,	 -0.6,	-0.6,
		0.0, 	0.5, 	-0.2,	 1.0,	 1.0,	-0.5,
		0.0, 	0.0, 	0.5,	 -0.5,	 1.0,	0.0,
		0.0, 	0.0, 	0.0,	 0.0,	 1.5,	-1.0,
		0.0, 	0.0, 	0.4,	 0.5,	 1.0,	1.0,
		-0.1, 	0.0, 	-1.5,	 0.0,	 -1.5,	0.0, # Hidden layer 1 weights
		
		-8.0,	2.5,	 3.0, 	-2.4, 	1.0, 	1.5, # Hidden layer 1 biases
		
		-0.5 # Output layer bias
	]

	np.set_printoptions (3)

	# best_state, value, iterations, convergence, timed = genetic (FrancoNeuralClassifier, best_state, 100, 999999999, 0.8, 0.6, 60*60*8.5, 10, 0.1)
	# init_state =  [random.randrange(1, 10, 1) for i in range(((11 * 8)  + (10 * 6) + 1))]
	
	init_state = list(np.random.rand(11*8 + 9*6 + 7*1) * 11 - 2)
	
	print(init_state)
	print(len(best_state))
	print('Vai treinar')
	# aiPlayerTreino = FrancoNeuralClassifier (best_state, 10, [8, 6], 1)
	best_state, value, iterations, convergence, timed = genetic(NeuralClassifier, init_state, 30, 999999999, 0.8, 0.6, 60*60*8, 10, 0.1, 10, [8, 6], 1)


	print('\n\n++++++++++++++++++++++++++++')
	print('++++++Termino o treino++++++')
	print('++++++++++++++++++++++++++++\n\n')
	print('bast_state => ',  best_state)

	aiPlayer = NeuralClassifier(best_state, 10, [8, 6], 1)
	res, value = manyPlaysResults(aiPlayer, 30)
	npRes = np.asarray (res)
	print (npRes)
	print ('%.2f '*6 % (npRes.min (), npRes.max (), np.median (npRes), npRes.mean (), npRes.std (), value))
	print ('Time:', time.time () - INITIAL_TIME)


if __name__ == '__main__':
	main ()
