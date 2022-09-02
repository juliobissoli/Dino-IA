# from dinoAI import *
from dinoDefoul import *
# from dinoOnlyBackgrond import *
import numpy as np

# from searches import genetic
from classifiers import FrancoNeuralClassifier

def main ():
	INITIAL_TIME = time.time()

	best_state = [
		-1.5, 0.0, 1.0, -0.9, -0.5, 0.0, -0.5, 0.0,
		91.5, 0.0, -1.0, 0.9, 0.0, 0.0, 0.4, 0.5,
		-0.6, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.1, 0.0, 0.0, 0.5, 0.0,
		-0.4, 0.0, -0.1, 2.9, 0.5, 0.0, 0.0, 0.0,
		0.0, 0.0, -1.0, 0.0, 0.5, 0.0, 0.5, -0.1,
		0.0, -0.5, 0.0, 0.0, 0.0, 1.0, 0.5, -2.0,
		0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 1.5, 0.0,
		-0.1, 0.0, 0.0, -0.5, 0.0, 0.0, 0.1, -0.9,
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, # Hidden layer 0 weights
		
		0.0, -0.5, 0.0, 11.6, 0.0, -0.5, 0.5, 2.1, # Hidden layer 0 biases
		
		1.0, 0.0, 0.0, 0.0, -0.5, 0.5,
		-0.5, 0.5,0.5, 0.5, 0.1, 0.0,
		0.0, -0.4, 0.5, -0.5, 0.0, 0.0,
		0.0, -0.5, 1.0, 0.0, -0.6, -0.6,
		0.0, 0.5, -0.2, 1.0, 1.0, -0.5,
		0.0, 0.0, 0.5, -0.5, 1.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 1.5, -1.0,
		0.0, 0.0, 0.4, 0.5, 1.0, 1.0,
		-0.1, 0.0, -1.5, 0.0, -1.5, 0.0, # Hidden layer 1 weights
		
		-8.0, 2.5, 3.0, -2.4, 1.0, 1.5, # Hidden layer 1 biases
		
		-0.5 # Output layer bias
	]

	np.set_printoptions (3)

	# best_state, value, iterations, convergence, timed = genetic (FrancoNeuralClassifier, best_state, 100, 999999999, 0.8, 0.6, 60*60*8.5, 10, 0.1)

	aiPlayer = FrancoNeuralClassifier (best_state)
	res, value = manyPlaysResults (aiPlayer, 1000)
	npRes = np.asarray (res)
	print (npRes)
	print ('%.2f '*6 % (npRes.min (), npRes.max (), np.median (npRes), npRes.mean (), npRes.std (), value))
	print ('Time:', time.time () - INITIAL_TIME)


if __name__ == '__main__':
	main ()
