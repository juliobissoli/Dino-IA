from dinoDefoult import *
import numpy as np
from classifiers import *


def main ():
	INITIAL_TIME = time.time()

	np.set_printoptions (3)

	init_state = list(np.random.rand(14*8 + 12*6 + 7*1) * 101 - 200)
	
	
	best_state, value = genetic(NeuralClassifier, init_state, 100, 999999999, 0.8, 0.6, 60*60*8, 10, 0.1, 12, [8, 4], 1)


	print('\n\n++++++++++++++++++++++++++++')
	print('++++++Termino o treino++++++')
	print('++++++++++++++++++++++++++++\n\n')
	print('bast_state => ',  best_state)

	aiPlayer = NeuralClassifier(best_state, 12, [8, 4], 1)
	res, value = manyPlaysResults(aiPlayer, 30)
	npRes = np.asarray (res)
	print (npRes)
	print ('%.2f '*6 % (npRes.min (), npRes.max (), np.median (npRes), npRes.mean (), npRes.std (), value))
	print ('Time:', time.time () - INITIAL_TIME)


if __name__ == '__main__':
	main ()
