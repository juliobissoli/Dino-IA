# from dinoAIrenderless import KeyClassifier, SmallCactus, LargeCactus, Bird
import numpy as np, pandas as pd

from dinoDefoult import KeyClassifier, SmallCactus, LargeCactus, Bird

from math import tanh, exp


# The state below is the best state found at the end of the 24 hour search.

# best_state = [
# 	-1.5, 0.0, 1.0, -0.9, -0.5, 0.0, -0.5, 0.0,
# 	91.5, 0.0, -1.0, 0.9, 0.0, 0.0, 0.4, 0.5,
# 	-0.6, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
# 	0.0, 1.0, 0.0, 0.1, 0.0, 0.0, 0.5, 0.0,
# 	-0.4, 0.0, -0.1, 2.9, 0.5, 0.0, 0.0, 0.0,
# 	0.0, 0.0, -1.0, 0.0, 0.5, 0.0, 0.5, -0.1,
# 	0.0, -0.5, 0.0, 0.0, 0.0, 1.0, 0.5, -2.0,
# 	0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 1.5, 0.0,
# 	-0.1, 0.0, 0.0, -0.5, 0.0, 0.0, 0.1, -0.9,
# 	0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, # Hidden layer 0 weights
	
# 	0.0, -0.5, 0.0, 11.6, 0.0, -0.5, 0.5, 2.1, # Hidden layer 0 biases
	
# 	1.0, 0.0, 0.0, 0.0, -0.5, 0.5,
# 	-0.5, 0.5,0.5, 0.5, 0.1, 0.0,
# 	0.0, -0.4, 0.5, -0.5, 0.0, 0.0,
# 	0.0, -0.5, 1.0, 0.0, -0.6, -0.6,
# 	0.0, 0.5, -0.2, 1.0, 1.0, -0.5,
# 	0.0, 0.0, 0.5, -0.5, 1.0, 0.0,
# 	0.0, 0.0, 0.0, 0.0, 1.5, -1.0,
# 	0.0, 0.0, 0.4, 0.5, 1.0, 1.0,
# 	-0.1, 0.0, -1.5, 0.0, -1.5, 0.0, # Hidden layer 1 weights
	
# 	-8.0, 2.5, 3.0, -2.4, 1.0, 1.5, # Hidden layer 1 biases
	
# 	-0.5 # Output layer bias
# ]

class FrancoNeuralClassifier (KeyClassifier):
	def __init__(self, state, input, hidden, output):
		self.hidden_weights_0 = np.array (state[:80]).reshape (10, 8)
		self.hidden_bias_0 = np.array (state[80:88]).reshape (1, 8)
		self.hidden_weights_1 = np.array (state[88:136]).reshape (8, 6)
		self.hidden_bias_1 = np.array (state[136:142]).reshape (1, 6)
		self.output_weights = np.array (state[142:148]).reshape (6, 1)
		self.output_bias = np.array (state[148]).reshape (1, 1)

	def keySelector(self, speed, obstacles, player):
		# The inputs to this function are the game speed, the full list of
		# obstacles, and the dinossaur object stored in the "player" variable.


		# Data preprocessing
		
		# Sometimes, the the first obstacle in the "obstacle" list is behind
		# the dinosaur. As an obstacle behind the dinosaur is irrelevant, this
		# code ignores all first elements of the list, until an obstacle appears
		# that is in front of the dinosaur.

		# Since the second object is only used to check if the Dino should jump
		# two obstacles at once, the second obstacle is ignored if it is a high
		# bird.
		obs = []
		for i in range (len (obstacles)):
			distance = obstacles[i].rect.right - player.dino_rect.left - speed
			if distance > 0:
				obs.append (obstacles[i])
				if i + 1 < len (obstacles):
					if not (obstacles[i+1].__class__.__name__ == 'Bird' and obstacles[i+1].getHeight() > 50):
						obs.append (obstacles[i+1])
				break
		
		# The information given in the function input is delayed by one frame,
		# and must be corrected.
		dino_feet_pos = player.dino_rect.bottom - player.jump_vel
		dino_current_vertical_speed = player.jump_vel - player.jump_grav
		ground_top = player.Y_POS_DUCK+(player.dino_rect.bottom - player.dino_rect.top)
		distance_dino_bottom_ground_top = - (ground_top - dino_feet_pos)
		is_dino_jumping = int (player.dino_jump)

		distance_to_cross_obs = 0
		distance_to_reach_obs = 0
		distance_obs_top_dino_bottom = 0
		distance_to_reach_next_obs = 0
		next_obs_height = 0
		is_obstacle_high_bird = 0
		is_obstacle_large_cactus = 0

		if len (obs) > 0:
			is_obstacle_high_bird = int (obs[0].__class__.__name__ == 'Bird' and obs[0].getHeight() > 50)
			is_obstacle_large_cactus = int (obs[0].__class__.__name__ == 'LargeCactus')

			distance_to_cross_obs = obs[0].rect.right - player.dino_rect.left - speed
			distance_to_reach_obs = obs[0].rect.left - player.dino_rect.right - speed
			# Distances in the Y axis must be inverted, since
			# the Y axis grows down
			distance_obs_top_dino_bottom = - (obs[0].rect.top - dino_feet_pos)

		if len (obs) > 1:
			distance_to_reach_next_obs = obs[1].rect.left - player.dino_rect.right - speed
			next_obs_height = - (obs[1].rect.top - ground_top)

		# Time to cross horizontal distances is simple, as the increase in speed
		# is infrequent enough not to interfere in calculations. The time then
		# is simply distance/speed.
		time_to_reach_obs = distance_to_reach_obs/speed
		time_to_cross_obs = distance_to_cross_obs/speed
		time_to_be_above_obs = -100
		if distance_obs_top_dino_bottom > 0: 
			# Time to cross vertical distances need to consider acceleration,
			# and therefore need one of the solutions to a second degree
			# equation.
			time_to_be_above_obs = self.parabola_roots (distance_obs_top_dino_bottom, 17, -1.1)[0]
		time_to_reach_obs_top_holding_down = self.parabola_roots (distance_obs_top_dino_bottom, dino_current_vertical_speed, -4.4)[1]
		time_to_reach_ground_holding_down = self.parabola_roots (distance_dino_bottom_ground_top, dino_current_vertical_speed, -4.4)[1]
		time_to_reach_next_obs = 0
		time_to_be_above_next_obs_from_ground = 0
		time_to_be_above_next_obs_from_here = 0

		if distance_to_reach_next_obs > 0:
			time_to_reach_next_obs = distance_to_reach_next_obs/speed
			time_to_be_above_next_obs_from_ground = self.parabola_roots (next_obs_height, 17, -1.1)[0]
			time_to_be_above_next_obs_from_here = time_to_reach_ground_holding_down + time_to_be_above_next_obs_from_ground

		KEY = self.classify (time_to_cross_obs, is_obstacle_high_bird, time_to_reach_next_obs, time_to_be_above_next_obs_from_here, is_obstacle_large_cactus, time_to_reach_obs_top_holding_down, is_dino_jumping, dino_current_vertical_speed, time_to_reach_obs, time_to_be_above_obs)

		return KEY

	def classify(self, time_to_cross_obs, is_obstacle_high_bird, time_to_reach_next_obs, time_to_be_above_next_obs_from_here, is_obstacle_large_cactus, time_to_reach_obs_top_holding_down, is_dino_jumping, dino_current_vertical_speed, time_to_reach_obs, time_to_be_above_obs):

		input_layer = np.array([time_to_cross_obs, is_obstacle_high_bird, time_to_reach_next_obs, time_to_be_above_next_obs_from_here, is_obstacle_large_cactus, time_to_reach_obs_top_holding_down, is_dino_jumping, dino_current_vertical_speed, time_to_reach_obs, time_to_be_above_obs]).reshape (1, 10)

		hidden_layer_0 = np.matmul (input_layer, self.hidden_weights_0)
		hidden_layer_0 += self.hidden_bias_0
		hidden_layer_0 = (hidden_layer_0 > 0).astype (float)
		hidden_layer_1 = np.matmul (hidden_layer_0, self.hidden_weights_1)
		hidden_layer_1 += self.hidden_bias_1
		hidden_layer_1 = (hidden_layer_1 > 0).astype (float)
		output_layer = np.matmul (hidden_layer_1, self.output_weights)
		output_layer += self.output_bias


		output = output_layer[0][0]

		if output < 0:
			return 'K_DOWN'
		else:
			return 'K_UP'


	def updateState(self, state):
		self.__init__ (state)


	def parabola_roots (self, dy, v, a):
		# The vertical position of the dinossaur is modified by "v" every frame.
		# Besides that, "v" gets modified by "a" also every frame.
		# We can use the above to calculate the position of the dinossaur at
		# any frame:
		# v(0) = v_0
		# y(0) = y_0
		# y(t) = y(t - 1) + v(t - 1)
		# v(t) = v(t - 1) + a
		# Solving the above we have
		# v(t) = v_0 + t*a
		# y(t) = y_0 + sum from i=0 to i=t-1 of v(i)
		# y(t) = y_0 + t*v_0 + (1*a + 2*a + ... + (t-1)*a)
		# y(t) = y_0 + t*v_0 + (t-1)*t*a/2
		# Since y(t) - y_0 = dy
		# 0 = -dy + t*v_0 + t^2*a/2 - t*a/2
		# 0 = -dy + t*(v_0 - a/2) + t^2*a/2
		# Solving the above for "t" gives
		# delta = (v-a/2)^2 +2*dy*a
		# t_1 = (-(v_0 - a/2) + sqrt (delta))/a
		# t_2 = (-(v_0 - a/2) - sqrt (delta))/a
		# The smaller of the solutions represent the first time the
		# dinossaur crosses the difference "dy", and the greater represets
		# the second.

		delta = max((v-a/2)*(v-a/2)+2*dy*a, 0)**0.5
		roots = [(-(v-a/2)-delta)/a, (-(v-a/2)+delta)/a]
		return (min (roots), max (roots), )



class NeuralClassifier(KeyClassifier):
	def __init__(self, state, input, hidden, output):

		state_start = 0
		state_end = state_start + input*hidden[0]
		self.hidden_weights = [np.array (state[state_start:state_end]).reshape (input, hidden[0])]
		state_start = state_end
		state_end = state_start + hidden[0] 
		self.hidden_bias = [np.array (state[state_start:state_end]).reshape (1, hidden[0])]
		i = 0
		while i+1 < len(hidden):
			state_start = state_end
			state_end = state_start + hidden[i]*hidden[i+1] 
			self.hidden_weights.append(np.array(state[state_start:state_end]).reshape(hidden[i], hidden[i+1]))
			state_start = state_end
			state_end = state_start + hidden[i+1] 
			self.hidden_bias.append(np.array(state[state_start:state_end]).reshape(1, hidden[i+1]))
			i+=1

		state_start = state_end
		state_end = state_start + hidden[i]*output 
		self.output_weights = np.array (state[state_start:state_end]).reshape (hidden[i], output)
		state_start = state_end
		state_end = state_start + output 
		self.output_bias = np.array (state[state_start:state_end]).reshape (1, output)

	def keySelector(self, speed, obstacles, player):
		obs = []
		for i in range (len (obstacles)):
			distance = obstacles[i].rect.right - player.dino_rect.left - speed
			if distance > 0:
				obs.append (obstacles[i])
				if i + 1 < len (obstacles):
					if not (obstacles[i+1].__class__.__name__ == 'Bird' and obstacles[i+1].getHeight() > 50):
						obs.append (obstacles[i+1])
				break
		
		# The information given in the function input is delayed by one frame,
		# and must be corrected.
		dino_feet_pos = player.dino_rect.bottom - player.jump_vel
		dino_current_vertical_speed = player.jump_vel - player.jump_grav
		ground_top = player.Y_POS_DUCK+(player.dino_rect.bottom - player.dino_rect.top)
		distance_dino_bottom_ground_top = - (ground_top - dino_feet_pos)
		is_dino_jumping = int (player.dino_jump)

		distance_to_cross_obs = 0
		distance_to_reach_obs = 0
		distance_obs_top_dino_bottom = 0
		distance_to_reach_next_obs = 0
		next_obs_height = 0
		is_obstacle_high_bird = 0
		is_obstacle_large_cactus = 0

		if len (obs) > 0:
			is_obstacle_high_bird = int (obs[0].__class__.__name__ == 'Bird' and obs[0].getHeight() > 50)
			is_obstacle_large_cactus = int (obs[0].__class__.__name__ == 'LargeCactus')

			distance_to_cross_obs = obs[0].rect.right - player.dino_rect.left - speed
			distance_to_reach_obs = obs[0].rect.left - player.dino_rect.right - speed
			# Distances in the Y axis must be inverted, since
			# the Y axis grows down
			distance_obs_top_dino_bottom = - (obs[0].rect.top - dino_feet_pos)

		if len (obs) > 1:
			distance_to_reach_next_obs = obs[1].rect.left - player.dino_rect.right - speed
			next_obs_height = - (obs[1].rect.top - ground_top)

		# Time to cross horizontal distances is simple, as the increase in speed
		# is infrequent enough not to interfere in calculations. The time then
		# is simply distance/speed.
		time_to_reach_obs = distance_to_reach_obs/speed
		time_to_cross_obs = distance_to_cross_obs/speed
		time_to_be_above_obs = -100
		if distance_obs_top_dino_bottom > 0: 
			# Time to cross vertical distances need to consider acceleration,
			# and therefore need one of the solutions to a second degree
			# equation.
			time_to_be_above_obs = self.parabola_roots (distance_obs_top_dino_bottom, 17, -1.1)[0]
		time_to_reach_obs_top_holding_down = self.parabola_roots (distance_obs_top_dino_bottom, dino_current_vertical_speed, -4.4)[1]
		time_to_reach_ground_holding_down = self.parabola_roots (distance_dino_bottom_ground_top, dino_current_vertical_speed, -4.4)[1]
		time_to_reach_next_obs = 0
		time_to_be_above_next_obs_from_ground = 0
		time_to_be_above_next_obs_from_here = 0

		if distance_to_reach_next_obs > 0:
			time_to_reach_next_obs = distance_to_reach_next_obs/speed
			time_to_be_above_next_obs_from_ground = self.parabola_roots (next_obs_height, 17, -1.1)[0]
			time_to_be_above_next_obs_from_here = time_to_reach_ground_holding_down + time_to_be_above_next_obs_from_ground

		KEY = self.classify (time_to_cross_obs, is_obstacle_high_bird, time_to_reach_next_obs, time_to_be_above_next_obs_from_here, is_obstacle_large_cactus, time_to_reach_obs_top_holding_down, is_dino_jumping, dino_current_vertical_speed, time_to_reach_obs, time_to_be_above_obs)

		return KEY

	def classify (self, time_to_cross_obs, is_obstacle_high_bird, time_to_reach_next_obs, time_to_be_above_next_obs_from_here, is_obstacle_large_cactus, time_to_reach_obs_top_holding_down, is_dino_jumping, dino_current_vertical_speed, time_to_reach_obs, time_to_be_above_obs):

		input_layer = np.array ([time_to_cross_obs, is_obstacle_high_bird, time_to_reach_next_obs, time_to_be_above_next_obs_from_here, is_obstacle_large_cactus, time_to_reach_obs_top_holding_down, is_dino_jumping, dino_current_vertical_speed, time_to_reach_obs, time_to_be_above_obs]).reshape (1, 10)

		hidden_layer = input_layer
		for hw,hb in zip(self.hidden_weights, self.hidden_bias):
			hidden_layer =  np.matmul (hidden_layer, hw)
			hidden_layer += hb 
			hidden_layer = (hidden_layer > 0).astype (float)
		
		output_layer = np.matmul (hidden_layer, self.output_weights)
		output_layer += self.output_bias

		output = output_layer[0][0]

		if output < 0:
			# print('K_DOWN')
			return 'K_DOWN'
		else:
			# print('K_UP')
			return 'K_UP'


	def updateState(self, state):
		self.__init__ (state)


	def parabola_roots (self, dy, v, a):
		# The vertical position of the dinossaur is modified by "v" every frame.
		# Besides that, "v" gets modified by "a" also every frame.
		# We can use the above to calculate the position of the dinossaur at
		# any frame:
		# v(0) = v_0
		# y(0) = y_0
		# y(t) = y(t - 1) + v(t - 1)
		# v(t) = v(t - 1) + a
		# Solving the above we have
		# v(t) = v_0 + t*a
		# y(t) = y_0 + sum from i=0 to i=t-1 of v(i)
		# y(t) = y_0 + t*v_0 + (1*a + 2*a + ... + (t-1)*a)
		# y(t) = y_0 + t*v_0 + (t-1)*t*a/2
		# Since y(t) - y_0 = dy
		# 0 = -dy + t*v_0 + t^2*a/2 - t*a/2
		# 0 = -dy + t*(v_0 - a/2) + t^2*a/2
		# Solving the above for "t" gives
		# delta = (v-a/2)^2 +2*dy*a
		# t_1 = (-(v_0 - a/2) + sqrt (delta))/a
		# t_2 = (-(v_0 - a/2) - sqrt (delta))/a
		# The smaller of the solutions represent the first time the
		# dinossaur crosses the difference "dy", and the greater represets
		# the second.

		delta = max((v-a/2)*(v-a/2)+2*dy*a, 0)**0.5
		roots = [(-(v-a/2)-delta)/a, (-(v-a/2)+delta)/a]
		return (min (roots), max (roots), )



    # def getKey(self, obDistance, obHeight, scSpeed, obWidth, diHeight):

# class JulioClassifier (KeyClassifier):
#     def __init__(self, weight):
#         self.weight = weight

#     def	keySelector(self, speed, obstacles, player):
#         distance = 1500
#         obHeight = 0
#         obType = 2
#         obWidth = 0
#         diHeight = player.getXY()[1]
#         if len(obstacles) != 0:
#             xy = obstacles[0].getXY()
#             distance = xy[0]
#             obHeight = obstacles[0].getHeight()
#             obType = obstacles[0]
#             obWidth = obstacles[0].rect.width

#         op1, pos = self.neuronsConnections([distance, obWidth, obHeight, speed, diHeight], 5, 7, 0)
#         op2, pos = self.neuronsConnections(op1, 7, 7, pos)
#         op3, pos = self.neuronsConnections(op2, 7, 7, pos)
#         op4, pos = self.neuronsConnections(op3, 7, 7, pos)
#         lastOp, pos = self.neuronsConnections(op4, 7, 1, pos)# qtdWeight = 5*7+3*7*7+7 = 189
#         # print(lastOp[0])
#         # if lastOp[0] > 0.9:
#         if lastOp[len(lastOp) -1 ] > 0:
#             return "K_UP"
#         return "K_DOWN"
#         #return "K_NO"


#     # def keySelector(game_speed, obstacles, player):
#     #     this.getKey(distance, obHeight, game_speed, obWidth, player.getXY()[1])

#     def neuronsConnections(self, value, input, output, position):
#         # print('val=> ', value, '\tinput=> ', input, '\toutput', output, '\tposition', position)
#         neurons = []
#         i = 0
#         for _ in range(output):
#             i += 1
#             count = 0
#             for j in range(input):
#                 count += value[j] * self.weight[position]
#                 position += 1
#             # neurons.append(sigmoid(count)) # tanh
#             neurons.append(tanh(count)) # tanh
#         return [neurons, position]
   
#     # def updateWeight(self, weight):
#     #     self.weight = weight

