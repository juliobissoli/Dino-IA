from math import tanh, exp
from unittest import result
import pygame
import os
import random
import time
from sys import exit
import random

from classifier import FrancoNeuralClassifier

pygame.init()

# Valid values: HUMAN_MODE or AI_MODE
GAME_MODE = "AI_MODE"

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus4.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


class Dinosaur:
    X_POS = 90
    Y_POS = 330
    Y_POS_DUCK = 355
    JUMP_VEL = 17
    JUMP_GRAV = 1.1

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = 0
        self.jump_grav = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, userInput):
        if self.dino_duck and not self.dino_jump:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 20:
            self.step_index = 0

        if userInput == "K_UP" and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput == "K_DOWN" and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif userInput == "K_DOWN":
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = True
        elif not (self.dino_jump or userInput == "K_DOWN"):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_duck:
            self.jump_grav = self.JUMP_GRAV * 4
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel
            self.jump_vel -= self.jump_grav
        if self.dino_rect.y > self.Y_POS + 10:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.jump_grav = self.JUMP_GRAV
            self.dino_rect.y = self.Y_POS

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

    def getXY(self):
        return (self.dino_rect.x, self.dino_rect.y)


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle():
    def __init__(self, image, type):
        super().__init__()
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()

        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < - self.rect.width:
            obstacles.pop(0)

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)

    def getXY(self):
        return (self.rect.x, self.rect.y)

    def getHeight(self):
        return y_pos_bg - self.rect.y

    def getType(self):
        return (self.type)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 345


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)

        # High, middle or ground
        if random.randint(0, 3) == 0:
            self.rect.y = 345
        elif random.randint(0, 2) == 0:
            self.rect.y = 260
        else:
            self.rect.y = 300
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 19:
            self.index = 0
        SCREEN.blit(self.image[self.index // 10], self.rect)
        self.index += 1


class KeyClassifier:
    def __init__(self, state):
        pass

    def keySelector(self, distance, obHeight, speed, obType):
        pass

    def updateState(self, state):
        pass


def first(x):
    return x[0]


class KeySimplestClassifier(KeyClassifier):
    def __init__(self, state):
        self.state = state

    def keySelector(self, distance, obHeight, speed, obType):
        self.state = sorted(self.state, key=first)
        for s, d in self.state:
            if speed < s:
                limDist = d
                break
        if distance <= limDist:
            if isinstance(obType, Bird) and obHeight > 50:
                return "K_DOWN"
            else:
                return "K_UP"
        return "K_NO"

    def updateState(self, state):
        self.state = state

def sigmoid(x):
    if x >= 0:
        z = exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = exp(x)
        sig = z / (1 + z)
        return sig

class KeyRicClassifier(KeyClassifier):
    def __init__(self, weight):
        self.weight = weight

    def keySelector(self, obDistance, obHeight, scSpeed, obWidth, diHeight):
        op1, pos = self.neuronsXtoY([obDistance, obWidth, obHeight, scSpeed, diHeight], 5, 7, 0)
        op2, pos = self.neuronsXtoY(op1, 7, 7, pos)
        op3, pos = self.neuronsXtoY(op2, 7, 7, pos)
        op4, pos = self.neuronsXtoY(op3, 7, 7, pos)
        lastOp, pos = self.neuronsXtoY(op4, 7, 1, pos)# qtdWeight = 5*7+3*7*7+7 = 189

        if lastOp[0] > 0.9:
            return "K_UP"
        return "K_DOWN"
        #return "K_NO"

    def neuronsXtoY(self, value, input, output, position):
        neurons = []
        for it in range(output):
            sum = 0
            for it2 in range(input):
                sum += value[it2] * self.weight[position]
                position += 1
            neurons.append(sigmoid(sum))
        return [neurons, position]

    def updateWeight(self, weight):
        self.weight = weight


def playerKeySelector():
    userInputArray = pygame.key.get_pressed()

    if userInputArray[pygame.K_UP]:
        return "K_UP"
    elif userInputArray[pygame.K_DOWN]:
        return "K_DOWN"
    else:
        return "K_NO"


def playGame():
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    random.seed (1481)


    run = True
    clock = pygame.time.Clock()
    player = Dinosaur()
    cloud = Cloud()
    game_speed = 10
    x_pos_bg = 0
    y_pos_bg = 383
    points = 0
    font = pygame.font.Font('freesansbold.ttf', 20)
    obstacles = []
    death_count = 0
    spawn_dist = 0

    def score():
        global points, game_speed
        points += 0.25
        if points % 100 == 0:
            game_speed += 1

        text = font.render("Points: " + str(int(points)), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        SCREEN.blit(text, textRect)

    def background():
        global x_pos_bg, y_pos_bg
        image_width = BG.get_width()
        SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
        SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
        if x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            x_pos_bg = 0
        x_pos_bg -= game_speed
    
    userInput = "K_NO"

	PAUSE = False
	ONE_FRAME = False

    while run:
        if ONE_FRAME:
            PAUSE = True
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                print ()
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    PAUSE = not PAUSE
                    ONE_FRAME = False
                
                elif event.button == 3:
                    ONE_FRAME = True
                    PAUSE = False
        
        if PAUSE:
            continue


    # while run:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             run = False
    #             exit()

        SCREEN.fill((255, 255, 255))
        
        distance = 1500
        obHeight = 0
        obType = 2
        if len(obstacles) != 0:
            # xy = obstacles[0].getXY()
            # obDistance = xy[0]
            # obHeight = obstacles[0].getHeight()
            # obType = obstacles[0]
            # obWidth = obstacles[0].rect.width
            xy = obstacles[0].getXY()
            distance = xy[0]
            obHeight = obstacles[0].getHeight()
            obType = obstacles[0]

        if GAME_MODE == "HUMAN_MODE":
            userInput = playerKeySelector()


        else:
            userInput = aiPlayer.keySelector(game_speed, obstacles, player)
            # userInput = aiPlayer.keySelector(obDistance, obHeight, game_speed, obWidth, player.getXY()[1])

        if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 5) == 5:
                obstacles.append(Bird(BIRD))

        player.update(userInput)
        player.draw(SCREEN)

        for obstacle in list(obstacles):
            obstacle.update()
            obstacle.draw(SCREEN)
            
        background()

        cloud.draw(SCREEN)
        cloud.update()

        score()

        clock.tick(600)#60
        pygame.display.update()

        for obstacle in obstacles:
            if player.dino_rect.colliderect(obstacle.rect):
                pygame.time.delay(200)#2000
                death_count += 1
                return points


# Change State Operator

def change_state(state, position, vs, vd):
    aux = state.copy()
    s, d = state[position]
    ns = s + vs
    nd = d + vd
    if ns < 15 or nd > 1000:
        return []
    return aux[:position] + [(ns, nd)] + aux[position + 1:]

def change_state_Ric(state, position):
    aux = state.copy()
    s = state[position]
    vs = random.randint(-20,20)
    ns = s + vs
    if ns < -100:
        ns += 200
    if ns > 100:
        ns -= 200
    newState = aux[:position] + [(ns)] + aux[position + 1:]
    return mutation(newState, 0.1)

# Neighborhood

def generate_neighborhood(state):
    neighborhood = []
    state_size = len(state)
    for i in range(state_size):
        d1 = random.randint(-5,5) 
        d2 = random.randint(-5,5)
        d3 = random.randint(-5,5)
        d4 = random.randint(-5,5)

        new_states = [change_state(state, i, d1), change_state(state, i, d2), change_state(state, i, d3), change_state(state, i, d4)]
        for s in new_states:
            if s != []:
                neighborhood.append(s)
    return neighborhood

def generate_neighborhood_Ric(state, p):
    neighborhood = []
    state_size = len(state)
    for j in range(1):
        for i in range(state_size):
            if random.randint(0,100) < 100*p:
                state_to_change = state
                new_states = [change_state_Ric(state_to_change, i)]
                for s in new_states:
                    if s != []:
                        neighborhood.append(s)
    return neighborhood

def generate_x_neighborhood_Ric(state, x):
    neighborhood = []
    state_size = len(state)
    for it in range(x):
        posToGet = random.randint(0,state_size-1)
        state_to_change = state
        new_states = [change_state_Ric(state_to_change, posToGet)]
        for s in new_states:
            if s != []:
                neighborhood.append(s)
    return neighborhood

def generate_states(states, neighborhoodQtd, bestStatesQtd, crossoverQtd):
    bests = states[0 : bestStatesQtd]
    
    auxNeighborhood = []
    for it in range(bestStatesQtd):
        auxNeighborhood += generate_x_neighborhood_Ric(bests[it][1], neighborhoodQtd)
        auxNeighborhood.append(bests[it][1])

    auxCrossover = []
    for it in range(bestStatesQtd):
        for it2 in range(bestStatesQtd):
            if it == it2:
                continue
            auxCrossover += crossover(bests[it][1], bests[it2][1], crossoverQtd)

    return auxNeighborhood + auxCrossover

# Mutation

def mutation(state, mutatationRate):
    aux = state.copy()
    state_size = len(state)
    for it in range(state_size):
        rand = random.randint(0, 100)
        if rand < mutatationRate*100:
            aux[it] =  random.randint(-100, 100)
    return aux

def mutationAll(states, mutatationRate):
    aux = []
    states_qtd = len(states)
    for it in range(states_qtd):
        aux.append(mutation(states[it][1], mutatationRate))
        
# Crossover

def crossover(state1, state2, childrensQtd):
    childrens = []
    for it in range(childrensQtd):
        randPos = random.randint(0, len(state1))
        newState = state1[:randPos] + state2[randPos:]
        childrens.append(mutation(newState, 0.1))
    return childrens

# Gradiente Ascent

def gradient_ascent(state, max_time):
    start = time.process_time()
    res, max_value = manyPlaysResults(1)
    better = True
    end = 0
    while better and end - start <= max_time:
        neighborhood = generate_neighborhood(state)
        better = False
        for s in neighborhood:
            aiPlayer = KeyRicClassifier(s)
            print(s)
            res, value = manyPlaysResults(1)
            if value > max_value:
                state = s
                max_value = value
                better = True
        end = time.process_time()
    return state, max_value

def begin(max_time):
    global aiPlayer
    f = open("log.txt", "w")
    f.write("")
    f.close()
    plays = 3
    start = time.process_time()
    res = 0
    states = []
    better = True
    end = 0
    generation = 1
    
    for it in range(30):
        newState = [random.randint(-100, 100) for col in range(189)]
        aiPlayer = KeyRicClassifier(newState)
        res, value = manyPlaysResults(plays)
        #print(newState, generation, it+1, value)
        print(generation, it+1, value)
        states.append([value, newState])

    states.sort()
    states.reverse()
    saveStates(states, generation, time.process_time() - start)

    generation+=1
    while end - start <= max_time:
        it = 0
        
        print("Time: ", time.process_time() - start)
        
        neighborhood = generate_states(states, 4, 3, 5) #gerar (4*3) + 3 + (5 crossovers * 3 * 2) = 45
        states.clear()

        for s in neighborhood:
            it+= 1
            aiPlayer = KeyRicClassifier(s)
            
            res, value = manyPlaysResults(plays)
            #print(s, generation, it, value)
            print(generation, it, value)
            states.append([value, s])
        end = time.process_time()
        states.sort()
        states.reverse()
        saveStates(states, generation, time.process_time() - start)
        generation+=1
    best_state = states[0][1]
    best_value = states[0][0]
    print(best_state)
    return best_state, best_value

def saveStates(states, gen, time):
    f = open("log.txt", "a")
    f.write("Generation: " + str(gen) + "\n")
    f.write("Time: " + str(time) + "\n\n")
    for state in states:
        f.write(str(state) + "\n")
    f.write("\n\n\n")
    f.close()


from scipy import stats
import numpy as np


def manyPlaysResults(aiPlayer, rounds):
    results = []
    for round in range(rounds):
        results += [playGame(aiPlayer, round)]
    npResults = np.asarray(results)
    return (results, npResults.mean() - npResults.std())


# def main():
#     global aiPlayer
#     best_state, best_value = begin(24*60*60) # rodar por 24 horas
#     aiPlayer = KeyRicClassifier(best_state)
#     res, value = manyPlaysResults(30)
#     npRes = np.asarray(res)
#     print(res, npRes.mean(), npRes.std(), value)
#     f = open("log.txt", "a")
#     f.write("Result: \n" + str(res) + "\nMean: " + str(npRes.mean()) + "\nStd: " + str(npRes.std()) + "\nValue: " + str(value))


# main()

# 5-6-6-6-2: 369.34166666666664 187.4603233946379 181.88134327202874
# 5-7-7-7-2:
#


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
