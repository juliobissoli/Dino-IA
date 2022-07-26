from math import tanh, exp, sqrt
import pygame
import os
import random
import time
from sys import exit

from scipy import stats
import numpy as np

import multiprocessing as mp
from functools import partial

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


class KeyNeuralClassifier(KeyClassifier):
    def __init__(self, weight):
        self.weight = weight

    def getKey(self, obDistance, obHeight, scSpeed, obWidth, diHeight, obType):
        op1, pos = self.neuronsConnections([obDistance, obWidth, obHeight, scSpeed, diHeight], 5, 5, 0)
        # print(op1, pos)
        # op2, pos = self.neuronsConnections(op1, 5, 5, pos)
        # op3, pos = self.neuronsConnections(op2, 7, 7, pos)
        # op4, pos = self.neuronsConnections(op2, 5, 5, pos)
        lastOp, pos = self.neuronsConnections(op1, 5, 3, pos)# Quantidade de pessos e dada por = 5*7+2*7*7+7 = 140

        keys = ["K_UP", "K_DOWN", "K_NO"]
        return keys[np.argmax(lastOp)]
        # if lastOp[len(lastOp) -1 ] < 0:
        #     return "K_UP"
        # return "K_DOWN"
        #return "K_NO"

    def neuronsConnections(self, value, input, output, position):
        # print('self=>', self.weight, 'len weight', len(self.weight), 'val=> ', value, '\tinput=> ', input, '\toutput', output, '\tposition', position)
        neurons = []
        i = 0
        for _ in range(output):
            i += 1
            count = 0
            for j in range(input):
                # print('j idx', j)
                # print('w idx', position)
                count += value[j] * self.weight[position]
                position += 1
            # neurons.append(sigmoid(count)) # tanh
            neurons.append(tanh(count)) # tanh
        # print(neurons)
        return [neurons, position]
   
    # def updateWeight(self, weight):
    #     self.weight = weight


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

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                exit()

        SCREEN.fill((255, 255, 255))

        distance = 1500
        obHeight = 0
        obType = 2
        obWidth = 0
        if len(obstacles) != 0:
            xy = obstacles[0].getXY()
            distance = xy[0]
            obHeight = obstacles[0].getHeight()
            obType = obstacles[0]
            obWidth = obstacles[0].rect.width

        if GAME_MODE == "HUMAN_MODE":
            userInput = playerKeySelector()
        else:
            # userInput = aiPlayer.keySelector(distance, obHeight, game_speed, obType)
            # print(distance, obHeight, game_speed, obWidth, player.getXY()[1], obType)
            userInput = aiPlayer.getKey(distance, obHeight, game_speed, obWidth, player.getXY()[1], obType)

        if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            # obstacles.append(SmallCactus(SMALL_CACTUS))
            

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

        clock.tick(600)
        pygame.display.update()

        for obstacle in obstacles:
            if player.dino_rect.colliderect(obstacle.rect):
                pygame.time.delay(200)
                death_count += 1
                return points



def changeState(state, position):
    copyState = state.copy()
    # print(state)
    s = state[position] # peso da ia no idx position
    vs = random.randint(-20,20)
    ns = s + vs
    if ns < -100:
        ns += 200
    if ns > 100:
        ns -= 200
    newState = copyState[:position] + [(ns)] + copyState[position + 1:]
    return mutation(newState, 0.1)


def generateKNeighborhoods(state, x):
    neighborhoodList = []
    # stateLen = len(state)
    for _ in range(x):
        posToGet = random.randint(0,len(state) - 1)
        state_to_change = state
        new_states = [changeState(state_to_change, posToGet)]
        for s in new_states:
            if s != []:
                neighborhoodList.append(s)
    return neighborhoodList

def createStates(states, neighborhoodQtd, bestStatesQtd, crossoverQtd):
    bests = states[0 : bestStatesQtd]
    
    auxNeighborhood = []
    for i in range(bestStatesQtd):
        auxNeighborhood += generateKNeighborhoods(bests[i][1], neighborhoodQtd)
        auxNeighborhood.append(bests[i][1])

    auxCrossover = []
    for i in range(bestStatesQtd):
        for j in range(bestStatesQtd):
            if i == j:
                continue
            auxCrossover += crossover(bests[i][1], bests[j][1], crossoverQtd)

    print ("bests = ", bests)
    print ("nei = ", auxNeighborhood)
    print ("cross = ", auxCrossover)
    print ("bests2 = ", bests)

    return [x[1] for x in bests] + auxNeighborhood + auxCrossover


def mutation(state, mutatationRate):
    aux = state.copy()
    state_size = len(state)
    for i in range(state_size):
        rand = random.randint(0, 100)
        if rand < mutatationRate*100:
            aux[i] +=  random.randint(-10, 10)
            if aux[i] > 100:
                aux[i] = 100
            if aux[i] < -100:
                aux[i] = -100
    return aux

def crossover(firstState, secondState, rangeChildrens):
    childrens = []
    for _ in range(rangeChildrens):
        randPos = random.randint(0, len(firstState))
        newState = firstState[:randPos] + secondState[randPos:]
        childrens.append(mutation(newState, 0.1))
    return childrens


def evaluateNeighbor(s, gen, manyPlays):
    
    global aiPlayer
    aiPlayer = KeyNeuralClassifier(s)
    
    res, value = manyPlaysResults(manyPlays)
    #print(s, generation, it, value)
    print(generation, it, value)
    return [value, s]

# best_state, best_value = playToIA(first_states, max_time, manyPlays, start, generation,) # rodar por 24 horas
def playToIA(states, max_time, manyPlays, start, generation):
    end = 0
    # states, aiPlayers = generateFistrState(generation, manyPlays, start)
    generation+=1
    while end - start <= max_time:
        it = 0
        
        print("Time: ", time.process_time() - start)
        
        neighborhood = createStates(states, 4, 3, 5) #gerar (4*3) + 3 + (5 crossovers * 3 * 2) = 45
        # (4 * 2) + 2 + (5 * 2 * 2) = 8 + 2 + 20 = 30
        states.clear()

        # with mp.Pool() as pool:
        #     states = pool.map(partial(evaluateNeighbor, it = it, gen = generation, manyPlays = manyPlays), neighborhood)

        for s in neighborhood:
            it+= 1
            global aiPlayer
            aiPlayer = KeyNeuralClassifier(s)
            
            res, value = manyPlaysResults(manyPlays)
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

def generateFistrState(generation, qtdPlayers, start):
    print('playToIA')
    states = []
    for i in range(30):
        # newState = [random.randint(-100, 100) for _ in range(46)]  # 140 # 31 para ter espaço
        newState = list(np.random.rand(46) * 200 - 100)
        global aiPlayer
        aiPlayer = KeyNeuralClassifier(newState)

        res, value = manyPlaysResults(qtdPlayers)
        #print(newState, generation, it+1, value)
        print(generation, i+1, value)
        states.append([value, newState])

    states.sort()
    states.reverse()
    saveStates(states, generation, time.process_time() - start)

    return states


def saveStates(states, gen, time):
    f = open("log.txt", "a")
    f.write("Generation: " + str(gen) + "\n")
    f.write("Time: " + str(time) + "\n\n")
    for state in states:
        f.write(str(state) + "\n")
    f.write("\n\n\n")
    f.close()

def manyPlaysResults(rounds):
    results = []
    for round in range(rounds):
        results += [playGame()]
    npResults = np.asarray(results)
    return (results, sqrt((npResults.mean() - npResults.std())**2))


def main():
    manyPlays = 10
    qtdPlayers = 10
    qtdRunsPerPlayer = 10
    start = time.process_time()
    states = []
    end = 0
    generation = 1
    max_time =  8*60*60
    # first_states = generateFistrState(generation, manyPlays, start)
    # best_state, best_value = playToIA(first_states, max_time, manyPlays, start, generation) # rodar por 24 horas
    first_states = generateFistrState(generation, qtdPlayers, start)
    # for i in range(max_iter):
    best_state, best_value = playToIA(first_states, max_time, qtdRunsPerPlayer, start, generation) # rodar por 24 horas
    # aiPlayer = KeyNeuralClassifier(best_state)
    res, value = manyPlaysResults(30)
    npRes = np.asarray(res)
    print(res, npRes.mean(), npRes.std(), value)
    f = open("log.txt", "a")
    f.write("Result: \n" + str(res) + "\nMean: " + str(npRes.mean()) + "\nStd: " + str(npRes.std()) + "\nValue: " + str(value))


main()
