# Base Simulation
import numpy as np
import numpy.matlib
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# import matplotlib
# matplotlib.use('gtkagg')

# OpenAI Utils
import gym

# OpenAI Gym Class
#-----------------------------------
class MapSimEnv(gym.Env):
    def __init__(self, gridSize=[100, 100], numObjects=50, maxSize=10, numAgents=1, maxIters=1000):
        # Set Simulation Params
        self.gridSize = gridSize
        self.numAgents = numAgents
        self.maxSize = maxSize
        self.action_space = (0, 1, 2, 3)
        self.step_count = 0
        self.maxIters = maxIters
        self.numObjects = numObjects
        self.term_step = random.randint(maxIters/10, maxIters)
        # Generate Master Map
        self.grid = generate().world(gridSize, numObjects, maxSize)
        # Single Agent (return single object)
        if numAgents == 1:
            # Generate Agents
            self.agents = agent(gridSize, 1)
            # Generate Plots (should this go in _render()?)
            self.ax, self.cmap = plot().init()
        # Multi-Agent (returns object of objects)
        else:
            # Generate Agents
            self.agents = [agent(gridSize, i) for i in range(numAgents)]
            # Generate Plots (should this go in _render()?)
            self.ax, self.cmap = plot().multiInit(numAgents)

        print('Environment "mapSim-v1" Successfully Initialized')


    def step(self, action):
        self.step_count += 1
        self.agents = self.agents.update_position(action, self.grid)
        self.agents = self.agents.get_reward(self.gridSize)
        if self.step_count == self.term_step:
            self._reset()
        return self

    def _reset(self):
        self.reward = 0
        self.step_count = 0
        self.term_step = random.randint(self.maxIters/10, self.maxIters)
        self.grid = generate().world(self.gridSize, self.numObjects, self.maxSize)
        if self.numAgents == 1:
            # Generate Agents
            self.agents = agent(self.gridSize, 1)
            # Generate Plots (should this go in _render()?)
            self.ax, self.cmap = plot().init()
        # Multi-Agent (returns object of objects)
        else:
            # Generate Agents
            self.agents = [agent(self.gridSize, i) for i in range(self.numAgents)]
            # Generate Plots (should this go in _render()?)
            self.ax, self.cmap = plot().multiInit(self.numAgents)

        return self


    def _render(self, mode='human', close='False'):
        # Need to figure out what this does. Look at cartpole for example
        # Does it just render the environment? Or capture an image?
        # If it captures an image, how? In what format (using PIL?)
        # Will just display image for now
        plot().update(self.ax, self.cmap, self.grid)
        # This conflicts with the core.py/render class.
        # Line 150: close
        # Ask Dustin about this tomorrow
        return self


# Base Simulation:
#------------------------------------
# Map Encoding:
# 0: Unblocked
# 1: Blocked
# 2: Hidden
# 3+: Agent

class generate:
    def world(self, gridSize, numObjects, maxSize):
        grid = np.zeros(gridSize)
        for i in range(0, numObjects):
            center = [random.randint(0.1*gridSize[0], 0.9*gridSize[0]), random.randint(0.1*gridSize[1], 0.9*gridSize[1])]
            size = random.randint(0, maxSize)
            grid[center[0], center[1]] = 1
            loc = center
            for j in range(0, size):
                index = random.randint(0, 3)
                loc, grid = self.growObject(grid, loc, index, gridSize)

        return grid


    def growObject(self, grid, location, index, gridSize):
        if index == 0: # Up
            location[1] = location[1]+1
        elif index == 1: # Right
            location[0] = location[0]+1
        elif index == 2: # Down
            location[1] = location[1]-1
        elif index == 3: # Left
            location[0] = location[0]-1
        if location[0] < gridSize[0] and location[1] < gridSize[1]:
            grid[location[0], location[1]] = 1

        return location, grid

class agent:
    def __init__(self, gridSize, ID):
        self.id = ID+3
        self.location = (random.randint(1, gridSize[0]-1), random.randint(1, gridSize[0]-1))
        # self.location = (random.randint(45, 55), random.randint(45, 55))
        self.map = numpy.matlib.repmat(2, gridSize[0], gridSize[1])
        self.map[self.location[0], self.location[1]] = ID+2
        self.reward = 0
        self.gridSize = gridSize

    def step(self, grid, gridSize, ax, cmap):
        self.map[self.location[0], self.location[1]] = grid[self.location[0], self.location[1]]
        # self = self.selectAction(grid, gridSize)
        self.map[self.location[0], self.location[1]] = self.id
        self = self.view(grid)
        ax = plot().update(ax, cmap, self.map)
        self = self.reward(gridSize)
#         img = plot().getImage()
#         return self, ax, img
        return self, ax

    def update_position(self, action, grid): # Incomplete
        self.validate_action(action, grid)
        self = self.view(grid)
        return self

    def validate_action(self, action, grid): # Incomplete
        valid = 'false'
        tempLocation = self.direction(action)
        if (tempLocation[0] > 1 and tempLocation[1] > 1 and tempLocation[0] < self.gridSize[0]-1 and tempLocation[1] < self.gridSize[1]-1):
            if grid[tempLocation[0], tempLocation[1]] == 0:
                self.location = tempLocation
                valid = 'true'
        if valid == 'false':
            self.reward -= 1
        return self

    def direction(self, inpt):
        location = list(self.location)
        if inpt == 0: # Up
            location[1] = location[1]+1
        elif inpt == 1: # Right
            location[0] = location[0]+1
        elif inpt == 2: # Down
            location[1] = location[1]-1
        elif inpt == 3: # Left
            location[0] = location[0]-1

        return location

    def view(self, grid):
        sGrid = self.map[self.location[0]-1:self.location[0]+2, self.location[1]-1:self.location[1]+2]
        for i in range(0, 3):
            for j in range(0, 3):
                if sGrid[i, j] == 2:
                    self.map[self.location[0]-1+i, self.location[1]-1+j] = grid[self.location[0]-1+i, self.location[1]-1+j]

        return self

    def selectAction(self, grid, gridSize): # Update to check for overlap. Must take both agents as input
        valid = 'false'
        while valid == 'false':
            action = random.randint(0, 3)
            tempLocation = self.direction(action)
            if (tempLocation[0] > 1 and tempLocation[1] > 1 and tempLocation[0] < gridSize[0]-1 and tempLocation[1] < gridSize[1]-1):
                if grid[tempLocation[0], tempLocation[1]] == 0:
                    valid = 'true'
                    self.location = tempLocation
                    return self

    def multiAgentStep(self, agents, numAgents, grid, gridSize, met, cmap):
        for i in range(numAgents):
            agents[i] = agents[i].step(grid, gridSize)
        met, cmap = self.meet(agents, gridSize, met, cmap)
        if met == 'true':
            agents = self.shareMap(agents, gridSize)

        return agents, met, cmap

    def meet(self, agents, gridSize, met, cmap):
        if (abs(agents[0].location[0]-agents[1].location[0]) <= 1 and abs(agents[0].location[1]-agents[1].location[1] <= 1)):
            met = 'true'
            cmap = ListedColormap(['w', 'b', 'k', 'r', 'g'])
        return met, cmap

    def shareMap(self, agents, gridSize):
        for i in range(0, gridSize[0]):
            for j in range(0, gridSize[1]):
                if (agents[0].map[i, j] < 2 and [i, j] != agents[1].location) or (agents[0].map[i, j] == agents[0].id and agents[0].location != agents[1].location):
                    agents[1].map[i, j] = agents[0].map[i, j]
                elif sum(numpy.matlib.rempat(2, grid))(agents[1].map[i, j] < 2 and [i, j] != agents[0].location) or (agents[1].map[i, j] == agents[1].id and agents[1].location != agents[0].location):
                    agents[0].map[i, j] = agents[1].map[i, j]

        return agents

    def get_reward(self, gridSize):
        #r = sum(abs(sum(numpy.matlib.repmat(2, gridSize[0], gridSize[1])-agents[0].map)+sum(numpy.matlib.repmat(2, gridSize[0], gridSize[1])-agents[1].map)))/(gridSize[0]*gridSize[1]*4)
        self.reward = sum(sum(numpy.matlib.repmat(2, gridSize[0], gridSize[1])-self.map))/[gridSize[0]*gridSize[1]*2]
        return self

class plot:
    def init(self):
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cmap = ListedColormap(['w', 'b', 'k', 'r'])
        return ax, cmap

    def multiInit(self, numAgents):
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax = (ax1, ax2)
        cmap = ListedColormap(['w', 'b', 'k', 'r'])

        return ax, cmap

    def update(self, ax, cmap, grid):
        ax.matshow(grid, cmap=cmap)
        plt.draw()
        plt.pause(0.1)
        plt.cla()
        return ax

    def multiUpdate(self, ax, cmap, agents, numAgents, fig):
        for i in range(numAgents):
            ax[i].matshow(agents[i].map, cmap=cmap)
            plt.draw()
        plt.pause(0.00000001)
        plt.cla()
        return ax

    def showGrid(self, grid) :
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cmap = ListedColormap(['w', 'k'])
        ax.matshow(grid, cmap=cmap)
        plt.draw()
        plt.pause(1)
        plt.close(fig)

#     def getImage(self):
#         img = plt.savefig()
#         return img
