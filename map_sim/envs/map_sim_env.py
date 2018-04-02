# Base Simulation
# import matplotlib
# matplotlib.use('gtkagg')
import numpy as np
import numpy.matlib
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import misc
from skimage.color import rgb2grey
# from skimage.transform import resize
from PIL import Image
# import matplotlib
# matplotlib.use('gtkagg')
import time

# OpenAI Utils
import gym
from gym import spaces

# OpenAI Gym Class
#-----------------------------------
class MapSimEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    def __init__(self, gridSize=[20, 20], numObjects=20, maxSize=8, numAgents=2, maxIters=300, interactive='False', test='False'):
        # random.seed(100) # TESTING SEED - Do Not Seed During Training
        # Set Simulation Params
        self.gridSize = gridSize
        self.numAgents = numAgents
        self.maxSize = maxSize
        self.test = test
        # self.action_space = (0, 1, 2, 3)
        self.action_space = spaces.Discrete(4)
        self.step_count = 0
        self.maxIters = maxIters
        self.numObjects = numObjects
        # self.term_step = random.randint(maxIters/2, maxIters)
        self.term_step = maxIters
        self.interactive = interactive
        # Generate Master Map
        self.grid = generate().world(gridSize, numObjects, maxSize)
        # plot().showGrid(self.grid)
        self.simID = random.randint(1000000, 2000000)
        # Single Agent (return single object)
        if numAgents == 1:
            # Generate Agents
            self.agents = agent(gridSize, self.grid, 1)
            # Generate Plots (should this go in _render()?)
            self.ax, self.cmap, self.fig = plot().init(interactive)
        # Multi-Agent (returns object of objects)
        else:
            # Generate Agents
            self.agents = []
            for i in range(numAgents):
                self.agents.append(agent(self.agents, gridSize, self.grid, i))
                # print(self.agents)
            # self.agents = [agent(self.agents, gridSize, self.grid, i) for i in range(numAgents)]
            # print(self.agents)
            # Generate Plots (should this go in _render()?)
            self.fig, self.ax, self.cmap = plot().multiInit(self.agents, self.interactive)

        # print('Environment "mapSim-v1" Successfully Initialized')


    def _step(self, action, ind):
        self.step_count += 1
        if self.numAgents == 1:
            self.agents = self.agents.update_position(self.agents, action, self.grid)
            self.agents = self.agents.get_reward(self.gridSize)
            reward = self.agents.reward
            self.ax, img = self.get_image()
        else:
            self.agents[ind] = self.agents[ind].update_position(self.agents, action, self.grid, not ind)
            self.agents, self.cmap = self.agents[ind].shareMap(self.agents, self.gridSize, ind)
            self.agents[ind] = self.agents[ind].get_reward(self.gridSize)
            # print('Agent 0 Position: ' + str(self.agents[0].location))
            # print('Agent 1 Position: ' + str(self.agents[1].location))
            reward = self.agents[ind].reward
            self.ax, img = self.get_image()
            # img = []
            # reward = []
            # for i in range(self.numAgents):
                # print('action= '+ str(action))
                # print('agents= ' + str(self.agents))
                # self.agents[i] = self.agents[i].update_position(action[i], self.grid)
                # self.agents[i] = self.agents[i].get_reward(self.gridSize)
                # reward.append(self.agents[i].reward)
                # self.ax, img_ = self.get_image()
                # img.append(img_)
        # print(self.step_count)
        if self.step_count == self.term_step:
            done = 1
            if self.numAgents == 1:
                self.percent_explored = self.agents.percent_exp(self.gridSize)
                print('Percent Explored: ' + str(self.agents.percent_exp(self.gridSize)) + '%')
            else:
                self.percent_explored = [[],[]]
                for i in range(self.numAgents):
                    self.percent_explored[i] = self.agents[i].percent_exp(self.gridSize)
                    # if self.interactive == 'True':
                    print('Agent ' + str(i) + ' | ' + 'Percent Explored: ' + str(self.agents[i].percent_exp(self.gridSize)) + '%')
            if self.test == 'False': # Leave false for multi-agent testing
                self._reset()
        else:
            done = 0
            # done = bool(0)

        # print('env obs')
        # print(np.asarray(img).shape)
        # print('env reward')
        # print(reward)
        return img, reward, done, {}

    def _reset(self):
        self.reward = 0
        self.step_count = 0
        # self.term_step = random.randint(self.maxIters/10, self.maxIters)
        plt.close('all') # OK for A2C. Will have to change for A3C
        # plt.close(self.fig)
        self.grid = generate().world(self.gridSize, self.numObjects, self.maxSize)
        if self.numAgents == 1:
            # Generate Agents
            self.agents = agent(self.gridSize, self.grid, 1)
            # Generate Plots (should this go in _render()?)
            self.ax, self.cmap, self.fig = plot().init(self.interactive)
            # if self.test == 'True':
            #     plot().showGrid(self.grid)
        # Multi-Agent (returns object of objects)
        else:
            # Generate Agents
            self.agents = []
            # self.agents = [agent(self.agents, self.gridSize, self.grid, i) for i in range(self.numAgents)]
            for i in range(self.numAgents):
                self.agents.append(agent(self.agents, self.gridSize, self.grid, i))
                # print(self.agents)
            # Generate Plots (should this go in _render()?)
            # self.cmap = ListedColormap(['w', 'b', 'k', 'r'])
            self.fig, self.ax, self.cmap = plot().multiInit(self.agents, self.interactive)
            # self.cmap = ListedColormap(['w', 'b', 'k', 'r'])
            self.ax = plot().multiUpdate(self.ax, self.cmap, self.agents, self.numAgents, self.interactive)

        self.ax, img = self.get_image()

        return self, img

    def get_image(self): # Needs updating
        img_shape = (84, 84, 3)
        if self.numAgents == 1:
            self.ax = plot().update(self.ax, self.cmap, self.agents.map, self.interactive)
            img = self.render_img()
            img = self.process_img(img, img_shape)
            img = np.asarray(img.convert('RGB'))
        else:
            self.ax = plot().multiUpdate(self.ax, self.cmap, self.agents, self.numAgents, self.interactive)
            for i in range(self.numAgents):
                self.agents[i].img = self.render_img(agent_ID=i)
                self.agents[i].img = self.process_img(self.agents[i].img, img_shape)
                self.agents[i].img = np.asarray(self.agents[i].img.convert('RGB'))

            img = []
            for i in range(self.numAgents):
                img.append(self.agents[i].img)
        return self.ax, img

    def process_img(self, img, img_shape):
        # img = rgb2grey(img)
        # img = resize(img, img_shape, order=3, preserve_range=False, clip=True)
        img = img.resize(img_shape[0:2])
        return img

    def render_img(self, agent_ID=1): # rewrite to avoid writing img to disk
        if self.numAgents == 1:
            filename = 'img' + str(self.simID) + '.png'
            self.fig.savefig(filename, bbox_inches='tight')
            # img = misc.imread(filename)
            img = Image.open(filename)
        else:
            filename = 'img' + str(self.simID) + str(agent_ID) + '.png'
            self.fig[agent_ID].savefig(filename, bbox_inches='tight')
            img = Image.open(filename)
            # img = Image.get_current_fig_manager().canvas
        # canvas = FigureCanvas(self.fig)
        # canvas.draw()
        # img = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
        return img


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
            # center = [random.randint(0.2*gridSize[0], 0.8*gridSize[0]), random.randint(0.2*gridSize[1], 0.8*gridSize[1])]
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
        elif index == 1: # Right        self.interactive = interactive
            location[0] = location[0]+1
        elif index == 2: # Down
            location[1] = location[1]-1
        elif index == 3: # Left
            location[0] = location[0]-1
        if location[0] < gridSize[0] and location[1] < gridSize[1]:
            grid[location[0], location[1]] = 1

        return location, grid

class agent:
    def __init__(self, agents, gridSize, grid, ID):
        self.id = ID+3
        # Need to add function that ensures starting position is valid
        # self.location = (random.randint(1, gridSize[0]-1), random.randint(1, gridSize[0]-1))
        self.location = self.init_location(agents, gridSize, grid, ID)
        self.map = numpy.matlib.repmat(2, gridSize[0], gridSize[1])
        self.map[self.location[0], self.location[1]] = ID+2
        self.reward = 0
        self.r_old = 0
        self.gridSize = gridSize
        self.img= []
        self.old_loc = self.location

    def init_location(self, agents, gridSize, grid, ID):
        valid = 'false'
        while valid == 'false':
            tempLocation = (random.randint(1, gridSize[0]-1), random.randint(1, gridSize[0]-1))
            if (tempLocation[0] > 1 and tempLocation[1] >= 1 and tempLocation[0] < gridSize[0]-1 and tempLocation[1] < gridSize[1]-1):
                if grid[tempLocation[0], tempLocation[1]] == 0:
                    # print('ID = '+str(ID))
                    # print(agents)
                    if ID == 0:
                        valid = 'true'
                        location = tempLocation
                    elif abs(tempLocation[0]-agents[0].location[0])>1 and abs(tempLocation[1]-agents[0].location[1])>1:
                        # print('First agents location: '+str(agents[0].location))
                        # print('Second agents location: '+str(tempLocation))
                        valid = 'true'
                        location = tempLocation
                    # else:
                        # print('Invalid Initial Location Selected - Retrying')
        return location


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

    def update_position(self, agents, action, grid, other_ind): # Incomplete
        self.map[self.location[0], self.location[1]] = grid[self.location[0], self.location[1]]
        self = self.validate_action(agents, action, grid, other_ind)
        self.map[self.location[0], self.location[1]] = self.id
        self = self.view(grid)
        return self


    def validate_action(self, agents, action, grid, other_ind): # Incomplete
        self.valid = 'false'
        tempLocation = self.direction(action)
        if (tempLocation[0] > 1 and tempLocation[1] > 1 and tempLocation[0] < self.gridSize[0]-1 and tempLocation[1] < self.gridSize[1]-1):
            if grid[tempLocation[0], tempLocation[1]] == 0:
                if tempLocation[0] != agents[other_ind].location[0] and tempLocation[1] != agents[other_ind].location[1]:
                    self.location = tempLocation
                    self.valid = 'true'
        # if valid == 'false':
            # self.reward -= 1
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

    def meet(self, agents, gridSize, ind):
        if (abs(agents[0].location[0]-agents[1].location[0]) <= 1 and abs(agents[0].location[1]-agents[1].location[1]) <= 1):
            # print('Agents Met')
            # print('Agent Zero Location: ' + str(agents[0].location))
            # print('Agent One Location: ' + str(agents[1].location))
            met = 'true'
            if ind == 0:
                cmap = ListedColormap(['w', 'b', 'k', 'r', 'g'])
            else:
                cmap = ListedColormap(['w', 'b', 'k', 'g', 'r'])

        else:
            met = 'false'
            cmap = ListedColormap(['w', 'b', 'k', 'r'])
        return met, cmap

    def shareMap(self, agents, gridSize, ind):
        met, cmap = self.meet(agents, gridSize, ind)
        if met == 'true':
            # print('Sharing Map Info')
            for i in range(0, gridSize[0]):
                for j in range(0, gridSize[1]):
                    if (agents[0].map[i, j] < 2 and [i, j] != agents[1].location) or (agents[0].map[i, j] == agents[0].id and agents[0].location != agents[1].location):
                        agents[1].map[i, j] = agents[0].map[i, j]
                    elif (agents[1].map[i, j] < 2 and [i, j] != agents[0].location) or (agents[1].map[i, j] == agents[1].id and agents[1].location != agents[0].location):
                        agents[0].map[i, j] = agents[1].map[i, j]
        else:
            for i in range(2):
                for j in (1, 0):
                    if agents[i].map[agents[j].old_loc[0], agents[j].old_loc[1]] >= 3 and agents[j].old_loc[0] != agents[i].location[0] and agents[j].old_loc[1] != agents[i].location[1]:
                        # print('Overwriting Agent ' + str(i) +'s storage of agent ' + str(j) + 's location')
                        agents[i].map[agents[j].old_loc[0], agents[j].old_loc[1]] = 0

        for i in range(2):
            agents[i].old_loc = agents[i].location

        return agents, cmap

    def get_reward(self, gridSize):
        #r = sum(abs(sum(numpy.matlib.repmat(2, gridSize[0], gridSize[1])-agents[0].map)+sum(numpy.matlib.repmat(2, gridSize[0], gridSize[1])-agents[1].map)))/(gridSize[0]*gridSize[1]*4)
        # r_current = float(sum(sum(numpy.matlib.repmat(2, gridSize[0], gridSize[1])-self.map))/[gridSize[0]*gridSize[1]*2])
        r_current = float(sum(sum(numpy.matlib.repmat(2, gridSize[0], gridSize[1])-self.map))/14)
        self.reward = r_current-self.r_old-0.1
        self.r_old = r_current
        if self.valid == 'false':
            self.reward -= 0.4
        return self
        # Single Agent (return single object)

    def percent_exp(self, gridSize):
        count = 0
        for i in range(gridSize[0]):
            for j in range(gridSize[1]):
                if self.map[i, j] != 2:
                    count = count + 1

        p_exp = 100*count/(gridSize[0]*gridSize[1])
        return p_exp


class plot:
    def init(self, interactive):
        if interactive == 'False':
            plt.ioff()
        else:
            plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cmap = ListedColormap(['w', 'b', 'k', 'r'])
        return ax, cmap, fig

    def multiInit(self, agents, interactive):
        # print('Generating New Map')
        if interactive == 'False':
            plt.ioff()
        else:
            plt.ion()
        fig1 = plt.figure()
        fig2 = plt.figure()
        fig = (fig1, fig2)
        ax1 = fig1.add_subplot(1, 1, 1)
        ax2 = fig2.add_subplot(1, 1, 1)
        ax = (ax1, ax2)
        # cmap = ListedColormap(['w', 'b', 'k', 'r'])
        met, cmap = agents[0].meet(agents, [], [])

        return fig, ax, cmap

    def update(self, ax, cmap, grid, interactive):
        ax.cla()
        ax.matshow(grid, cmap=cmap)
        if interactive == 'True':
            plt.draw()
            plt.pause(0.00000001)
        return ax

    def multiUpdate(self, ax, cmap, agents, numAgents, interactive):
        for i in range(numAgents):
            ax[i].cla()
            ax[i].matshow(agents[i].map, cmap=cmap)
        if interactive == 'True':
            plt.draw()
            plt.pause(0.00000001)
        return ax

    def showGrid(self, grid) :
        fig13 = plt.figure(2)
        ax1 = fig3.add_subplot(111)
        cmap = ListedColormap(['w', 'k'])
        ax1.matshow(grid, cmap=cmap)
        plt.draw()
        plt.pause(2)
        plt.close('all')

#     def getImage(self):
#         img = plt.savefig()
#         return img
