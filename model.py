#!/usr/bin/env python
# coding: utf-8
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agent import MoneyAgent

#saving propensity = lamda



#model

#saving propensity = lamda
class BoltzmannWealthModel(Model):

    def __init__(self,T,N,lamda, width=10, height=10):
        self.num_agents = N
        self.T = T
        self.grid = MultiGrid(height, width, True)
        self.lamda = lamda
        self.count = 0
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(agent_reporters={ 'mi':'m'})
        # Create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self, n):
        for i in tqdm(range(1,n)):
            self.count+=1
            #print("step:{}".format(i))
            self.step()

