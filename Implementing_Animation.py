#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import seaborn as sns
import networkx as nx
import os, sys
from itertools import product
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import math


# In[2]:


sns.set_style("darkgrid")
sns.set(rc={'figure.figsize':(20,8)})


#global function that calculates the weight of the edge, args: the 2 nodes (agent class objects)
def Edge_Weight(node1,node2, b, alpha):
        try:
             weight = 1+math.exp(alpha*((node1.m-node2.m)-b))
        except OverflowError:
             weight = float('inf')
        return 1/weight
                


# In[3]:


class MoneyAgent(Agent):
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.m = self.model.T
        self.lamda = round(random.uniform(0.1,1),1)
        while (self.lamda == 1):
            self.lamda = round(random.uniform(0.1,1),1)
        self.model.agents.append(self)
        
    def neighbors(self):
        neighbors_nodes = list(nx.all_neighbors(self.model.G,self.unique_id))
        neighbors = []
        for node in neighbors_nodes:
            for agent in self.model.agents:
                if(agent.unique_id == node):
                    neighbors.append(agent)
        return neighbors
            
    def give_money(self):
        b = self.model.b
        alpha = self.model.alpha
        neighbors = self.neighbors()
        epsilon = random.random()
        if len(neighbors) > 1 :
            other = self.random.choice(neighbors)
            while(other.unique_id == self.unique_id):
                other = self.random.choice(neighbors)  
            w = self.model.G[self.unique_id][other.unique_id]['weight'] 
            #print(w)
            if(w >= random.random()): 
                xi = self.m
                xj = other.m
                delta_m = (1-self.lamda)*(xi - epsilon*(xi + xj))
                xi_new = xi - delta_m
                xj_new = xj + delta_m
                other.m = xj_new
                self.m = xi_new
                self.model.trade+=1
                for neighbor in neighbors:
                    self.model.G[self.unique_id][neighbor.unique_id]['weight'] = Edge_Weight(self,neighbor,b, alpha)
                other_neighbors = other.neighbors()
                for neighbor in other_neighbors:
                    if(neighbor.unique_id != other.unique_id):
                        #print("Other: {}  Neighbor: {}".format(other.unique_id,neighbor.unique_id))
                        self.model.G[other.unique_id][neighbor.unique_id]['weight'] = Edge_Weight(other,neighbor,b, alpha)
                
    def Local_Attachment(self):
        b =  self.model.b
        alpha = self.model.alpha
        node1 = random.choice(self.model.nodes)
        node2 = random.choice(self.model.nodes)
        while(self.model.G.has_edge(node1,node2)==True):
            node2 = random.choice(self.model.nodes)
            node1 = random.choice(self.model.nodes)
        for agent in self.model.agents:
            if(agent.unique_id == node1):
                node1_a = agent
            if(agent.unique_id == node2):
                node2_a = agent
        self.model.G.add_edge(node1,node2,weight = Edge_Weight(node1_a,node2_a, b, alpha))
        
    
    def Link_Deletion(self):
        node1 = random.choice(self.model.nodes)
        node2 = random.choice(self.model.nodes)
        while(self.model.G.has_edge(node1,node2)==False):
            node1 = random.choice(self.model.nodes)
            node2 = random.choice(self.model.nodes)
        self.model.G.remove_edge(node1,node2)
        
                 
    def step(self):
        if self.m > 0:
            self.give_money()
        self.Local_Attachment()
        self.Link_Deletion()


# In[18]:


class BoltzmannWealthModelNetwork(Model):
    """A model with some number of agents."""

    def __init__(self,b, alpha, fig, ax, steps, T=100,N=500): #N- number of agents

        self.N = N
        self.T = T
        self.trade = 0
        self.count = 0
        self.b =b
        self.alpha = alpha
        self.agents = []
        self.fig = fig
        self.ax = ax
        self.steps = steps
        self.G = nx.barabasi_albert_graph(n=N, m = 1)
        nx.set_edge_attributes(self.G, 1, 'weight') #setting all initial edges with a weight of 1
        #print(nx.get_edge_attributes(self.G,'weight'))
        #nx.draw(self.G)
        self.nodes = np.linspace(0,N-1,N, dtype = 'int') #to keep track of the N nodes   
        self.layout = nx.spring_layout(self.G)
        
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(model_reporters = {'trade': 'trade', 'beta': 'b', 'alpha': 'alpha'},agent_reporters={"mi":'m','lamda':'lamda' })
        
        for i, node in enumerate(self.G.nodes()):
            a = MoneyAgent(i, self)
            self.schedule.add(a)
           
        self.running = True
        self.datacollector.collect(self)
        
    def Global_Attachment(self):
        #print("Global Attachment no: {}".format(self.count))
        node1 = random.choice(self.nodes)
        node2 = random.choice(self.nodes)
        while(self.G.has_edge(node1,node2)==True):
            node2 = random.choice(self.nodes)
            node1 = random.choice(self.nodes)
        #adding the edge node1-node2
        for agent in self.agents:
            if(agent.unique_id == node1):
                node1_a = agent
            if(agent.unique_id == node2):
                node2_a = agent
        self.G.add_edge(node1,node2,weight = Edge_Weight(node1_a,node2_a, self.b, self.alpha))

        
    def update(self, i):
        self.ax.clear()
        #print("This is getting called {}".format(i))
        self.ax.axhline(y = 100, label = 'initial money', ls = '--')
        self.ax.set_title("Variation of Money by different agents (Frame no: {})".format(i))
        for n, agent in zip(self.nodes, self.agents):
            self.ax.scatter(n, agent.m, color = 'red')  
            #plot_data.append(ax.scatter(n, agent.m, color = 'red'))
            #print(n, agent.m)
        self.ax.legend()
        self.run_model()
       
    
    def animate(self):
        self.anim = FuncAnimation(self.fig, self.update,frames = self.steps, interval=200, repeat = False, save_count = self.steps)
        plt.show()
        f = r"c://Users/Namitha Tresa Joppan/Documents/UvA/Graduation Project/Codes/animation.mp4" 
        writervideo = animation.FFMpegWriter(fps=1) 
        self.anim.save(f, writer=writervideo)
        self.anim.save('animation.mp4', fps=1, extra_args=['-vcodec', 'libx264'])

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self):
        #for i in tqdm(range(n)):
        #self.ani = FuncAnimation(self.fig, self.update, interval=200)
        #self.animate()
        self.step()
        self.trade = 0
        self.Global_Attachment()
        #here, find mean and variance and save it as model parameters


# In[19]:


N = 100
T = 100
steps = 100
b = 35.98
alpha = 0.6933
fig, ax = plt.subplots(figsize = (10,6))
model = BoltzmannWealthModelNetwork(b, alpha,fig, ax,steps, T,N)
model.animate()
#animator = FuncAnimation(fig, model.network,frames = 100, interval = 200)
#plt.show()
#model.run_model(steps)
model_df = model.datacollector.get_model_vars_dataframe()
agent_df = model.datacollector.get_agent_vars_dataframe()
agent_df.reset_index(level=1, inplace = True)
agent_df['mt'] = agent_df.mi/T


# In[ ]:




