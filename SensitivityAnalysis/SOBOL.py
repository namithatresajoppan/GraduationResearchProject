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
import scipy.stats as stats 
import math
import SALib
from SALib.sample import saltelli
from SALib.analyze import sobol
from mesa.batchrunner import BatchRunner
from mesa.batchrunner import FixedBatchRunner
from IPython.display import clear_output
from itertools import combinations
import statistics
#plt.style.use('seaborn-pastel')
#from tqdm.autonotebook import tqdm


# In[77]:


#0<gamma_L<gamma_H<1
gamma_L = 0.3
gamma_H = 0.45
fixed_cost = 0.45
sigma = 1.5
g = 1
N = 100
alpha = np.random.normal(loc = 1.08, scale = 0.074, size = N) 
capital = np.random.uniform(low = 0.1, high = 10, size = N)


#global function that calculates the weight of the edge, args: the 2 nodes (agent class objects)
def Edge_Weight(node1,node2, b, a):
        try:
             weight = 1+math.exp(a*((node1.k-node2.k)-b))
        except OverflowError:
             weight = float('inf')
        return 1/weight  
    
def calculating_k_c(agent, gamma, E_t, time):
        a1 = pow(agent.k,gamma) 
        k_new = agent.model.theta*(agent.alpha*a1-agent.consum + (1-agent.model.delta)*agent.k)
        slope = gamma*agent.alpha*pow(agent.k, gamma -1) + 1 - agent.model.delta - 1/agent.model.theta
        a2 = pow(k_new,(gamma-1)) 
        e1 = pow(agent.model.beta, time - 1)*E_t*agent.model.theta*(agent.alpha*gamma*a2 + (1-agent.model.delta)) 
        e2 = pow(e1, (1/sigma))
        con = agent.consum * e2
        return k_new, con, slope
    
def isocline(agent):
        if(agent.tec == 'H'):
            con_cond = agent.alpha*pow(agent.k, gamma_H) + (1-agent.model.delta)*agent.k - agent.k/agent.model.theta
        if(agent.tec == 'L'):
            con_cond = agent.alpha*pow(agent.k, gamma_L) + (1-agent.model.delta)*agent.k - agent.k/agent.model.theta   
        return con_cond   


# In[78]:


class MoneyAgent(Agent):
    
    def __init__(self, unique_id, model):
        
        super().__init__(unique_id, model)
        #self.ID = unique_id
        self.k = (capital[unique_id]) #initial stock of wealth
        self.lamda = round(random.uniform(0.1,1),1) #saving propensity
        while (self.lamda == 1):
            self.lamda = round(random.uniform(0.1,1),1)    
        self.alpha = alpha[unique_id]#human capital 
        self.tec = 'NA'
        self.ID = unique_id
        self.a = self.model.a
        self.b = self.model.b
        self.beta = self.model.beta
        self.theta = self.model.theta
        self.delta = self.model.delta
        self.income = 0 #initialising income
        self.income_generation() #finding income corresponding to the human capital,
                                 #needed here to set the initial consumption
        con_cond = isocline(self)
        #self.consum = isocline(self)
        #if(self.consum < 0):
            #self.consum = 0.1
        if(self.tec == 'H'):
            self.slope = gamma_H*self.alpha*pow(self.k, gamma_H -1) + 1 - self.model.delta - 1/self.model.theta
        else:
            self.slope = gamma_L*self.alpha*pow(self.k, gamma_L -1) + 1 - self.model.delta - 1/self.model.theta
    
        if(self.slope > 0): #small k_t
            if(con_cond > 0 and con_cond<self.k):
                self.consum = con_cond
            else:
                con = con_cond - random.random()
                while(con>self.k or con < 0):
                    con = con_cond - random.random()
                self.consum = con
        else:
            if(con_cond > 0 and con_cond <self.k):
                self.consum = con_cond
            else:
                con = con_cond + random.random()
                while(con>self.k or con<0):
                    con = con_cond + random.random()
                self.consum = con
        self.model.agents.append(self)
            
    #function that decides income based on the type of technology
    def income_generation(self): 
        b1 = pow(self.k,gamma_H)
        H = self.alpha*b1 - fixed_cost
        
        b2 = pow(self.k,gamma_L)
        L = self.alpha*b2
        
        self.front = H
        if(H>=L): #
            self.income = H
            self.tec = 'H'
        else:
            self.income = L
            self.tec = 'L'
            
    
    #function that updates the capital and consumption for the next time step    
    def income_updation(self):
        
        #finding expected value of income at each time step
        e_t = [a.income for a in self.model.agents] #is this k or f(alpha,k)?
        E_t = statistics.mean(e_t)
        k = self.k
        alpha = self.alpha
        consum = self.consum
        if(self.tec == 'H'):
            
            k_new, con, slope = calculating_k_c(self, gamma_H, E_t, self.model.time)
            self.k = k_new
            
            c_cond = isocline(self)

            if(slope > 0):
                #print("1st quadrant")
                if(con <=c_cond and con<self.k):
                    self.consum = con
                else:
                    con = c_cond - random.random()
                    while(con>self.k or con < 0):
                        con = c_cond - random.random()
                    self.consum = con
            else:
                if(con > c_cond and con<self.k):
                    self.consum = con
                else:
                    con = c_cond - random.random()
                    while(con>self.k or con < 0):
                        con = c_cond - random.random()
                    self.consum = con

        if(self.tec == 'L'):
            
            k_new, con, slope = calculating_k_c(self, gamma_L, E_t, self.model.time)
            self.k = k_new
            
            c_cond = isocline(self)
            
            if(slope > 0):
                #print("1st quadrant")
                if(con <=c_cond and con<self.k):
                    self.consum = con
                else:
                    con = c_cond - random.random()
                    while(con>self.k or con < 0):
                        con = c_cond - random.random()
                    self.consum = con
            else:
                if(con > c_cond and con<self.k):
                    self.consum = con
                else:
                    con = c_cond - random.random()
                    while(con>self.k or con<0):
                        con = c_cond - random.random()
                    self.consum = con  

    #finding neighbor nodes for the purpose of making an edge/connection
    def neighbors(self):
        neighbors_nodes = list(nx.all_neighbors(self.model.G,self.unique_id))
        neighbors = []
        for node in neighbors_nodes:
            for agent in self.model.agents:
                if(agent.unique_id == node):
                    neighbors.append(agent)
        return neighbors
    
     #function used to trade/communicate     
    def give_money(self): 
        b = self.model.b
        a = self.model.a
        neighbors = self.neighbors()
        epsilon = random.random()
        if len(neighbors) > 1 :
            other = self.random.choice(neighbors)
            while(other.unique_id == self.unique_id):
                other = self.random.choice(neighbors)  
            w = self.model.G[self.unique_id][other.unique_id]['weight'] 
            if(w >= random.random()): 
                xi = self.income
                xj = other.income
                delta_income = (1-self.lamda)*(xi - epsilon*(xi + xj))
                xi_new = xi - delta_income
                xj_new = xj + delta_income
                other.income = xj_new
                self.income = xi_new
                for neighbor in neighbors:
                    self.model.G[self.unique_id][neighbor.unique_id]['weight'] = Edge_Weight(self,neighbor,b, a)
                other_neighbors = other.neighbors()
                for neighbor in other_neighbors:
                    if(neighbor.unique_id != other.unique_id):
                        self.model.G[other.unique_id][neighbor.unique_id]['weight'] = Edge_Weight(other,neighbor,b, a)
        
   
    #link addition happening at every time step
    def Local_Attachment(self): 
        b = self.model.b
        a = self.model.a
        node1 = random.choice(self.model.nodes)
        node2 = random.choice(self.model.nodes)
        count = 0 #to avoid an infinite loop when all agents have already made links with each other
        while(self.model.G.has_edge(node1,node2)==True and count <5):
            node2 = random.choice(self.model.nodes)
            node1 = random.choice(self.model.nodes)
            count +=1
        for agent in self.model.agents:
            if(agent.unique_id == node1):
                node1_a = agent
            if(agent.unique_id == node2):
                node2_a = agent
        self.model.G.add_edge(node1,node2,weight = Edge_Weight(node1_a,node2_a, b, a))
    
    
   #links are deleted randomly at every time step
    def Link_Deletion(self):
        node1 = random.choice(self.model.nodes)
        node2 = random.choice(self.model.nodes)
        while(self.model.G.has_edge(node1,node2)==False):
            node1 = random.choice(self.model.nodes)
            node2 = random.choice(self.model.nodes)
        self.model.G.remove_edge(node1,node2)
                    
    def step(self):
        #if(self.k > 0):
        self.income_updation()
        self.give_money()
        self.Local_Attachment()
        self.Link_Deletion()
        self.income_generation() 


# In[79]:


class BoltzmannWealthModelNetwork(Model):
    """A model with some number of agents."""

    def __init__(self,b =35.98, a = 0.6933, beta = 0.95, delta = 0.08, theta = 0.8,N=100): #N- number of agents
        self.N = N
        self.b =b
        self.a = a
        self.agents = []
        self.fit_alpha = 0
        self.fit_loc = 0
        self.fit_beta = 0
        self.t = 0
        self.beta = 0.95
        self.delta = 0.08
        self.theta = 0.8
        self.time = 1 #for sensitivity analysis
        self.G = nx.barabasi_albert_graph(n=N, m = 1)
        nx.set_edge_attributes(self.G, 1, 'weight') #setting all initial edges with a weight of 1
        self.nodes = np.linspace(0,N-1,N, dtype = 'int') #to keep track of the N nodes   
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(agent_reporters={'k':'k','lamda':'lamda',
            'abilitity':'alpha', 'technology':'tec','b': 'b', 'a': 'a','beta': 'beta',
            'theta': 'theta', 'delta': 'delta'})
                                                                   
        
        for i, node in enumerate(self.G.nodes()):
            agent = MoneyAgent(i, self)
            self.schedule.add(agent)
           
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
        self.G.add_edge(node1,node2,weight = Edge_Weight(node1_a,node2_a, self.b, self.a)) 
        

    def step(self):
        #print(self.time)
        self.schedule.step()
        # collect data
        self.Global_Attachment() #for sensitivity analysis
    
    def run_model(self, n):
        for i in tqdm(range(n)):
            self.time = i+1
            self.step()
            
           


# In[83]:


problem = {
    'num_vars': 5,
    'names': ['b', 'a', 'beta', 'delta','theta'],
    'bounds': [[1, 10], [1, 10], [0.01, 0.99],[0.01, 0.99],[0.01, 0.99]]
}
#shape parameter is alpha, rate parameter is beta
fixed_parameters = {'N' : 100}
#agent_reporters = {'kt': lambda m:m.k,'alpha':lambda m:m.alpha,'lamda': lambda m:m.lamda,
                   #'tec':lambda m:m.tec,'a': lambda m:m.a, 'b': lambda m:m.b,
                   #'theta':lambda m:m.theta,'beta': lambda m:m.beta, 'delta':lambda m:m.delta}

agent_reporters = {'kt': 'k','alpha':'alpha','lamda': 'lamda',
                   'tec':'tec','a': 'a', 'b': 'b',
                   'theta':'theta','beta': 'beta', 'delta':'delta'}
#
# Set the repetitions, the amount of steps, and the amount of distinct values per variable
replicates = 10
max_steps = 100#
distinct_samples = 512

# generating samples
param_values = saltelli.sample(problem, distinct_samples)
variable_parameters={name:[] for name in problem['names']}
#print(variable_parameters)
#print(len(param_values))
#print(param_values[450])


# In[ ]:


batch = BatchRunner(BoltzmannWealthModelNetwork,variable_parameters = variable_parameters,
                    fixed_parameters = fixed_parameters,max_steps=max_steps,iterations=2,
                    agent_reporters = agent_reporters,
                    display_progress = True)
count = 0
for i in range(replicates):
    for vals in param_values: 
        vals = list(vals)
        #print(vals)

        # Transform to dict with parameter names and their values
        variable_parameters = {}
        for name, val in zip(problem['names'], vals):
            variable_parameters[name] = val
        #print(variable_parameters)
        batch.run_iteration(variable_parameters, tuple(vals), count)
        count += 1

        clear_output(wait=True)
        print(f'{count / (len(param_values) * (replicates)) * 100:.2f}% done')
data = batch.get_collector_agents()
df = pd.DataFrame()
for key, value in data.items():
    df = df.append(value)
df.reset_index(level = ['Step', 'AgentID'],inplace = True)
df.to_csv("SOBOL.csv")


# In[ ]:




