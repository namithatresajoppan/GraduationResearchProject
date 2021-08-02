#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import math
import statistics
import cmath


# In[ ]:


#0<gamma_L<gamma_H<1
gamma_L = 0.3
gamma_H = 0.45
fixed_cost = 0.45
sigma = 1.5
beta = 0.95
delta = 0.08
theta = 0.8
g = 1


# In[ ]:


#global function that calculates the weight of the edge, args: the 2 nodes (agent class objects)
def Edge_Weight(node1,node2, b, a):
        try:
             weight = 1+math.exp(a*((node1.k-node2.k)-b))
        except OverflowError:
             weight = float('inf')
        return 1/weight  
    
def calculating_k_c(agent, gamma, E_t, time):
        a1 = pow(agent.k,gamma) 
       
        #k_t+1 = theta*(alpha*k_t^gamma - C_t + (1-delta)*k_t)
        k_new = theta*(agent.alpha*a1-agent.consum + (1-delta)*agent.k)
        #print("New k = ", k_new)

        slope = gamma*agent.alpha*pow(agent.k, gamma -1) + 1 - delta - 1/theta
        #print("Slope = ", slope)
    
        #k_t+1^(gamma-1)
        a2 = pow(k_new,(gamma-1)) 

        #beta*E*theta*(alpha*gamma*k_t+1^(gamma-1)+(1-delta))
        e1 = pow(beta,time-1)*E_t*theta*(agent.alpha*gamma*a2 + (1-delta))
        #e1 = beta*E_t*theta*(agent.alpha*gamma*a2 + (1-delta))

        #(beta*E_t*theta*(alpha*gamma_H*a2 + (1-delta)))^(1/sigma)
        e2 = pow(e1, (1/sigma))
        
        #c*(beta*E_t*theta*(alpha*gamma_H*a2 + (1-delta)))^(1/sigma)
        con = agent.consum * e2
        #print("Calculated consumption :", con)
        
        return k_new, con, slope
    
def isocline(agent):
        if(agent.tec == 'H'):
            con_cond = agent.alpha*pow(agent.k, gamma_H) + (1-delta)*agent.k - agent.k/theta
        if(agent.tec == 'L'):
            con_cond = agent.alpha*pow(agent.k, gamma_L) + (1-delta)*agent.k - agent.k/theta   
        return con_cond   


# In[ ]:


class MoneyAgent(Agent):
    
    def __init__(self, unique_id, model):
        
        super().__init__(unique_id, model)
        self.k = (capital[unique_id]) #initial stock of wealth
        self.lamda = round(random.uniform(0.1,1),1) #saving propensity
        while (self.lamda == 1):
            self.lamda = round(random.uniform(0.1,1),1)    
        self.alpha = alpha[unique_id]#human capital 
        self.tec = 'NA'
        self.income = 0 #initialising income
        self.income_generation() #finding income corresponding to the human capital,
                                 #needed here to set the initial consumption
       
        con_cond = isocline(self)
        #self.consum = isocline(self)
        #if(self.consum < 0):
            #self.consum = 0.1
        if(self.tec == 'H'):
            self.slope = gamma_H*self.alpha*pow(self.k, gamma_H -1) + 1 - delta - 1/theta
        else:
            self.slope = gamma_L*self.alpha*pow(self.k, gamma_L -1) + 1 - delta - 1/theta
    
        if(self.slope > 0): #small k_t
            #print("1st quadrant")
            if(con_cond > 0 and con_cond<self.k):
                self.consum = con_cond
            else:
                con = con_cond - random.random()
                while(con>self.k or con < 0):
                    con = con_cond - random.random()
                self.consum = con
        else:
            #print("4th quadrant")
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
            
    
    # 
    #function that updates the capital and consumption for the next time step    
    def income_updation(self):
        #finding the initial gini coefficient and total number of poor at time step 1
        if(self.model.time == 1):
            count = 0
            for agent in self.model.agents:
                if(agent.income<yp):
                    count+=1
            self.model.poor_no = count
            agent_wealths = [agent.k for agent in self.model.agents]
            x = sorted(agent_wealths)
            B = sum(xi * (self.model.N - i) for i, xi in enumerate(x)) / (self.model.N * sum(x))
            self.model.gini =  1 + (1 / self.model.N) - 2 * B
        #finding expected value of income at each time step
        e_t = [a.income for a in self.model.agents] #is this k or f(alpha,k)?
        E_t = statistics.mean(e_t)
        k = self.k
        alpha = self.alpha
        consum = self.consum
        #print("Agent:{}  Tec: {}".format(self.unique_id, self.tec))
        #print("Old k = {}, alpha = {} " .format(k, alpha))
        #print("mean : ", E_t)
       
        if(self.tec == 'H'):
            k_new, con, slope = calculating_k_c(self, gamma_H, E_t, self.model.time)
            self.k = k_new
            self.slope = slope
            
            c_cond = isocline(self)
            #print("c_cond = ", c_cond)
            
            if(self.slope > 0):
                #print("1st quadrant")
                if(con <=c_cond and con<self.k):
                    self.consum = con
                else:
                    con = c_cond - random.random()
                    while(con>self.k or con < 0) : # or con>c_cond
                        #print("stuck")
                        con = c_cond - random.random()
                    self.consum = con
            else:
                #print("4th quadrant")
                if(con >= c_cond and con<self.k):
                    self.consum = con
                else:
                    con = c_cond - random.random()
                    while(con>self.k  or con < 0): #or con < c_cond
                        #print("stuck")
                        con = c_cond - random.random()
                    self.consum = con

        if(self.tec == 'L'):
            k_new, con, slope = calculating_k_c(self, gamma_L, E_t, self.model.time)
            self.k = k_new
            self.slope = slope
            
            c_cond = isocline(self)
            #print("c_cond = ", c_cond)

            if(self.slope > 0):
                #print("1st quadrant")
                if(con <=c_cond and con<self.k):
                    self.consum = con
                else:
                    con = c_cond - random.random()
                    while(con>=self.k or con < 0) : # or con>c_cond
                        #print("Loop")
                        con = c_cond - random.random()
                    self.consum = con
            else:
                #print("4th quadrant")
                if(con >= c_cond and con<self.k):
                    self.consum = con
                else:
                    con = c_cond - random.random()
                    while(con>=self.k  or con < 0): #or con < c_cond
                        #print("Loop")
                        con = c_cond - random.random()
                    self.consum = con
        #print("Old C:", consum)   
        #print("New Consum :", self.consum)
    
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


# In[ ]:


class BoltzmannWealthModelNetwork(Model):
    """A model with some number of agents."""

    def __init__(self,b, a,N): #N- number of agents

        self.N = N
        self.b =b
        self.a = a
        self.agents = []
        self.gini = 0
        self.time = 0
        self.Budget = 0
        self.G = nx.barabasi_albert_graph(n=N, m = 1)
        nx.set_edge_attributes(self.G, 1, 'weight') #setting all initial edges with a weight of 1
        self.nodes = np.linspace(0,N-1,N, dtype = 'int') #to keep track of the N nodes   
        
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(model_reporters = {"Gini": 'gini', 'Agents below yp':'poor_no'},
        agent_reporters={"slope": "slope","k_t":'k','income':'income','consumption':'consum','lamda':'lamda',
                         'alpha':'alpha','technology':'tec' })       
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
        
    def compute_gini(self):
        agent_wealths = [agent.k for agent in self.schedule.agents]
        x = sorted(agent_wealths)
        B = sum(xi * (self.N - i) for i, xi in enumerate(x)) / (self.N * sum(x))
        return 1 + (1 / self.N) - 2 * B
    
    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self, n):
        for i in (range(n)):
            self.time = i+1
            self.step()
            self.count_L = 0
            self.count_H = 0
            self.Global_Attachment()
            self.gini = self.compute_gini()
            print("Step: ",self.time)
            data = self.datacollector.get_agent_vars_dataframe()
            data = data.reset_index()
            data = data.loc[data.Step == i]
            if(self.time == 1):
                self.Budget = 0.09*sum(data.income)#0.025*sum(data.income) #isn't this the budget? (at time step=1)
        
            #calculating the total poverty shortfall
            yi_data = data.loc[data.income<yp]
            income = yi_data.income.reset_index(drop = True)
            #print(income)  
            S = sum(yp - (yi_data.income.reset_index(drop=True).to_numpy()))
            #print(S)
            poor_agents = yi_data.AgentID.reset_index(drop = True).to_numpy()
            self.poor_no = len(poor_agents)
            #print("Total Poor: ", len(poor_agents))
            self.count_L = 0
            self.count_H = 0
            for poor in poor_agents:
                for agent in self.agents: #accessing the class object
                        if(poor == agent.unique_id):
                            if(agent.tec == 'L'):
                                self.count_L +=1
                            else:
                                self.count_H +=1
            #print("Low: ", self.count_L)
            #print("High :", self.count_H)
            if(self.Budget > S):
                #print("out")
                for poor in poor_agents: #accessing the poor node
                    for agent in self.agents: #accessing the class object
                        if(poor == agent.unique_id):
                            additional = yp - agent.income
                            agent.income += additional
            else:
                #print("in")
                for poor in poor_agents: #accessing the poor node
                    for agent in self.agents: #accessing the class object
                        if(poor == agent.unique_id):
                            additional = yp*(self.Budget/S)
                            agent.income += additional
                
                
           


# In[ ]:


N = 100
steps = 500
b = 35
a = 0.69
alpha = np.random.normal(loc = 1.08, scale = 0.074, size = N) 
capital = np.random.uniform(low = 0.1, high = 10, size = N)
yp = 1.62
model = BoltzmannWealthModelNetwork(b, a,N)
model.run_model(steps)
model_df = model.datacollector.get_model_vars_dataframe()
agent_df = model.datacollector.get_agent_vars_dataframe()
agent_df.reset_index(level=1, inplace = True)
agent_df = agent_df.reset_index()
agent_df.to_csv("Experiment1_Agent.csv")
model_df.loc[1:].to_csv("Experiment1_Model.csv")


# In[ ]:




