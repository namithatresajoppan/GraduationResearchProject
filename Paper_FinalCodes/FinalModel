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


#0<gamma_L<gamma_H<1
gamma_L = 0.3
gamma_H = 0.45
fixed_cost = 0.45
sigma = 1.5
beta = 0.95
delta = 0.08
theta = 0.8
g = 1
p_ga = random.random()#for global attachment
p_ld = random.random()#for link deletion
p_delta = 0.3#for local attachment


#global function that calculates the weight of the edge, args: the 2 nodes (agent class objects)
#this is the homophily edge weight formula
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
        e1 = pow(beta, time - 1)*E_t*theta*(agent.alpha*gamma*a2 + (1-delta)) 

        #(beta*E_t*theta*(alpha*gamma_H*a2 + (1-delta)))^(1/sigma)
        e2 = pow(e1, (1/sigma))
        
        #c*sigmathroot(beta*E_t*theta*(alpha*gamma_H*a2 + (1-delta)))^(1/sigma)
        con = agent.consum * e2
        #print("Calculated consumption :", con)
        
        return k_new, con, slope
    
def isocline(agent):
        if(agent.tec == 'H'):
            con_cond = agent.alpha*pow(agent.k, gamma_H) + (1-delta)*agent.k - agent.k/theta
        if(agent.tec == 'L'):
            con_cond = agent.alpha*pow(agent.k, gamma_L) + (1-delta)*agent.k - agent.k/theta   
        return con_cond   
        
        
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
        self.front = 0 #for micawber frontier
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
        #print("Generating income")
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
        #print("Income updation")
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
            
            c_cond = isocline(self)
            #print("c_cond = ", c_cond)
            
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
                #print("4th quadrant")
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
            #print("c_cond = ", c_cond)

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
                #print("4th quadrant")
                if(con > c_cond and con<self.k):
                    self.consum = con
                else:
                    con = c_cond - random.random()
                    while(con>self.k or con<0):
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
        
    def LocalAttachment_v1(self): 
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
        
        
    #1. select nodes i and j, with a probability proportional to the weight (wij) between them.
    #2. j selects a neighbour with a probability proportional to the weight between them such that there is no edge between 
    #i and k. 
    #3. a link is made between i and k with edge weight w0 (here, w0 = 1) and all edge weights are increased by wr(wr-calculated 
    #new edge weight between i and j)
    def LocalAttachment_v2(self):
        b = self.model.b
        a = self.model.a
        #print("LA done")
        links_nodes = list(self.model.G.edges(data= 'weight'))
        #print('Link nodes=', links_nodes)
        edge_weight_sum = sum(k for i, j,k in links_nodes)
        #print('Weight sum = ', edge_weight_sum)
        edge_weights = []
        for edge in links_nodes:
            edge_weights.append(edge[2])
        #print('Edge weights = ', edge_weights)
        edge_prob = []
        for edge_w in edge_weights:
            edge_prob.append(edge_w/edge_weight_sum)
        #print('Edge prob = ', edge_prob)
        arr_pos = [i for i in range(len(links_nodes))]
        pos = np.random.choice(arr_pos, p = edge_prob)
        chosen = links_nodes[pos]
        #print("Chosen nodes:", chosen[0:2])
        node1 = chosen[0]
        node2 = chosen[1]
        node1_a = next((agent for agent in self.model.agents if agent.unique_id == node1), None)
        node2_a = next((agent for agent in self.model.agents if agent.unique_id == node2), None)
        #print("Keeping node1 as:", node1)
        #print("Keeping node2 as:", node2)
        #finding neighbors of node2
        neighbors = [n for n in self.model.G.neighbors(node2)]
        #print("Neighbors of {} are:{}".format(node2,neighbors))
        if(len(neighbors)>1):
            #print("Finding 3rd node")
            links_nodes = list(self.model.G.edges(node2, data= 'weight'))
            #print('Link nodes=', links_nodes)
            edge_weight_sum = sum(k for i, j,k in links_nodes)
            #print('Weight sum = ', edge_weight_sum)
            edge_weights = []
            for edge in links_nodes:
                edge_weights.append(edge[2])
            #print('Edge weights = ', edge_weights)
            edge_prob = []
            for edge_w in edge_weights:
                edge_prob.append(edge_w/edge_weight_sum)
            #print('Edge prob = ', edge_prob)
            arr_pos = [i for i in range(len(links_nodes))]
            pos = np.random.choice(arr_pos, p = edge_prob)
            chosen = links_nodes[pos]
            #print("Chosen nodes:", chosen[0:2])
            for node in chosen[0:2]: #because the 3rd element is weight
                if(node!= node2):
                    node3 = node
                    #print('3rd node = ', node3)
                    if(self.model.G.has_edge(node1,node3)==False and random.random()>p_delta):
                        node3_a = next((agent for agent in self.model.agents if agent.unique_id == node3), None)
                        self.model.G.add_edge(node1,node3,weight = Edge_Weight(node1_a,node3_a, b, a))
                    
    
   #links are deleted randomly at every time step
    def Link_Deletion(self):
        #print('Deletion')
        if(random.random()>p_ld):
            node1 = random.choice(self.model.nodes)
            node2 = random.choice(self.model.nodes)
            count = 0
            while(self.model.G.has_edge(node1,node2)==False and count<5):
                node2 = random.choice(self.model.nodes)
                count +=1
            if(count !=5):
                self.model.G.remove_edge(node1,node2)
        #print('deletion done')
                    
    def step(self):
        #if(self.k > 0):
        self.income_updation()
        self.give_money()
        #self.LocalAttachment_v1()
        self.LocalAttachment_v2()
        self.Link_Deletion()
        self.income_generation() 
        
        
        
    class BoltzmannWealthModelNetwork(Model):
    """A model with some number of agents."""

    def __init__(self,b, a,N=500): #N- number of agents

        self.N = N
        self.b =b
        self.a = a
        self.agents = []
        self.gini = 0
        self.time = 0
        self.count_GA = 0
        self.G = nx.barabasi_albert_graph(n=N, m = 1)
        nx.set_edge_attributes(self.G, 1, 'weight') #setting all initial edges with a weight of 1
        self.nodes = np.linspace(0,N-1,N, dtype = 'int') #to keep track of the N nodes   
        
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(model_reporters = {"Gini": 'gini'},agent_reporters={"k_t":'k','income':'income',
                                           'H':'front', 'consumption':'consum','lamda':'lamda','alpha':'alpha', 'technology':'tec' })       
        for i, node in enumerate(self.G.nodes()):
            agent = MoneyAgent(i, self)
            self.schedule.add(agent)
           
        self.running = True
        self.datacollector.collect(self)
        
    def Global_Attachment(self):
        if(random.random()>p_ga):
            #print("Global Attachment no: {}".format(self.count))
            self.count_GA+=1
            node1 = random.choice(self.nodes)
            node2 = random.choice(self.nodes)
            while(self.G.has_edge(node1,node2)==True):
                node2 = random.choice(self.nodes)
                node1 = random.choice(self.nodes)
            #adding the edge node1-node2
            #first: find the class object corresponding to the node
            for agent in self.agents:
                if(agent.unique_id == node1):
                    node1_a = agent
                if(agent.unique_id == node2):
                    node2_a = agent
            #a and b are pre-determined, a is the homophily parameter and b is the characteristic distance
            self.G.add_edge(node1,node2,weight =  Edge_Weight(node1_a,node2_a, self.b, self.a))
        
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
        for i in tqdm(range(n)):
            #print("Step:", i+1)
            self.time = i+1
            self.step()
            self.Global_Attachment()
            self.gini = self.compute_gini()
            
            
N = 500
steps = 125
b = 35
a = 0.69
alpha = np.random.normal(loc = 1.08, scale = 0.074, size = N) 
capital = np.random.uniform(low = 0.1, high = 10, size = N)
model = BoltzmannWealthModelNetwork(b, a,N)
model.run_model(steps)
model_df = model.datacollector.get_model_vars_dataframe()
agent_df = model.datacollector.get_agent_vars_dataframe()
agent_df.reset_index(level=1, inplace = True)
agent_df.to_csv("V2_{}Agents_{}Steps.csv".format(N,steps))
model_df.to_csv("V2_Model_{}Agents_{}Steps.csv".format(N,steps))
