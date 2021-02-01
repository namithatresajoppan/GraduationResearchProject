#!/usr/bin/env python
# coding: utf-8

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
import numpy as np


#agent

class MoneyAgent(Agent):
    """ An agent with fixed initial wealth."""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        #wealth is the total wealth, P-most probable money, m is the wealth
        #at each time step
        self.m = self.model.T      

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position   = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        lamda = self.model.lamda
        epsilon = random.random()
        if len(cellmates) > 1 :
            other = self.random.choice(cellmates)
            if(other.unique_id != self.unique_id):
                #print(self)
                #print(other)
                #print(self.m, other.m, self.m + other.m)
                xi = self.m
                xj = other.m
                xi_new= lamda*xi + epsilon*(1-lamda)*(xi+xj)
                xj_new = lamda*xj + (1-epsilon)*(1-lamda)*(xi+xj)
                other.m = xj_new
                self.m = xi_new
                #print(self.m, other.m, self.m + other.m)
                #print(xi_new, xj_new, xi_new+xj_new)
                #print("break")

    def step(self):
        if self.m > 0:
            self.give_money()
        self.move()
