#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 08:23:22 2021

@author: thom
"""

# Python program to demonstrate
# polytopes
  
  
# Using numpy to create matrices
import numpy as np 
import polytope as pc 
  
xmin1 = -4
xmax1 = -2
ymin1 = 0
ymax1 = 4

xmin2 = -2.5
xmax2 = 2
ymin2 = 1
ymax2 = 2.5

p1 = pc.box2poly([[xmin1, xmax1], [ymin1, ymax1]])
p2 = pc.box2poly([[xmin2, xmax2], [ymin2, ymax2]])

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

p1.plot(ax)
p2.plot(ax)

plt.autoscale() 
plt.show()

##

# Now compute difference

fig, ax = plt.subplots()

p_newA = p1.diff(p2)
p_newB = p2.diff(p1)

if len(p_newA) < len(p_newB):
    p_new = p_newA
else:
    p_new = p_newB

p_new.plot(ax)

plt.autoscale() 
plt.show()

##

p_int = p1.intersect(p2)
p_union = p1.union(p2.diff(p_int))

for p in p_union:
    fig, ax = plt.subplots()
    
    p.plot(ax)
    
    plt.xlim([-5, 3])
    plt.ylim([-1, 5])
    plt.show()