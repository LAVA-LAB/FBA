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
import itertools as it  

xmin1 = -4
xmax1 = -1
ymin1 = -4
ymax1 = 1

xmin2 = 0
xmax2 = 4
ymin2 = -3
ymax2 = 2

xmin3 = -2
xmax3 = 2
ymin3 = 0
ymax3 = 4

xmin4 = 4
xmax4 = 7
ymin4 = 0
ymax4 = 4

xmin5 = -3
xmax5 = 0
ymin5 = 3
ymax5 = 5

polys = np.array([ pc.box2poly([[xmin1, xmax1], [ymin1, ymax1]]),
                   pc.box2poly([[xmin2, xmax2], [ymin2, ymax2]]),
                   pc.box2poly([[xmin3, xmax3], [ymin3, ymax3]]),
                   pc.box2poly([[xmin4, xmax4], [ymin4, ymax4]]),
                   pc.box2poly([[xmin5, xmax5], [ymin5, ymax5]]) ])

nr_p = len(polys)


def maxDictKey(d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value"""  
     v=list(d.values())
     k=list(d.keys())
     return k[v.index(max(v))]

def mergePolys(polys):
    
    overlap_dict = {}
    zeroOverlap = True
    
    for (i1, i2) in it.combinations(range(nr_p), 2):
        
        volume = np.round( polys[i1].intersect(polys[i2]).volume, 5)
        
        overlap_dict[(i1,i2)] = volume
        
        if volume > 0:
            zeroOverlap = False
    
    if not zeroOverlap:
    
        idxs = maxDictKey(overlap_dict)
        
        print('Merge polytope indices:',idxs)
        
        region = polys[idxs[0]].union(polys[idxs[1]], check_convex=True)
        polys_new = np.array(region.list_poly)
        
        other_polys = [poly for i,poly in enumerate(polys) if i not in idxs]
        
        polys = np.concatenate((other_polys, polys_new))
    
    return polys, zeroOverlap



import matplotlib.pyplot as plt

fig, ax = plt.subplots()
    
for poly in polys:
    poly.plot(ax, alpha=0.5)

plt.xlim([-5, 8])
plt.ylim([-5, 5])
            
plt.show()

done = False
while not done:
    polys, done = mergePolys(polys)

# Plot    
fig, ax = plt.subplots()

pc.Region(polys).plot(ax, alpha=0.5)

plt.xlim([-5, 8])
plt.ylim([-5, 5])
        
plt.show()

# %%

# Determine which regions are adjacent

from scipy.sparse.csgraph import connected_components

adj_mat = np.zeros((nr_p, nr_p))

for (i1, i2) in it.combinations(range(nr_p), 2):
    
    if pc.is_adjacent(polys[i1], polys[i2], overlap=True):
        
        adj_mat[i1, i2] = adj_mat[i2, i1] = 1
        
merge_for = np.argmax(np.sum(adj_mat, axis=0))
connected_to = np.argwhere(adj_mat[:, merge_for]).flatten()
pol = [poly for i,poly in enumerate(polys) if i in connected_to]

union = polys[merge_for].union(pc.Region(pol))

# Plot    
fig, ax = plt.subplots()

union.plot(ax, alpha=0.5)

plt.xlim([-5, 8])
plt.ylim([-5, 5])
        
plt.show()

# %%
        
nr_comp, comp = connected_components(adj_mat)

####

all_polys = []

# Remove intersection for connected regions
for nr in range(nr_comp):
    
    c_polys = [poly for i,poly in enumerate(polys) if comp[i] == nr]

    if len(c_polys) > 1:

        print('Merge for component', nr)
        print('Number of components:',len(c_polys))        

        union = c_polys[0].union(pc.Region(c_polys[1:]), check_convex = True)
        
        all_polys += union.list_poly
        
    else:
        
        all_polys += c_polys

union = pc.Region(all_polys)

# Plot    
fig, ax = plt.subplots()

union.plot(ax, alpha=0.5)

plt.xlim([-5, 8])
plt.ylim([-5, 5])
        
plt.show()

# %%

r1 = pc.Region([p1])
r2 = pc.Region([p2])
r3 = pc.Region([p3])
r4 = pc.Region([p4])

def nonOverlappingRegion(polytopesIn):
    
    rNo = pc.Region([p1])
    
    for i,poly in enumerate(polytopesIn[1:]):
        
        print('Merge for polytope',i+1)
        
        polys_new = []
        
        for pNo in rNo.list_poly:
            
            p_diff = poly.diff(pNo)
            
            p_diffA = pNo.diff(poly)
            p_diffB = poly.diff(pNo)
            
            if len(p_diffA) < len(p_diffB):
                p_half  = p_diffA
                p_full  = p2
            else:
                p_half  = p_diffB
                p_full  = p1
                
            if len(p_half) > 0:
                # Is region
                polys_new += [p_full] + p_half.list_poly
            else:
                # Is polytope
                polys_new += [p_full, p_half]
                
        rNo = pc.Region(polys_new)
        
        # Plot
        fig, ax = plt.subplots()
        rNo.plot(ax, alpha=0.5)
        
        plt.xlim([-5, 8])
        plt.ylim([-5, 5])
        
        plt.title('Step '+str(i))
        plt.show()
        
    return rNo
        
rNoOverlap = nonOverlappingRegion([p1,p2,p3,p4])
    
# Plot
fig, ax = plt.subplots()
rNoOverlap.plot(ax, alpha=0.5)

plt.xlim([-5, 8])
plt.ylim([-5, 5])

plt.show()

'''
def nonOverlappingRegion(regions_in):
    
    regions = regions_in
    
    while len(regions) > 1:
        
        print('length:',len(regions))
        
        allDisconnected = True
        
        iterator = it.combinations(regions, 2)
        
        regions = [0 for i in iterator]
        
        for i,(r1, r2) in enumerate(iterator):
            
            poly_list = []
            
            for p1 in r1.list_poly:
                for p2 in r2.list_poly:
                    
                    if p1.intersect(p2).volume > 0:
                        allDisconnected = False
                    
                    p_newA = p1.diff(p2)
                    p_newB = p2.diff(p1)
                    
                    if len(p_newA) < len(p_newB):
                        p_cut  = p_newA
                        p_full = p2
                    else:
                        p_cut  = p_newB
                        p_full = p1
                        
                    if len(p_cut) > 0:
                        # Is region
                        poly_list += [p_full] + p_cut.list_poly
                    else:
                        # Is polytope
                        poly_list += [p_full, p_cut]
            
            regions[i] = pc.Region(poly_list)            
            
            
            # # Plot
            # fig, ax = plt.subplots()
            # regions[i].plot(ax, alpha=0.5)
            
            # plt.xlim([-5, 8])
            # plt.ylim([-5, 5])
            
            # plt.show()
            
            
            # If all regions are still disconnected, break
            if allDisconnected is True:
                break
        
    return regions
        
nonOverlappingRegion([r1,r2,r3,r4])
'''

'''
def polyDiffToRegion(region,poly):
    
    return poly.diff(region)

def uniquePolys(polyList):
    
    for i, poly in enumerate(polyList):
        
        otherPolys = [p for j,p in enumerate(polyList) if i != j]
        
        polySep = polyDiffToRegion(pc.Region(otherPolys), poly)
        
        # Plot
        fig, ax = plt.subplots()
        polySep.plot(ax, alpha=0.5)
        
        plt.xlim([-5, 8])
        plt.ylim([-5, 5])
        
        plt.show()
    
    
uniquePolys([p1,p2,p3,p4])
    

def addPoly2Region(region, poly):
    
    p_newA = region.diff(poly)
    
    print(p_newA)
    
    region_new = pc.Region( list(p_newA) + [p2] )
        
    return region_new

def polyUnion(polyList):
    
    region = pc.Region([polyList[0]])
    
    for poly in polyList[1:]:
        
        # Add next polytope to region
        region = addPoly2Region(region, poly)

        # Print final region
        fig, ax = plt.subplots()
        region.plot(ax, alpha=0.5)
        
        plt.xlim([-5, 8])
        plt.ylim([-1, 5])
        
        plt.show()
    

##
'''

'''
p_newA = p1.diff(p2)
p_newB = p2.diff(p1)

if len(p_newA) < len(p_newB):
    p_new = p_newA + p2
else:
    p_new = p_newB + p1

fig, ax = plt.subplots()
p_new.plot(ax, alpha=0.5)

plt.autoscale() 
plt.show()

p_newA = p_new.diff(p3)
p_newB = p3.diff(p_new)

if len(p_newA) < len(p_newB):
    p_new = p_newA + p3
else:
    p_new = p_newB + p_new

fig, ax = plt.subplots()
p_new.plot(ax, alpha=0.5)

plt.autoscale() 
plt.show()
'''


'''
R = pc.Region([p2,p3,p1,p4])
R_env = pc.envelope(R)

fig, ax = plt.subplots()
R.plot(ax, alpha=0.5)

plt.autoscale() 
plt.show()

R_unique = R_env.diff(R)

##

# p_union = p4.union(pc.Region([p1,p2,p3]), check_convex=True)
# p_union = pc.envelope(p_union)

print('Volume:',R_unique.volume)
print('Nr polytopes:',len(R_unique))

fig, ax = plt.subplots()
R_unique.plot(ax, alpha=0.5)

plt.autoscale() 
plt.show()
'''

'''
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
'''