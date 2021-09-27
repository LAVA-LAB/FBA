#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 ______________________________________
|                                      |
|  SCENARIO-BASED ABSTRACTION PROGRAM  |
|______________________________________|

Implementation of the method proposed in the paper:
 "Sampling-Based Robust Control of Autonomous Systems with Non-Gaussian Noise"

Originally coded by:        <anonymized>
Contact e-mail address:     <anonymized>
______________________________________________________________________________
"""

import numpy as np
import csv                      # Import to create/load CSV files
import sys                      # Allows to terminate the code at some point
import os                       # Import OS to allow creationg of folders

import itertools
from .commons import table, is_invertible, floor_decimal, confidence_ellipse, \
    Chi2probability, tocDiff, printWarning, maxDictKey
from scipy.spatial import Delaunay
from scipy.linalg import sqrtm
import cvxpy as cp
from scipy.stats import mvn

import matplotlib.pyplot as plt

def in_hull(p, hull):
    '''
    Test if points in `p` are in `hull`.

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    '''
    
    if not isinstance(hull,Delaunay):
        print(' -- Creating hull...')
        hull = Delaunay(hull, qhull_options='QJ')

    boolArray = hull.find_simplex(p) >= 0

    return boolArray
    
def point_in_hull(queries, hull):
    equations = hull.get_simplex_facet_array()[2].T
    
    return np.all(queries @ equations[:-1] < - equations[-1], axis=1)

def computeRegionCenters(points, partition, precision):
    '''
    Function to compute to which region (center) a list of points belong

    Parameters
    ----------
    points : 2D Numpy array
        Array, with every row being a point to determine the center point for.
    partition : dict
        Dictionary of the partition.

    Returns
    -------
    2D Numpy array
        Array, with every row being the center coordinate of that row of the
        input array.

    '''
    
    # Check if 'points' is a vector or matrix
    if len(np.shape(points)) == 1:
        points = np.reshape(points, (1,len(points)))
    
    # Retreive partition parameters
    region_width = np.array(partition['width'])
    region_nrPerDim = partition['nrPerDim']
    dim = len(region_width)
    
    # Boolean list per dimension if it has a region with the origin as center
    originCentered = [True if nr % 2 != 0 else False for nr in region_nrPerDim]

    # Initialize centers array
    centers = np.zeros(np.shape(points)) 
    
    # Shift the points to account for a non-zero origin
    originShift = np.array(partition['origin'] )
    pointsShift = points - originShift
    
    for q in range(dim):
        # Compute the center coordinates of every shifted point
        if originCentered[q]:
            
            centers[:,q] = ((pointsShift[:,q]+0.5*region_width[q]) // region_width[q]) * region_width[q]
        
        else:
            
            centers[:,q] = (pointsShift[:,q] // region_width[q]) * region_width[q] + 0.5*region_width[q]
    
    # Add the origin again to obtain the absolute center coordinates
    return np.round(centers + originShift, precision)

def loadScenarioTable(tableFile):
    '''
    Load tabulated bounds on the transition probabilities (computed using
    the scenario approach).

    Parameters
    ----------
    tableFile : str
        File from which to load the table.

    Returns
    -------
    memory : dict
        Dictionary containing all loaded probability bounds / intervals.

    '''
    
    memory = dict()
    
    if not os.path.isfile(tableFile):
        sys.exit('ERROR: the following table file does not exist:'+
                 str(tableFile))
    
    with open(tableFile, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for i,row in enumerate(reader):
            
            strSplit = row[0].split(',')
            
            key = tuple( [int(float(i)) for i in strSplit[0:2]] + 
                         [float(strSplit[2])] )
            
            value = [float(i) for i in strSplit[-2:]]
            memory[key] = value
                
    return memory

def kalmanFilter(model, cov0):    
    '''
    For a given model in `model` and prior belief covariance `cov0`, perform
    The Kalman filter steps related to the covariance (mean is not needed).

    Parameters
    ----------
    model : dict
        Main dictionary of the LTI system model.
    cov0 : ndarray
        n by n matrix of the covariance of the prior belief.

    Returns
    -------
    cov_pred : ndarray
        Predicted covariance (after the prediction step).
    K_gain : ndarray
        Optimal Kalman gain.
    cov_tilde : ndarray
        Covariance of the mean of the posterior belief.
    cov : ndarray
        Covariance of the posterior belief.
    cov_tilde_measure : float
        Measure on the covariance of the mean of the posterior belief.

    '''
    
    # Predicted covariance of the belief
    cov_pred = model['A'] @ cov0 @ model['A'].T + model['noise']['w_cov']
    
    # If matrix is invertible
    if is_invertible(model['C'] @ cov_pred @ model['C'].T + model['noise']['v_cov']):
        # Determine optimal Kalman gain
        
        K_gain   = cov_pred @ model['C'].T @ \
            np.linalg.inv(model['C'] @ cov_pred @ model['C'].T + model['noise']['v_cov'])
        
    else:
        # If not invertible, set Kalman gain to zero
        K_gain   = np.zeros((model['n'], model['r']))
        
    # Covariance of the expected mean of the future belief
    cov_tilde = K_gain @ \
    (model['C'] @ cov_pred @ model['C'].T + model['noise']['v_cov']) @ K_gain.T
        
    # Covariance of the future belief
    cov = (np.eye(model['n']) - K_gain @ model['C']) @ cov_pred
    
    # Calculate measure on the covariance matrix of the mean of the future belief
    cov_tilde_measure, _ = covarianceEllipseSize( cov_tilde )
    
    max_error_bound = np.max( minimumEpsilon( cov, beta=0.01, stepSize=0.01, singleParam = True ) )
    
    return {'cov_pred': cov_pred,
            'K_gain': K_gain,
            'cov_tilde': cov_tilde,
            'cov': cov,
            'cov_tilde_measure': cov_tilde_measure,
            'error_bound': max_error_bound}
    
def minimumEpsilon(cov, beta=0.01, stepSize=0.01, singleParam=True):
    
    n = np.shape(cov)[0]
    
    if not singleParam:
        ### Method with separated dimensions
        
        # Compute the maximum error between the mean of the belief, and the actual
        # state
        cumprob = 0.5 * np.ones(n)
        epsilon = np.zeros(n)
        
        origin = np.zeros(n)
        
        for dim in range(n):
            while cumprob[dim] > beta:
                epsilon[dim] += stepSize
                
                
                lower = [-100 for d in range(n)]
                upper = [100 if d != dim else -epsilon[d] for d in range(n)]
                
                cumprob[dim] = mvn.mvnun(lower=lower, upper=upper, means=origin, covar=cov)[0]
        
        
        max_error_bound = epsilon
        
    else:
        ### Hyper-rectangular method with single parameters
        
        cumprob = 0
        epsilon = 0
        inf_dims = []
        inf_lim = 10
        
        count = 0
        count_max = 10000
        while cumprob < 1-beta:
            if count > count_max:
                print(' -- Computing error bound aborted after',count_max,'iterations')
                break
            
            epsilon += stepSize
            
            lower = [-epsilon if d not in inf_dims else -inf_lim for d in range(n)]
            upper = [ epsilon if d not in inf_dims else  inf_lim for d in range(n)]
            
            cumprob = mvn.mvnun(lower=lower,
                                upper=upper,
                                means=np.zeros(n), covar=cov)[0]
            
        max_error_bound = np.array([epsilon if d not in inf_dims else inf_lim 
                                    for d in range(n)])
        
    return max_error_bound

def steadystateCovariance(covariances, verbose=False):
    '''
    Compute the best/worst case covariance matrix that contains a list of 
    other covariance matrices

    Parameters
    ----------
    covariances : list of arrays
        List of numpy arrays describing the covariances.

    Returns
    -------
    smallest_cov : array
        Numpy array describing the covariances that contains all given inputs

    '''
    
    print('Compute best/worst-case covariance (iterative method)')
    
    if verbose:
        fig, ax = plt.subplots(figsize=(6, 6))
        mean = np.array([0,0])
    
        confidence_ellipse(mean, covariances[0], ax, n_std=1, edgecolor='firebrick')
    
    # Initialize the output covariance matrix
    if not is_invertible(covariances[0]):
        eigvalues, eigvectors = np.linalg.eig(covariances[0])
        eigvalues_nonzero = np.maximum(1e-9, eigvalues)
        cov = eigvectors @ np.diag(eigvalues_nonzero) @ np.linalg.inv(eigvectors)
        worst_cov = cov
        best_cov = cov
    else:
        worst_cov = covariances[0]
        best_cov  = covariances[0]
    
    # Compute smallest ellipse that contains all ellipses starting from some k
    for cov in covariances[1:]:
        
        if verbose:
            # Plot current ellipse
            confidence_ellipse(mean, cov, ax, n_std=1, edgecolor='firebrick')
        
        if not is_invertible(cov):
            eigvalues, eigvectors = np.linalg.eig(cov)
            eigvalues_nonzero = np.maximum(1e-9, eigvalues)
            cov = eigvectors @ np.diag(eigvalues_nonzero) @ np.linalg.inv(eigvectors)
        
        # Compute the rotation matrix that transforms one ellipse to a circle    
        R = sqrtm(np.linalg.inv(cov))
        
        # Apply this rotation matrix to the other ellipse
        ellipse_worst = R @ worst_cov @ R.T
        ellipse_best = R @ best_cov @ R.T
        
        # Compute the eigen values and vectors of the transformed ellipse
        ellipse_worst_evals, ellipse__worst_evecs = np.linalg.eig(ellipse_worst)
        ellipse_best_evals, ellipse_best_evecs = np.linalg.eig(ellipse_best)
        
        # Determine which eigen values (i.e. radii) are larger: ellipse vs circle
        biggest_eigvals = np.maximum(1, ellipse_worst_evals)
        smallest_eigvals = np.minimum(1, ellipse_best_evals)
        
        # Compute the smallest/biggest ellipse that contains both the circle and ellipse
        worst_cov_transf = (ellipse__worst_evecs @ np.diag(biggest_eigvals) @ np.linalg.inv(ellipse__worst_evecs))
        best_cov_transf  = (ellipse_best_evecs @ np.diag(smallest_eigvals) @ np.linalg.inv(ellipse_best_evecs))
        
        # Transform the smallest/biggest ellipse back using the inverse rotation matrix
        worst_cov = np.linalg.inv(R) @ worst_cov_transf @ np.linalg.inv(R).T
        best_cov  = np.linalg.inv(R) @ best_cov_transf @ np.linalg.inv(R).T
        
    if verbose:
        confidence_ellipse(mean, np.real(worst_cov), ax, n_std=1, 
                           label= 'Worst-case covariance', edgecolor='blue')
        confidence_ellipse(mean, np.real(best_cov), ax, n_std=1, 
                           label= 'Best-case covariance', edgecolor='red')
           
        ax.autoscale()
        ax.set_aspect('equal')
        ax.set_title('Iterative eigenvalue approach')
        plt.show()
    
    return {'worst': np.real(worst_cov), 'best': np.real(best_cov)}

def steadystateCovariance_sdp(covariances, verbose=True):

    print('Compute best/worst-case covariance (SDP method)')    

    # Dimension of the covariance matrix
    n = len(covariances[0])    

    # Solve for worst-case covariance matrix
    X_w = cp.Variable((n,n), PSD=True)
    X_b = cp.Variable((n,n), PSD=True)
    
    constraints_w = [X_w >> 0]
    constraints_b = [X_b >> 0]
    
    if verbose:
        fig, ax = plt.subplots(figsize=(6, 6))
        
        mean = np.array([0,0])
    
    # The operator >> denotes matrix inequality.
    for cov in covariances:
        
        if verbose:
            # Plot current ellipse
            confidence_ellipse(mean, cov, ax, n_std=1, edgecolor='firebrick')
        
        # Increase eigenvalues artificially if some are zero
        if not is_invertible(cov):
        
            eigvals, eigvecs = np.linalg.eig(cov)
            
            eigvals = np.maximum(1e-9, eigvals)
            cov = eigvecs @ np.diag(eigvals) @ np.linalg.inv(eigvecs)
        
        constraints_w += [X_w >> cov]
        constraints_b += [cov >> X_b]
        
    prob_w = cp.Problem(cp.Minimize(cp.trace(X_w)), constraints_w)
    prob_w.solve()
    
    prob_b = cp.Problem(cp.Maximize(cp.trace(X_b)), constraints_b)
    prob_b.solve()
    
    worst_cov = np.copy(X_w.value)
    best_cov  = np.copy(X_b.value)
        
    if verbose:
        confidence_ellipse(mean, worst_cov, ax, n_std=1, 
                           label= 'Worst-case covariance', edgecolor='blue')
        confidence_ellipse(mean, best_cov, ax, n_std=1, 
                           label= 'Best-case covariance', edgecolor='red')
           
        ax.autoscale()
        ax.set_aspect('equal')
        ax.set_title('Semi-definite programming approach')
        plt.show()
        
    # Check if best-case covariance was computed incorrectly. If multiple 
    # covariance matrices are not full-rank (i.e. they are not expressed as an
    # ellipse, but as a line), this may happen.
    if len(np.shape(best_cov)) == 0:
        printWarning('Warning: NoneType best-case covariance. Set arbitrary small covariance.')
        best_cov = 1e-6*np.eye(n)
    
    return {'worst': worst_cov, 'best': best_cov}

def covarianceEllipseSize(cov):
    '''
    Computes the maximum radius of the ellipse that is associated with the
    covariance matrix `cov`.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix.

    Returns
    -------
    ellipse_size : float
        Maximum radius of the ellipse associated with the covariance matrix.

    '''
    
    # Determine the largest size parameter of the rotated ellipse associated
    # with the given covariance matrix
    
    # Determine eigenvectors and values
    eigenval = np.linalg.eigvals(cov)
    
    # Retreive the largest/smallest value
    max_eigenval = eigenval[np.argmax(eigenval)]
    min_eigenval = eigenval[np.argmin(eigenval)]
    
    # Determine width parameter of ellipse
    max_ellipse_size = np.sqrt( max(0, max_eigenval) )
    min_ellipse_size = np.sqrt( max(0, min_eigenval) )
    
    return max_ellipse_size, min_ellipse_size

def definePartitions(dim, nrPerDim, regionWidth, origin, onlyCenter=False):
    '''
    Define the partitions object `partitions` based on given settings.

    Parameters
    ----------
    dim : int
        Dimension of the state (of the LTI system).
    nrPerDim : list
        List of integers, where each value is the number of regions in that 
        dimension.
    regionWidth : list
        Width of the regions in every dimension.
    origin : list
        Coordinates of the origin of the continuous state space.
    onlyCenter : Boolean, default=False
        If True, only the center of the regions is computed. 
        If False, the full partition (e.g. vertices) is computed        

    Returns
    -------
    partitions : dict
        Dictionary containing the info regarding partisions.

    '''
    
    regionWidth = np.array(regionWidth)
    origin      = np.array(origin)
    
    elemVector   = dict()
    for i in range(dim):
        
        elemVector[i] = np.linspace(-(nrPerDim[i]-1)/2, 
                                     (nrPerDim[i]-1)/2,
                                     int(nrPerDim[i]))
        
    widthArrays = [[x*regionWidth[i] for x in elemVector[i]] 
                                              for i in range(dim)]
    
    if onlyCenter:        
        partitions = np.array(list(itertools.product(*widthArrays))) + origin
        
    else:
        partitions = dict()
        
        for i,elements in enumerate(itertools.product(*widthArrays)):
            partitions[i] = dict()
            
            center = np.array(elements) + origin
            
            partitions[i]['center'] = center
            partitions[i]['low'] = center - regionWidth/2
            partitions[i]['upp'] = center + regionWidth/2
    
    return partitions

def makeModelFullyActuated(model, manualDimension='auto', observer=False):
    '''
    Given a model in `model`, render it fully actuated.

    Parameters
    ----------
    model : dict
        Main dictionary of the LTI system model.
    manualDimension : int or str, optional
        Desired dimension of the state of the model The default is 'auto'.
    observer : Boolean, default=False
        If True, it is assumed that the system is not directly observable, so
        an observer is created.

    Returns
    -------
    model : dict
        Main dictionary of the LTI system model, which is now fully actuated.

    '''
    
    if manualDimension == 'auto':
        # Determine dimension for actuation transformation
        dim    = int( np.size(model['A'],1) / np.size(model['B'],1) )
    else:
        # Group a manual number of time steps
        dim    = int( manualDimension )
    
    # Determine fully actuated system matrices and parameters
    A_hat  = np.linalg.matrix_power(model['A'], (dim))
    B_hat  = np.concatenate([ np.linalg.matrix_power(model['A'], (dim-i)) \
                                      @ model['B'] for i in range(1,dim+1) ], 1)
    
    Q_hat  = sum([ np.linalg.matrix_power(model['A'], (dim-i)) @ model['Q']
                       for i in range(1,dim+1) ])
    
    w_sigma_hat  = sum([ np.array( np.linalg.matrix_power(model['A'], (dim-i) ) \
                                      @ model['noise']['w_cov'] @ \
                                      np.linalg.matrix_power(model['A'].T, (dim-i) ) \
                                    ) for i in range(1,dim+1) ])
    
    # Overwrite original system matrices
    model['A']               = A_hat
    model['B']               = B_hat
    model['Q']               = Q_hat
    model['Q_flat']          = Q_hat.flatten()
    
    model['noise']['w_cov']  = w_sigma_hat
    
    # Redefine sampling time of model
    model['tau']             *= (dim+1)
    
    return model

def defineDistances(partition, target):
    '''
    Define the mutual distance (per dimension) between any two target points.

    Parameters
    ----------
    partition : dict
        Dictionary containing the info regarding partisions.
    target : int
        Index of the base region to which we want to measure.

    Returns
    -------
    distance_list : list
        List of distances (per dimension) to the `base_region`.

    '''
    
    # Define center of the base region as point A
    
    # For the center of every other region, compute difference to point A
    distance_list = [ target - region['center']
                     for region in partition.values()]
        
    return distance_list 

def cubic2skew(cubic, abstr):
    origin = abstr['origin']
    return (np.array(cubic) - origin) @ abstr['basis_vectors'] + origin
    
def skew2cubic(skew, abstr):
    origin = abstr['origin']
    return (np.array(skew) - origin) @ abstr['basis_vectors_inv'] + origin

def computeRegionOverlap(limitsA, limitsB):
    '''
    Compute the intersection (overlap) between two given regions

    Parameters
    ----------
    limitsA : Array
        Specification of the first region.
    limitsA : Array
        Specification of the second region.

    Returns
    -------
    limits : Array
        Specification of the intersections of the two input regions.

    '''
    
    low = np.maximum(limitsA[:,0], limitsB[:,0])
    upp = np.minimum(limitsA[:,1], limitsB[:,1])
    
    if all(low < upp):
        
        return np.vstack((low, upp)).T
    
    else:
        
        return None
    
def getNeighbors(point, partition, depth=1):
        
        nr      = partition['nrPerDim']
        width   = partition['width']
        dim     = len(nr)
        
        prange = np.array([np.arange(-depth, depth+1)*width[i] for i in range(dim)])
        
        return point + np.array(list(itertools.product(*prange)))
    
def mergePolys(polys, verbose=False):
    
    overlap_dict = {}
    zeroOverlap = True
    
    for (i1, i2) in itertools.combinations(range(len(polys)), 2):
        
        volume = np.round( polys[i1].intersect(polys[i2]).volume, 5)
        
        overlap_dict[(i1,i2)] = volume
        
        if volume > 0:
            zeroOverlap = False
    
    if not zeroOverlap:
    
        idxs = maxDictKey(overlap_dict)
        
        if verbose:
            print(' ---- Merge polytope',idxs[0],'with',idxs[1])
        
        region = polys[idxs[0]].union(polys[idxs[1]], check_convex=True)
        polys_new = np.array(region.list_poly)
        
        other_polys = [poly for i,poly in enumerate(polys) if i not in idxs]
        
        polys = np.concatenate((other_polys, polys_new))
    
    return polys, zeroOverlap