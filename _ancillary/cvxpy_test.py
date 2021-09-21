#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import packages.
import cvxpy as cp
import numpy as np

def steadystateCovariance_sdp(covariances):

    # Dimension of the covariance matrix
    n = len(covariances[0])    

    # Solve for worst-case covariance matrix
    X_w = cp.Variable((n,n), symmetric=True)
    X_b = cp.Variable((n,n), symmetric=True)
    
    # The operator >> denotes matrix inequality.
    for cov_mat in covariances:
        
        # Increase eigenvalues artificially if some are zero
        if not is_invertible(cov_mat):
        
            eigvals, eigvecs = np.linalg.eig(cov_mat)
            eigsvals = np.maximum(1e-9, eigsvals)
            cov_mat = eigvecs @ np.diag(eigvals) @ np.linalg.inv(eigvecs)
        
        constraints_w = [X_w >> cov_mat]
        constraints_b = [X_b >> cov_mat]
    prob_w = cp.Problem(cp.Minimize(cp.trace(X_w)),
                      constraints_w)
    prob_w.solve()
    
    prob_b = cp.Problem(cp.Minimize(cp.trace(X_b)),
                      constraints_b)
    prob_b.solve()
    
    worst_cov = np.copy(X_w.value)
    best_cov  = np.copy(X_b.value)
    
    return worst_cov, best_cov