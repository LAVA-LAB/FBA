#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse, Rectangle

from ..commons import cm2inch

def confidence_ellipse(mean, cov, ax, n_std=1.0, facecolor='none', **kwargs):
    """
    Create a plot of the 2-dimensional covariance ellipse

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    
    mean = np.array(mean)
    cov = np.array(cov)
    
    if not all(cov.shape[i] == mean.size for i in range(mean.size)) and \
            mean.size == 2:
        raise ValueError("Size of mean and/or covariance not consistent")

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    
    if np.round(pearson, decimals=6) == -1:
        ell_radius_x = 0
    else:
        ell_radius_x = np.sqrt(1 + pearson)
    if np.round(pearson, decimals=6) == 1:
        ell_radius_y = 0
    else:
        ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plotTrajectory(setup, fID, k0, k_end, x, y, actions, deltas, waitings,
                   mu_pred, mu, cov_tilde, cov, abstr):
    
    # fig = plt.figure(figsize=cm2inch(12, 7))
    # ax = fig.add_subplot(111)
    fig, ax = plt.subplots(figsize=(6, 4))
    
    plt.xlabel('$x_1$', labelpad=0)
    plt.ylabel('$x_2$', labelpad=-10)
    
    goalRegion = abstr['P'][abstr['goal'][0]]
    goal_lower = goalRegion['low']
    
    origin_x = setup['stateSpaceOrigin'][0]
    origin_y = setup['stateSpaceOrigin'][1]
    min_x = origin_x - setup['stateSpaceDomainMax']
    max_x = origin_x + setup['stateSpaceDomainMax']
    min_y = origin_y - setup['stateSpaceDomainMax']
    max_y = origin_y + setup['stateSpaceDomainMax']
    
    width = setup['stateSpaceDomainMax'] / (setup['PartitionsFromCenter']+0.5)
    
    # Draw grid
    major_ticks_x = np.linspace(min_x, max_x+width, setup['PartitionsFromCenter']+2)
    major_ticks_y = np.linspace(min_y, max_y+width, setup['PartitionsFromCenter']+2)
    
    minor_ticks_x = np.linspace(min_x, max_x, setup['PartitionsFromCenter']*2+2)
    minor_ticks_y = np.linspace(min_y, max_y, setup['PartitionsFromCenter']*2+2)
    
    ax.set_xticks(major_ticks_x)
    ax.set_yticks(major_ticks_y)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(minor_ticks_y, minor=True)
    
    plt.grid(which='major', color='#CCCCCC', linestyle='--')
    plt.grid(which='minor', color='#CCCCCC', linestyle='--')
    
    # Goal x-y limits
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    # Draw goal state
    goalState = Rectangle(goal_lower[0:2], width=width, height=width, color="green", alpha=0.3)
    ax.add_patch(goalState)
    
    # Draw covariance ellipse
    confidence_ellipse(mu[k0][0:2], cov[k0][0:2,0:2], ax, facecolor='blue', edgecolor='blue', alpha=0.1)
    
    k = 0
    
    while k < k_end:
        
        # Retreive action and delta value
        a = actions[k]
        d = deltas[k]
        g = waitings[k]
        
        # Draw arrow to next predicted mean
        dx = mu_pred[k+d][0] - mu[k][0]
        dy = mu_pred[k+d][1] - mu[k][1]
        plt.arrow(mu[k][0], mu[k][1], dx=dx, dy=dy, color='blue', 
                  length_includes_head=True, head_width=0.2, head_length=0.2)   
        # Action label
        halfx = mu[k][0] + dx/2
        halfy = mu[k][1] + dy/2
        
        plt.text(halfx, halfy, r"$a^{\delta="+str(d)+"}_{"+str(a)+"}$", ha="left", va="bottom")
        
        # Draw covariance ellipse
        confidence_ellipse(mu_pred[k+d][0:2], cov_tilde[k+d+g][0:2,0:2], ax, facecolor='red', edgecolor='red', alpha=0.2)
        plt.text(mu_pred[k+d][0], mu_pred[k+d][1], r"$\tilde{\Sigma}_{"+str(k+d+g)+"}$", ha="center", va="bottom")
        
        # Draw covariance ellipse
        confidence_ellipse(mu[k+d+g][0:2], cov[k+d+g][0:2,0:2], ax, facecolor='blue', edgecolor='blue', alpha=0.1)
        
        # cov_merged = np.array(cov_tilde[k+d][0:2,0:2]) + np.array(cov[k+d][0:2,0:2])
        # confidence_ellipse(pred_mean[k+d][0:2], cov_merged, ax, edgecolor='silver')
    
        # Plot measurement    
        plt.scatter(y[k+d+g], mu[k+d+g][1], c='k', s=30, marker='|')
        plt.text(y[k+d+g], mu[k+d+g][1], r"$y_{"+str(k+d+g)+"}$", ha="left", va="bottom")
        
        # Plot true state
        plt.scatter(x[k+d+g, 0], x[k+d+g, 1], c='k', s=12)
        plt.text(x[k+d+g, 0], x[k+d+g, 1], r"$x_{"+str(k+d+g)+"}$", ha="left", va="bottom")
    
        # Increment time step
        k += d + g
    
    # Set tight layout
    fig.tight_layout()
    
    # # Save figure
    # filename = setup['folder']['outputFcase']+'MC_trajectory_'+str(fID)
    # plt.savefig(filename+'.pdf', bbox_inches='tight')
    # plt.savefig(filename+'.png', bbox_inches='tight')
    
    # # Show figure
    # plt.close()