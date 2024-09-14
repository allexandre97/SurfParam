#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:38:59 2024

@author: alexandre
"""

import numpy as np

from sklearn.neighbors     import NearestNeighbors
from sklearn.decomposition import PCA

class Maths:
    
    def __init__(self,
                 POINTS     : np.array,
                 NNeighbors : int  = 100) -> None:
        
        self.POINTS = POINTS
        self.InitNeighbors(NNeighbors)
        
    def InitNeighbors(self, 
                      K : int = 100) -> None:
        
        self.NN = NearestNeighbors(n_neighbors = K).fit(self.POINTS)
        
    def KNN(self,
            POINTS : np.array,
            K      : int = 100) -> tuple:
        
        dist, ids = self.NN.kneighbors(POINTS, n_neighbors = K)
        
        return dist, ids
        

            
    def EstimateNormals(self, 
                        K : int = 100) -> np.array:
        '''
        Estimates the normal vector to each point via Principal Component Analysis

        Parameters
        ----------
        K : int, optional
            Number of neighbors to each point. The default is 50.

        Returns
        -------
        normals : NumPy array
            Array containing the normal vector to each point.

        '''
        
        # Init normals array
        normals = np.zeros_like(self.POINTS)
        # Get K nearest neighbors to each point
        dist, ids = self.KNN(self.POINTS, K)
        
        # Iterate over all points and estimate normals through PCA
        for i, point in enumerate(self.POINTS):
            
            neighs = self.POINTS[ids[i]]
            
            pca = PCA(n_components = 3).fit(neighs)
            normals[i] = pca.components_[-1]
            
        return normals
    
    def CreatePeriodicImages(self, 
                             box_sizes     : tuple,
                             periodic_dims : str ='XY') -> np.array:
        '''
        Takes the points in 3d space and returns another array of points
        with their periodic images along the specified dimensions

        Parameters
        ----------
        box_sizes : tuple
            Size of the bounding box containing the points.
        periodic_dims : str, optional
            Dimensions along which the periodic images are to be contructed.
            The default is 'XY'.

        Returns
        -------
        periodic_positions : NumPy.array
            Array containing the original point plus their periodic images.

        '''
        
        N = self.POINTS.shape[0]
        box_sizes = np.asarray(box_sizes)
        
        # Create base offsets
        base_offsets = np.array([[0, 0, 0]])
        
        # Generate offsets based on periodic dimensions
        for dim, axis in zip('XYZ', range(3)):
            if dim in periodic_dims.upper():
                new_offsets = []
                for offset in base_offsets:
                    for direction in [-1, 1]:
                        new_offset = offset.copy()
                        new_offset[axis] = direction
                        new_offsets.append(new_offset)
                base_offsets = np.vstack((base_offsets, new_offsets))
        
        # Number of images (including original)
        num_images = len(base_offsets)
        
        # Tile the original positions
        tiled_positions = np.tile(self.POINTS, (num_images, 1))
        
        # Create offsets for all particles
        all_offsets = np.repeat(base_offsets, N, axis=0)
        
        # Scale offsets by box sizes
        scaled_offsets = all_offsets * box_sizes
        
        # Add offsets to the tiled positions
        periodic_positions = tiled_positions + scaled_offsets
        
        return np.array([p for p in periodic_positions if ((p[0] >= -box_sizes[0]/2 and p[0] <= box_sizes[0]*(1+0.5)) and
                                                           (p[1] >= -box_sizes[1]/2 and p[1] <= box_sizes[1]*(1+0.5)) and
                                                           (p[2] >= -box_sizes[2]/2 and p[2] <= box_sizes[2]*(1+0.5)))])