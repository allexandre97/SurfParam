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
                 POINTS : np.array) -> None:
        
        self.POINTS = POINTS
        self.InitNeighbors()
        
    def InitNeighbors(self, 
                         K : int = 100) -> None:
        
        self.NN = NearestNeighbors(n_neighbors = K).fit(self.POINTS)
        
    def KNN(self,
            POINTS : np.array,
            K      : int = 100) -> tuple:
        
        dist, ids = self.NN.kneighbors(POINTS, n_neighbors = K)
        
        return dist, ids
        
    
    def OrientNormal(self, 
                     normal : np.array,
                     point  : np.array,
                     cog    : np.array,
                     geom   : str = 'SPHERE'):
        
        if geom.upper() == 'SPHERE':
            
            return np.sign(np.dot(normal, point - cog)) * normal    
        
        else:
            
            print('Specified Geometry not supported!')
            
    def EstimateNormals(self, 
                        K    : int = 50,
                        geom : str = 'SPHERE'):
        
        cog = np.mean(self.POINTS,
                      axis = 0)
        
        normals = np.zeros_like(self.POINTS)
        dist, ids = self.KNN(self.POINTS, K)
        
        for i, point in enumerate(self.POINTS):
            
            neighs = self.POINTS[ids[i]]
            
            pca = PCA(n_components = 3).fit(neighs)
            normals[i] = pca.components_[-1]#self.OrientNormal(pca.components_[-1], point, cog)
        
        return normals