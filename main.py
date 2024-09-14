#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 12:12:53 2024

@author: alexandre
"""


#%%
import numpy      as np
import MDAnalysis as mda

import matplotlib.pyplot as plt

from mayavi import mlab
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
from MDAnalysis.analysis.leaflet import LeafletFinder as LF

from reconstruction.reconstructer import Surface

from pyellipsoid import analysis

def compute_inertia_ellipsoid(points):
    # Calculate the center of mass
    center_of_mass = np.mean(points, axis=0)

    # Calculate the inertia tensor
    shifted_points = points - center_of_mass
    inertia_tensor = np.zeros((3, 3))
    for p in shifted_points:
        inertia_tensor += np.outer(p, p)

    # Diagonalize the inertia tensor
    eigenvalues, eigenvectors = np.linalg.eig(inertia_tensor)

    # Calculate the radii of the ellipsoid
    radii = np.sqrt(len(points) / eigenvalues)

    return center_of_mass, eigenvectors, radii
    

def gaussian_kernel(distance, bandwidth):
    return np.exp(-0.5 * (distance / bandwidth) ** 2)

def kernel_gradient_estimation(points    : np.array,
                               values    : np.array,
                               bandwidth : float = 0.1,
                               k         : int   = 50) -> np.array:
    '''
    Computes the gradient of a scalar using a smoothing kernel.

    Parameters
    ----------
    points : np.array
        3D points defining positions.
    values : np.array
        Scalar value at each point position.
    bandwidth : float, optional
        Kernel bandwidth. The default is 0.1.
    k : int, optional
        Number of neighbors for each point. The default is 50.

    Returns
    -------
    gradients : np.array
        Output gradient for the scalars at each point.

    '''

    gradients = np.zeros_like(points)
    nn = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nn.kneighbors(points)

    for i in range(len(points)):
        neighbors = indices[i]
        neighbor_points = points[neighbors]
        neighbor_values = values[neighbors]

        # Calculate differences
        diff_values = neighbor_values - values[i]
        diff_points = neighbor_points - points[i]
        

        # Weigh differences using the kernel function
        weights = gaussian_kernel(distances[i], bandwidth)
        weighted_diffs = diff_points.T * diff_values * weights

        # Sum up the weighted differences and normalize
        gradient = np.sum(weighted_diffs, axis=1) / np.sum(weights)

        gradients[i] = gradient

    return gradients

def AreaPerFace(Faces  : np.array,
                Points : np.array):
    
    Area = []
    
    for face in Faces:
        
        verts = Points[face]
        
        v1 = verts[0] - verts[1]
        v2 = verts[0] - verts[2]
        
        cross = np.cross(v1, v2)
        norm  = np.linalg.norm(cross)
        
        area = 0.5*norm
        
        Area.append(area)
        
    return np.array(Area)

def VertexArea(Faces    : np.array,
               FaceIds  : np.array,
               Points   : np.array,
               FaceArea : np.array,
               p        : int):
    
    Suma = 0
    
    for face_index in FaceIds:
        
        area = FaceArea[face_index]
        ids  = Faces[face_index]
        
        k = np.where(ids == p)[0][0]
        
        k1 = (k+1) % 3
        k2 = (k+2) % 3
        
        e0 = Points[ids[2]] - Points[ids[1]]
        e1 = Points[ids[0]] - Points[ids[2]] # Compute the side vectors of the face
        e2 = Points[ids[1]] - Points[ids[0]]

        l0 = np.sqrt(e0[0]**2+e0[1]**2+e0[2]**2)
        l1 = np.sqrt(e1[0]**2+e1[1]**2+e1[2]**2) # Compute the length of the sides
        l2 = np.sqrt(e2[0]**2+e2[1]**2+e2[2]**2)

        ls = [l0,l1,l2]

        mu0 = ls[0]*(ls[1] + ls[2] - ls[0])
        mu1 = ls[1]*(ls[2] + ls[0] - ls[1]) # Compute the barycentric coordinates of circumcenter
        mu2 = ls[2]*(ls[0] + ls[1] - ls[2])

        mus  = np.array([mu0, mu1, mu2]) # Normalize mu
        mus /= (mus[0]+mus[1]+mus[2])

        Suma += area*((mus[k1] + mus[k2])/2) # Add to total area

    return Suma

def AreaPerVertex(Points   : np.array,
                  Faces    : np.array,
                  FaceArea : np.array):
    
    APV = []
    
    for point_index in range(Points.shape[0]):
        
        faces_ids = np.where(Faces == point_index)[0]
        vertex_area = VertexArea(Faces, faces_ids, Points, FaceAreas, point_index)
        
        APV.append(vertex_area)
    
    return np.array(APV)

def Cart2Sph(X, Y, Z):
    
    rho   = np.sqrt(X*X + Y*Y + Z*Z)
    theta = np.arccos(Z/rho)
    phi   = np.arctan2(Y, X)
    
    return rho, theta, phi

def Sph2Cart(rho, theta,phi):
    
    X = rho * np.sin(theta) * np.cos(phi)
    Y = rho * np.sin(theta) * np.sin(phi)
    Z = rho * np.cos(theta)
    
    return X, Y, Z

#%%
if __name__ == '__main__':
    
    k = 150
    
    U = mda.Universe('check.gro')
    PO4 = U.select_atoms('name PO4')
    
    leafs = LF(U, PO4)
    
    points = leafs.groups(0).positions
    
    N = len(leafs.groups(0))
    
    #%%
    
    SURFACE = Surface(points, NNeighbors = k, D = 6, Geometry = 'plane')
    
    #%%
    
    verts, simps = SURFACE.MeshToArray()
    
    fig = mlab.figure(size = (800, 800))

    
    mlab.points3d(SURFACE.POINTS[:N,0],
                  SURFACE.POINTS[:N,1],
                  SURFACE.POINTS[:N,2],
                  scale_factor = 10,
                  color = (0.25, 0.05, 0.85))

    mlab.triangular_mesh(verts[:,0],
                         verts[:,1],
                         verts[:,2],
                         simps,
                         figure = fig,
                         color  = (1,1,1))
    
    #%%
    
    UV, tri = SURFACE.TriangulatePlane()
    
    #%%
    fig, ax = plt.subplots(figsize = (7,7))
    
    ax.triplot(UV[:,0], UV[:,1], tri)

    #%%
    FaceAreas = AreaPerFace(tri, SURFACE.POINTS)
    
    APL = AreaPerVertex(SURFACE.POINTS, tri, FaceAreas)
    
    
    #%%
    
    fig = mlab.figure(size=(800, 800))
    
    # Plotting the points
    mesh = mlab.triangular_mesh(SURFACE.POINTS[:,0],
                                SURFACE.POINTS[:,1],
                                SURFACE.POINTS[:,2],
                                tri,
                                colormap = 'inferno',
                                # vmin = -1*maxK, vmax = maxK,
                                scalars  = APL,
                                opacity = 1,
                                figure = fig)

    points= mlab.points3d(SURFACE.POINTS[:,0],
                          SURFACE.POINTS[:,1],
                          SURFACE.POINTS[:,2],
                          color = (0,0,0),
                          scale_factor = 3,
                          figure = fig)