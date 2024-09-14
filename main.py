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
        
        e0 = points[ids[2]] - points[ids[1]]
        e1 = points[ids[0]] - points[ids[2]] # Compute the side vectors of the face
        e2 = points[ids[1]] - points[ids[0]]

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
    
    rho   = np.sqrt(X*x + Y*Y + Z*Z)
    theta = np.arccos(Z/rho)
    phi   = np.arctan2(Y, X)
    
    return rho, theta, phi

def Sph2Cart(rho, theta,phi):
    
    X = rho * np.sin(theta) * np.cos(phi)
    Y = rho * np.sin(theta) * np.sin(phi)
    Z = rho * np.cos(theta)
    
    return X, Y, Z

def create_periodic_images(positions, box_sizes, periodic_dims='XY'):
    """
    Create periodic images of particles along specified dimensions.
    
    Parameters:
    positions (numpy.ndarray): Array of shape (N, 3) containing particle positions.
    box_sizes (tuple or numpy.ndarray): Lengths of the box in X, Y, and Z directions.
    periodic_dims (str): String containing 'X', 'Y', and/or 'Z' to specify periodic dimensions.
    
    Returns:
    numpy.ndarray: Array containing original positions and their periodic images.
    """
    N = positions.shape[0]
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
    tiled_positions = np.tile(positions, (num_images, 1))
    
    # Create offsets for all particles
    all_offsets = np.repeat(base_offsets, N, axis=0)
    
    # Scale offsets by box sizes
    scaled_offsets = all_offsets * box_sizes
    
    # Add offsets to the tiled positions
    periodic_positions = tiled_positions + scaled_offsets
    
    return periodic_positions
#%%

from sklearn.decomposition import PCA

X, Y, Z = np.random.normal(0,1,100), np.random.normal(0,1,100), np.random.normal(0,0.1,100)

XYZ = np.array([X, Y, Z]).T

pca = PCA().fit(XYZ)

left, s, right = np.linalg.svd(XYZ)

print(pca.components_, right)


#%%
if __name__ == '__main__':
    
    k = 150
    
    U = mda.Universe('check.gro')
    PO4 = U.select_atoms('name PO4')
    
    leafs = LF(U, PO4)
    
    points = leafs.groups(0).positions - leafs.groups(0).center_of_geometry()
    
    #%%
    
    periodixy = 'XY'
    
    N = points.shape[0]
    LX, LY, LZ = 2*points[:,0].max(), 2*points[:,1].max(), 2*points[:,2].max()
    
    periodic_points = np.zeros((int(N*9), 3), dtype =  points.dtype)
    
    periodic_points[:N, :]      = points
    periodic_points[N:2*N, :]   = points + np.array([-LX, LY, 0])
    periodic_points[2*N:3*N, :] = points + np.array([0, LY, 0])
    periodic_points[3*N:4*N, :] = points + np.array([LX, LY, 0])
    periodic_points[4*N:5*N, :] = points + np.array([-LX, 0, 0])
    periodic_points[5*N:6*N, :] = points + np.array([LX, 0, 0])
    periodic_points[6*N:7*N, :] = points + np.array([-LX, -LY, 0])
    periodic_points[7*N:8*N, :] = points + np.array([0, -LY, 0])
    periodic_points[8*N:, :]    = points + np.array([LX, -LY, 0])
    
    #%%
    periodic_points = create_periodic_images(points, (LX, LY, LZ), periodic_dims = 'YZ')
    
    fig = mlab.figure(size = (720,720))
    
    mlab.points3d(periodic_points[:,0],
                  periodic_points[:,1],
                  periodic_points[:,2],
                  scale_factor = 10,
                  figure = fig)
    
    mlab.show()
    
    #%%
    
    SURFACE = Surface(points, K = k)
    
    #%%
    
    verts, simps = SURFACE.MeshToArray()
    
    fig = mlab.figure(size = (800, 800))
    
    mlab.triangular_mesh(verts[:,0],
                         verts[:,1],
                         verts[:,2],
                         simps,
                         figure = fig,
                         color  = (1,1,1))
    
    #%%
    
    UV, tri = SURFACE.TriangulateSphere()
    
    fig, ax = plt.subplots(figsize = (7,7))
    
    ax.triplot(UV[:,0], UV[:,1], tri)
    
    #%%
    
    def signed_distance_ellipsoid(C, R, A, P):
        # Transform P into the ellipsoid's coordinate system
        P_transformed = np.linalg.inv(A) @ (P - C)
    
        # Check if the point is inside the ellipsoid
        inside_check = sum((P_transformed / R) ** 2)
    
        # Find the nearest point on the ellipsoid's surface
        P_normalized = P_transformed / R
        norm_P_normalized = np.linalg.norm(P_normalized)
        closest_point_normalized = P_normalized / norm_P_normalized
    
        # Transform this point back to the original coordinate system
        closest_point = A @ (closest_point_normalized * R) + C
    
        # Calculate the distance from P to this point
        distance = np.linalg.norm(P - closest_point)
    
        # Sign the distance based on whether P is inside or outside the ellipsoid
        if inside_check < 1:
            distance = -distance
    
        return distance
    
    inertia_ellipsoid = analysis.compute_inertia_ellipsoid(SURFACE.POINTS)

    h = np.array([signed_distance_ellipsoid(inertia_ellipsoid.center,
                                            inertia_ellipsoid.radii, 
                                            inertia_ellipsoid.axes, 
                                            point) for point in SURFACE.POINTS])
    
    h -= h.mean()
    
    #%%
    
    k = 150
    bwidth = 100
    
    #h = rho
    
    grad_h = kernel_gradient_estimation(SURFACE.POINTS, h, k = k, bandwidth = bwidth)
    
    grad_hx = kernel_gradient_estimation(SURFACE.POINTS, grad_h[:,0], k = k, bandwidth = bwidth)
    grad_hy = kernel_gradient_estimation(SURFACE.POINTS, grad_h[:,1], k = k, bandwidth = bwidth)
    grad_hz = kernel_gradient_estimation(SURFACE.POINTS, grad_h[:,2], k = k, bandwidth = bwidth)
    
    hess_h = np.hstack((grad_hx,
                        grad_hy,
                        grad_hz)).reshape((SURFACE.POINTS.shape[0],
                                           3,3))
    
    lapl_h = np.trace(hess_h, axis1 = 1, axis2 = 2)
                                           
    evals, evecs = np.linalg.eigh(hess_h)
    
    K  = 0.5*(evals[:,1] + evals[:,2])
    KG = (evals[:,1] * evals[:,2])
    
    maxK  = np.min([abs(K.min()),
                    abs(K.max())])
    maxKG = np.min([abs(KG.min()),
                    abs(KG.max())])
    
    x, y, z = SURFACE.POINTS.T
    
    rho, theta, phi = Cart2Sph(x, y, z)
    
    #%%
    
    FaceAreas = AreaPerFace(tri, SURFACE.POINTS)
    
    APL = AreaPerVertex(SURFACE.POINTS, tri, FaceAreas)
    
    
    #%%
    
    fig = mlab.figure(size=(800, 800))
    
    # Plotting the points
    mesh = mlab.triangular_mesh(x,
                                y,
                                z,
                                tri,
                                colormap = 'bwr',
                                vmin = -1*maxK, vmax = maxK,
                                scalars  = K,
                                opacity = 1,
                                figure = fig)

    points= mlab.points3d(x,
                          y,
                          z,
                          color = (0,0,0),
                          scale_factor = 3,
                          figure = fig)
    
    #%%
    
    def GetNeighbors(tri : np.array) -> dict:
        
        Min = np.min(tri)
        Max = np.max(tri)
        
        IDS = np.arange(Min, Max+1, 1)
        
        NEIGHBORS = {}
        
        for i in IDS:
        
            connections = [_ for _ in tri[np.where(tri == i)[0]].ravel() if _ != i]
            
            nghb = []
            for other in connections:
                
                if not other in nghb:
                    nghb.append(other)
            
            NEIGHBORS[i] = nghb
                
        return NEIGHBORS
    
    NGHB = GetNeighbors(tri)
    
    #%%
    
    def FEGradient(position : np.array, 
                   values   : np.array,
                   neighs   : dict) -> np.array:
        
        grad = []
        
        for i, (pos, val) in enumerate(zip(position, values)):
            
            others_ids = neighs[i]
            
            others_pos = position[others_ids]
            others_val = values[others_ids]
            
            d_pos = others_pos - pos[None,:] + 1e-9
            
            d_val = others_val - val
            
            g = np.mean(d_val[:,None]/(d_pos), axis = 0)
            
            grad.append(g)
            
        return np.array(grad)
        
        
    GRAD = FEGradient(points, rho, NGHB)
    
    LAPL = FEGradient(points, GRAD[:,0], NGHB)[:,0] +\
           FEGradient(points, GRAD[:,1], NGHB)[:,1] +\
           FEGradient(points, GRAD[:,2], NGHB)[:,2]
           
    print(LAPL)
    
    #%%
    
    
    fig = mlab.figure(size=(800, 800))
    
    # Plotting the points
    mesh = mlab.triangular_mesh(x,
                                y,
                                z,
                                tri,
                                colormap = 'inferno',
                                #vmin = -1*maxK, vmax = maxK,
                                scalars  = rho,
                                opacity = 1,
                                figure = fig)
    
    #%%
    
    import scipy.special as sp
    
    rho, theta, phi = Cart2Sph(x, y, z)
    
    rho -= rho.mean()
    rho /= rho.max()
    
    def spherical_harmonics_coeffs(r, theta, phi, l_max):
        coeffs = np.zeros((l_max + 1, l_max + 1), dtype=complex)
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                # Y_l^m(theta, phi) complex conjugate
                Y_lm_conj = np.conj(sp.sph_harm(m, l, phi, theta))
                coeffs[l, m] = np.sum(r * Y_lm_conj)
    
        return coeffs

    def plot_power_spectrum(coeffs):
        l_max = coeffs.shape[0] - 1
        powers = np.sum(np.abs(coeffs)**2, axis=1)
    
        plt.figure(figsize=(10, 6))
        plt.plot(range(l_max + 1), powers, marker='o')
        plt.xlabel('Degree (l)')
        plt.ylabel('Power')
        plt.title('Power Spectrum of Spherical Harmonics Coefficients')
        plt.grid(True)
        plt.show()
    
    def plot_spherical_harmonic(l, m, coeffs_lm):
        # Create a grid of points
        phi, theta = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
        r = 0 + np.abs(sp.sph_harm(m, l, phi, theta))  # Absolute value for visualization
        # Convert to Cartesian coordinates for plotting
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
    
        # Plot
        mlab.figure(bgcolor=(1, 1, 1), size = (800, 800))
        mlab.mesh(x, y, z, scalars = r, opacity=1, colormap = 'bwr')
        #mlab.show()
        
    def reconstruct_shape(coeffs, l_max):
        phi, theta = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
        r = np.ones_like(phi, dtype=np.complex_) + np.sum(np.abs(coeffs))
    
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                index_m = m + l  # Adjusted index for m
                if 0 <= index_m < coeffs.shape[1]:  # Ensure index is within bounds
                    r += coeffs[l, index_m] * sp.sph_harm(m, l, phi, theta)
    
        r = np.abs(r)
    
        # Convert to Cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
    
        # Plot
        mlab.figure(bgcolor=(1, 1, 1), size = (800, 800))
        mlab.mesh(x, y, z, scalars=r, opacity=1, colormap = 'bwr')
        #mlab.show()
    
    l_max = 50  # Maximum degree of spherical harmonics
    coeffs = spherical_harmonics_coeffs(rho, theta, phi, l_max)
    
    #coeffs /= coeffs.max()
    
    
    plot_power_spectrum(coeffs)
    
    reconstruct_shape(coeffs, l_max-2)
    