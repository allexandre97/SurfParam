#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:27:53 2024

@author: alexandre
"""

import bpy
import bmesh
import numpy    as np
import open3d   as o3d
import networkx as nx

from mathutils     import Vector
from maths.maths   import Maths
from scipy.spatial import Delaunay

class Surface:
    
    def __init__(self,
                 POINTS      : np.array,
                 NNeighbors  : int = 100,
                 D           : int = 6,
                 Geometry    : str = 'PLANE',
                 Periodicity : str = 'XY') -> None:
        
        if Geometry.upper() == 'PLANE':
            
            self.N_POINTS = len(POINTS)
            
            tempHandler = Maths(POINTS, NNeighbors)
            
            self.POINTS = tempHandler.CreatePeriodicImages((POINTS[:,0].max(),
                                                            POINTS[:,1].max(),
                                                            POINTS[:,2].max()),
                                                           Periodicity)
        else:
            
            self.POINTS = POINTS
            
        
        self.MathHandler = Maths(self.POINTS, NNeighbors)
        
        mesh0 = self.__PoissonSurface(NNeighbors, D)
        
        mesh0.compute_vertex_normals()
        mesh0.compute_triangle_normals()
        
        self.POISSON_VERTEX_NORMALS = np.asarray(mesh0.vertex_normals)
        
        # Create a new mesh and a new object
        self.MESH = bpy.data.meshes.new("TriMesh")
        self.OBJ  = bpy.data.objects.new("TriMeshObj", 
                                         self.MESH)
        
        # Link the object to the scene
        bpy.context.collection.objects.link(self.OBJ)
        bpy.context.view_layer.objects.active = self.OBJ
        self.OBJ.select_set(True)
        
        self.__BlenderMesh(mesh0)
        
    def __PoissonSurface(self,
                         K, D):
        
        pcd = o3d.geometry.PointCloud()
        pcd.points  = o3d.utility.Vector3dVector(self.POINTS)
        pcd.normals = o3d.utility.Vector3dVector(self.MathHandler.EstimateNormals(K))
        pcd.orient_normals_consistent_tangent_plane(K)
        
        self.NORMALS = np.asarray(pcd.normals)
        
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning) as cm:
            mesh0, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth = D)
        
        mesh0 = mesh0.remove_non_manifold_edges()
        mesh0 = mesh0.remove_duplicated_vertices()
        mesh0 = mesh0.remove_degenerate_triangles()
        mesh0 = mesh0.remove_duplicated_triangles()
        mesh0 = mesh0.remove_unreferenced_vertices()
        
        return mesh0
    
    def __BlenderMesh(self,
                      mesh0):
        
        bm = bmesh.new()
        
        # Add vertices
        for v in mesh0.vertices:
            bm.verts.new(Vector(v))
            
        # Update the bmesh's internal structures
        bm.verts.ensure_lookup_table()
        
        # Set to keep track of faces added
        added_faces = set()
        # Add faces (triangles), ensuring no duplicates
        for t in mesh0.triangles:
            # Sort the indices to avoid different orderings of the same face
            sorted_t = tuple(sorted(t))
            if sorted_t not in added_faces:
                bm.faces.new([bm.verts[i] for i in t])
                added_faces.add(sorted_t)
        
        bm.to_mesh(self.MESH)
        bm.free()
        
    def __GenGraph(self,):
        
        G = nx.Graph()
        
        for vertex in self.MESH.vertices:
            
            G.add_node(vertex.index,
                       x = vertex.co[0],
                       y = vertex.co[1],
                       z = vertex.co[2])
        
        for e, edge in enumerate(self.MESH.edges):
            
            i1 = edge.vertices[0]
            i2 = edge.vertices[1]
            
            v1 = G.nodes[i1]
            v2 = G.nodes[i2]
            
            
            d = np.sqrt((v1['x']-v2['x'])**2+
                        (v1['y']-v2['y'])**2+
                        (v1['z']-v2['z'])**2)
            
            G.add_edge(i1, i2, weight = d, index = e)
            
        return G
    
    def AddSeams(self,
                 DIM : str = 'Z'):
        
        G = self.__GenGraph()
        
        if DIM.upper() == 'X':
            sort = sorted(G.nodes, key = lambda n: G.nodes[n]['x'])
        elif DIM.upper() == 'Y':
            sort = sorted(G.nodes, key = lambda n: G.nodes[n]['y'])
        elif DIM.upper() == 'Z':
            sort  = sorted(G.nodes, key = lambda n: G.nodes[n]['z'])
        else:
            
            print('Could not understand dimension input, no seams added')
            return
        
        path = nx.dijkstra_path(G, sort[ 0], sort[-1], weight = 'weight')
        path_edges = []
        for i in range(len(path)-1):        
            path_edges.append(G.edges[path[i], path[i+1]]['index'])
        
        # Assume you are working with the active mesh object in the scene
        obj = bpy.context.active_object
        
        # Make sure we are in edit mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        bm = bmesh.new()
        bm.from_mesh(self.MESH)
        
        for edge in bm.edges:
            
            if edge.index in path_edges:
                edge.seam = True
            else:
                edge.seam = False
        
        # Write the bmesh back to the mesh
        bm.to_mesh(self.MESH)
        bm.free()
        
        # Update the mesh
        obj.data.update()
        
        self.PATH = path
        
    
    
    def UnwrapMesh(self,):
        
        bpy.ops.object.mode_set(mode = 'EDIT')
        bpy.ops.mesh.select_all(action = 'SELECT')
        bpy.ops.uv.unwrap(method = 'ANGLE_BASED',
                          margin = 0.001)
        bpy.ops.object.mode_set(mode = 'OBJECT')
    
    def DeleteUV(self,):
        
        uv_textures = self.OBJ.data.uv_layers
        uv_textures.remove(uv_textures[0])
    
    def __GetTricenters(self,):
        
        bm = bmesh.new()
        bm.from_mesh(self.MESH)
        
        tricenters = []
        
        for face in bm.faces:
            
            verts = np.array([np.asarray(v.co) for v in face.verts])
            tricenters.append(verts.mean(axis = 0))
        
        tricenters = np.array(tricenters)
        
        bm.free()
        
        return tricenters
    
    
    def ProjectPointsOnUV(self,
                          POINTS : np.array):
        
        UV = []
        
        bm = bmesh.new()
        bm.from_mesh(self.MESH)
        
        uv_layer = bm.loops.layers.uv.active
        bm.faces.ensure_lookup_table()
        
        V = np.array([v.co for v in bm.verts])
        
        tmpMaths = Maths(V, )
        triangles = np.array([[v.index for v in face.verts] for face in bm.faces])
        
        _, vert_ids = tmpMaths.KNN(POINTS, 1)
        
        nearest_triangles = np.array([np.where(triangles == i[0])[0][0] for i in vert_ids])
        
        for i, point in enumerate(POINTS):
            
            tri = triangles[nearest_triangles[i]]
            
            A, B, C = V[tri[0]], V[tri[1]], V[tri[2]]

            # Using barycentric coordinates for a 3D point
            v0, v1, v2 = B - A, C - A, point - A
            d00, d01, d11, d20, d21 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1), np.dot(v2, v0), np.dot(v2, v1)
            denom = d00 * d11 - d01 * d01
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1 - v - w
    
            # Mapping to UV coordinates
            face = bm.faces[nearest_triangles[i]]
            uvs = np.array([np.asarray(loop[uv_layer].uv) for loop in face.loops])
            
            uvA, uvB, uvC = uvs[0], uvs[1], uvs[2]
            
            UV.append(u * uvA + v * uvB + w * uvC)
        
        UV = np.array(UV)
        
        bm.free()
        
        return UV
    
    def TriangulateSphere(self,) -> tuple:
        
        z_seams = self.AddSeams('Z')
        self.UnwrapMesh()
        
        UV = self.ProjectPointsOnUV(self.POINTS)
        
        tri1 = Delaunay(UV,
                        qhull_options = 'QJ')
        hull = tri1.convex_hull.ravel()
        
        tri1 = [sorted(t) for t in tri1.simplices if not any(_ in hull for _ in t)]
        
        self.AddSeams('Y')
        self.UnwrapMesh()
        
        UV = self.ProjectPointsOnUV(self.POINTS)
        tri2 = Delaunay(UV,
                        qhull_options = 'QJ')
        hull = tri2.convex_hull.ravel()
        
        tri2 = [sorted(t) for t in tri2.simplices if not any(_ in hull for _ in t)]
        
        tri = list(tri1) + list(tri2)
        
        tri = np.unique(np.vstack(tri), axis = 0)
        
        return UV, tri
    
    def TriangulatePlane(self,) -> tuple:
        
        self.UnwrapMesh()
        UV = self.ProjectPointsOnUV(self.POINTS)
        
        tri = Delaunay(UV,
                       qhull_options = 'QJ')
        hull = tri.convex_hull.ravel()
        
        tri = np.array([sorted(t) for t in tri.simplices if ((not any(_ in hull for _ in t)) and np.any(t < self.N_POINTS))])
        
        return UV, np.unique(tri, axis = 0)
        
    def MeshToArray(self,):
        
        vertices  = []
        simplices = []
        
        bm = bmesh.new()
        bm.from_mesh(self.MESH)
        
        for vertex in bm.verts:
            
            vertices.append(np.asarray(vertex.co))
            
        for face in bm.faces:
            
            triangle = np.array([v.index for v in face.verts])
            simplices.append(triangle)
        
        bm.free()
        
        return np.array(vertices), np.array(simplices)
        
        
        
        
        
    
    