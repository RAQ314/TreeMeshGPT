import os
import numpy as np
import open3d as o3d



def GenerateAreaToRemesh(iInputMesh, iTrianglesIndicesToRemove) :
  #Duplicate input mesh
  triangularMesh=o3d.geometry.TriangleMesh()
  triangularMesh.vertices=iInputMesh.vertices
  triangularMesh.triangles=iInputMesh.triangles
  
  #Create halfedge data structure
  halfedgeMesh=o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(triangularMesh)

  #Extract halfedges on the boundary of the area
  boundaryEdgesOfArea=set()
  for halfEdge in halfedgeMesh.half_edges :
    isOnBoudaryArea=(halfEdge.triangle_index in iTrianglesIndicesToRemove)
    isTwinOnBoudaryArea=(halfedgeMesh.half_edges[halfEdge.twin].triangle_index in iTrianglesIndicesToRemove)
    if isOnBoudaryArea != isTwinOnBoudaryArea : #xor
      boundaryEdgesOfArea.add(halfEdge)

  print(f"{len(boundaryEdgesOfArea)} boundary edges on local area")

  #Remove triangles from the area to remesh
  triangularMesh.remove_triangles_by_index(iTrianglesIndicesToRemove)

  #Extract boundary to remesh as chained list of vertices indices
  halfedgeMesh=o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(triangularMesh)  #Create halfedge structure anew
  finalBoundaryVertices=halfedgeMesh.boundary_vertices_from_vertex(boundaryEdgesOfArea[0][0])

  #Extract the one ring of triangles around the boundary
  oneRingOfFaces=set()
  for vertex in finalBoundaryVertices :
    for localHE in halfedgeMesh :
      halfedgeMesh.add(halfedgeMesh.halfedgeMesh[localHE].triangle_index)
      halfedgeMesh.add(halfedgeMesh.halfedgeMesh[localHE.twin].triangle_index)
  print(f"{len(oneRingOfFaces)} triangles to keep around boundary (one ring)")

  #....

  #Remove all other triangles to generate the final mesh
  #......
  


    
  
    

