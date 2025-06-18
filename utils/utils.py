import os
import numpy as np
import open3d as o3d


def ExtractRingsAroundTriangles(iInputMesh, iTrianglesList, iMaxRingSize=1) :
  """
  Extracts the rings of triangles around a list of triangles in a mesh.
  
  Parameters:
  - iInputMesh: Open3D TriangleMesh object.
  - iTrianglesList: List of triangle indices to extract rings around.
  - iMaxRingSize: Maximum size of the ring to extract.
  
  Returns:
  - A set of triangle indices that form the ring around the specified triangles.
  """
  # Create halfedge data structure
  halfedgeMesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(iInputMesh)

  # Initialize set for triangles to keep
  trianglesToKeep = set()
  trianglesToKeep.update(iTrianglesList)

  for ringSize in range(iMaxRingSize):
    trianglesInCurrentArea = trianglesToKeep
    trianglesInRing = set()
    
    # Iterate through each triangle index in the input list
    for triangleIndex in trianglesInCurrentArea:
      #Get all triangles around each triangle vertex
      for triangleVertex in halfedgeMesh.triangles[triangleIndex]:
        # Get the half-edges connected to the triangle vertex
        halfEdgesAroundVertex = halfedgeMesh.ordered_half_edge_from_vertex[triangleVertex]
        
        for localHalfEdgeIndex in halfEdgesAroundVertex:
          currentTriangleIndex = halfedgeMesh.half_edges[localHalfEdgeIndex].triangle_index
          if not currentTriangleIndex in trianglesInCurrentArea:
            trianglesInRing.add(currentTriangleIndex)

          neighbourTriangleIndex = halfedgeMesh.half_edges[localHalfEdgeIndex].twin
          if neighbourTriangleIndex== -1:
            raise Exception("@@@@ Halfedge on boundary")

          neighbourTriangleIndex = halfedgeMesh.half_edges[neighbourTriangleIndex].triangle_index

          if not neighbourTriangleIndex in trianglesInCurrentArea:
            trianglesInRing.add(neighbourTriangleIndex)
    
    trianglesToKeep.update(trianglesInRing)
    
  return trianglesToKeep

def CopyMesh(iInputMesh, iTrianglesToCopy = None) :
  trianglesToCopy= iTrianglesToCopy if iTrianglesToCopy is not None else range(len(iInputMesh.triangles))

  newMesh = o3d.geometry.TriangleMesh()

  mapVertices = dict()
  mapTriangles = dict()

  #Duplicate vertices
  for triangleIndex in trianglesToCopy :
    triangleVertices = iInputMesh.triangles[triangleIndex]
    for vertexIndex in triangleVertices :
      if vertexIndex not in mapVertices :
        mapVertices[vertexIndex] = len(newMesh.vertices)
        newMesh.vertices.append(iInputMesh.vertices[vertexIndex])
      
  #Duplicate triangles
  for triangleIndex in trianglesToCopy :
    triangleVertices = iInputMesh.triangles[triangleIndex]
    newTriangle = [mapVertices[vertexIndex] for vertexIndex in triangleVertices]
    mapTriangles[triangleIndex] = len(newMesh.triangles)
    newMesh.triangles.append(newTriangle)

  return newMesh, mapVertices, mapTriangles


def GenerateAreaToRemesh(iInputMesh, iTrianglesIndicesToRemove, iMaxRingSize = 1, iNbSamples = 0) :
  #--- Compute context area : 'iMaxRingSize' rings of triangles around the triangles to remove
  trianglesInContext = ExtractRingsAroundTriangles(iInputMesh, iTrianglesIndicesToRemove, iMaxRingSize)
  
  #--- Create sub mesh with trianglesInContext and compute map between original and sub mesh
  subMesh, mapVertices, mapTriangles=CopyMesh(iInputMesh, trianglesInContext)

  #Debug only
  #o3d.io.write_triangle_mesh("d:/tmp/subMesh.obj", subMesh)

  #--- Sample subMesh (optional)
  sampledPoints=o3d.geometry.PointCloud()
  if iNbSamples > 0 :
    # # Sample points on the mesh
    # sampledPoints = subMesh.sample_points_uniformly(number_of_points=iNbSamples)
    # print(f"Sampled {len(sampledPoints.points)} points on the submesh")

    # Sample points on triangles to remove
    complementSubMesh, _, _ = CopyMesh(iInputMesh, iTrianglesIndicesToRemove)
    sampledPoints = complementSubMesh.sample_points_uniformly(number_of_points=iNbSamples)
    print(f"Sampled {len(sampledPoints.points)} points on the complement submesh")


  #--- Compute the boundary of the area to remesh in submesh
  trianglesToRemoveInSubMesh = [mapTriangles[triangleIndex] for triangleIndex in iTrianglesIndicesToRemove ]
  
  halfedgeSubMesh=o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(subMesh)
  boundaryEdgesOfArea=set()
  for halfEdge in halfedgeSubMesh.half_edges :
    if halfEdge.triangle_index in trianglesToRemoveInSubMesh:
      if halfEdge.twin == -1 :
        continue
      
      #halfEdge.triangle_index is in the area, we consider the twin only if ots triangle is not in the area
      if not halfedgeSubMesh.half_edges[halfEdge.twin].triangle_index in trianglesToRemoveInSubMesh :
        vertex1=halfEdge.vertex_indices[0]
        vertex2=halfEdge.vertex_indices[1]
        if vertex1 > vertex2 :
          vertex1, vertex2 = vertex2, vertex1
        boundaryEdgesOfArea.add((vertex1, vertex2))
  print(f"{len(boundaryEdgesOfArea)} boundary edges on local area")

  #--- Duplicate the subMesh with triangles removed from it
  trianglesToKeepInSubMesh=set(range(len(subMesh.triangles))) - set(trianglesToRemoveInSubMesh)
  subMesh, mapVertices, mapTriangles=CopyMesh(subMesh, trianglesToKeepInSubMesh)
  
  #Debug only
  #o3d.io.write_triangle_mesh("d:/tmp/subMesh_AfterRemove.obj", subMesh)

  #Extract boundary to remesh as chained list of vertices indices
  halfedgeSubMesh=o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(subMesh)
  finalBoundaryHalfEdgesIndices=halfedgeSubMesh.boundary_half_edges_from_vertex(mapVertices[boundaryEdgesOfArea.pop()[0]])

  finalBoundaryVertices=[]
  for i in finalBoundaryHalfEdgesIndices :
    finalBoundaryVertices.append(halfedgeSubMesh.half_edges[i].vertex_indices[0])
  finalBoundaryVertices.append(finalBoundaryVertices[0])  # Close the loop

  return subMesh, finalBoundaryVertices, sampledPoints
    
  
def SaveAreaToRemeshInOBJ(iMesh, iBoundaryVertices, iOutputFileName, iSampledPoints=None) :
  """
  Saves the area to remesh in an OBJ file.
  
  Parameters:
  - iMesh: Open3D TriangleMesh object.
  - iBoundaryVertices: List of vertex indices that form the boundary of the area to remesh.
  - iOutputFileName: Output file name for the OBJ file.
  """
  
  # Save the mesh to an OBJ file
  o3d.io.write_triangle_mesh(iOutputFileName, iMesh)

  #Append to the OBJ the boundary to resmesh as a polyline
  with open(iOutputFileName, 'a') as f:
    f.write("\n# Boundary to remesh\n")
    f.write("g boundary\n")
    f.write("l ")
    for vertexIndex in iBoundaryVertices:
      f.write(f"{vertexIndex+1} ")  # OBJ indices are 1-based
    f.write("\n")

    if iSampledPoints is not None:
      f.write("\n# Sampled points\n")
      f.write("g sampled_points\n")
      for point in iSampledPoints.points:
        f.write(f"v {point[0]} {point[1]} {point[2]}\n")
      f.write("\n# End of sampled points\n")


def LoadAreaToRemeshFromOBJ(iInputFileName) :
  """
  Loads the area to remesh from an OBJ file.
  
  Parameters:
  - iInputFileName: Input file name for the OBJ file.
  
  Returns:
  - A tuple containing the Open3D TriangleMesh object, the boundary vertices, and the sampled points.
  """
  
  mesh = o3d.io.read_triangle_mesh(iInputFileName)
  
  # Extract boundary vertices
  boundaryVertices = []
  sampledPoints = None

  with open(iInputFileName, 'r') as f:
    for line in f:
      if line.startswith("l "):
        boundaryVertices = list(map(int, line.strip().split()[1:]))
        # Convert to zero-based indices
        boundaryVertices = [v - 1 for v in boundaryVertices]
      elif line.startswith("g sampled_points"):
        # Read sampled points
        readPoints=[]
        for pointLine in f:
          if pointLine.startswith("v "):
            point = list(map(float, pointLine.strip().split()[1:]))
            readPoints.append(point)
        sampledPoints = o3d.geometry.PointCloud()
        sampledPoints.points = o3d.utility.Vector3dVector(readPoints)

  return mesh, boundaryVertices, sampledPoints
