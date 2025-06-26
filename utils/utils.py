import os
import numpy as np
import open3d as o3d
import colorsys
from fns import center_vertices, normalize_vertices_scale, quantize_verts, dequantize_verts



def ExtractRingsAroundTriangles(iInputMesh, iTrianglesList, iMaxRingSize=1) :
  """
  Extracts the rings of triangles around a list of triangles in a mesh.
  
  Parameters:
  - iInputMesh: Open3D TriangleMesh object.
  - iTrianglesList: List of triangle indices to extract rings around.
  - iMaxRingSize: Maximum size of the ring to extract.
                  If -1 : returns all the triangles
  
  Returns:
  - A set of triangle indices that form the ring around the specified triangles.
  """
  if iMaxRingSize==-1 :
    # If max ring size is -1, return all triangles in the mesh
    return set(range(len(iInputMesh.triangles)))
  else :
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
  """
  Generates the area to remesh by extracting a submesh around specified triangles
   and computes sampling points.
  
  Parameters:
  - iInputMesh: Input mesh from which a sub mesh will be extracted, Open3D TriangleMesh object.
  - iTrianglesIndicesToRemove: List of triangle indices to remove from the input mesh to generate the area to remesh.
  - iMaxRingSize: Number of faces rings to extract around the triangles to remove to generate a context area.
                  Default is 1, if -1 the complement of the triangles to remove is returned.
  - iNbSamples : Number of points to sample on the submesh.
  
  Returns:
  - A tuple containing the Open3D TriangleMesh object, the boundary vertices, and the sampled points.
  """

  #--- Compute context area : 'iMaxRingSize' rings of triangles around the triangles to remove
  trianglesInContext = ExtractRingsAroundTriangles(iInputMesh, iTrianglesIndicesToRemove, iMaxRingSize)
  
  #--- Create sub mesh with trianglesInContext and compute map between original and sub mesh
  subMesh, mapVertices, mapTriangles=CopyMesh(iInputMesh, trianglesInContext)

  #Debug only
  #o3d.io.write_triangle_mesh("d:/tmp/subMesh.obj", subMesh)

  #--- Sample subMesh (optional)
  sampledPoints=o3d.geometry.PointCloud()
  if iNbSamples > 0 :
    # Sample points on the sub mesh (ie context + triangles to remove)
    sampledPoints = subMesh.sample_points_uniformly(number_of_points=iNbSamples)
    print(f"Sampled {len(sampledPoints.points)} points on the submesh")

    # # Sample points on triangles to remove (hole)
    # complementSubMesh, _, _ = CopyMesh(iInputMesh, iTrianglesIndicesToRemove)
    # sampledPoints = complementSubMesh.sample_points_uniformly(number_of_points=iNbSamples)
    # print(f"Sampled {len(sampledPoints.points)} points on the complement submesh")

    # # Sample points on context only (hole)
    # contextSubMesh, _, _ = CopyMesh(iInputMesh, set(trianglesInContext)-set(iTrianglesIndicesToRemove))
    # sampledPoints = contextSubMesh.sample_points_uniformly(number_of_points=iNbSamples)
    # print(f"Sampled {len(sampledPoints.points)} points on the complement submesh")


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
  #o3d.io.write_triangle_mesh("./subMesh_AfterRemove.obj", subMesh)

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


#Save OBJ file
def SaveOBJ(vertices, faces, filepath : str, colorOrderedVertices = False) :
  
  startHSVColor=np.array(colorsys.rgb_to_hsv(255, 0, 0))
  endHSVColor=np.array(colorsys.rgb_to_hsv(0, 255, 0))

  nbVertices=len(vertices)
  with open(filepath, 'w') as file :
      for vertexIndex, vertex in enumerate(vertices) :
          if colorOrderedVertices :
            c=vertexIndex/(nbVertices-1.0)
            hsvColor=(1.0-c)*startHSVColor+c*endHSVColor
            rgbColor=np.array(colorsys.hsv_to_rgb(hsvColor[0], hsvColor[1], hsvColor[2]))
            rgbColor/=255.0
            file.write(f"v  {vertex[0]}  {vertex[1]}  {vertex[2]}  {rgbColor[0]}  {rgbColor[1]}  {rgbColor[2]}\n")
          else :
            file.write(f"v  {vertex[0]}  {vertex[1]}  {vertex[2]}\n")
      
      for face in faces :
          file.write("f")
          for elem in face :
            file.write(f"  {elem+1}")
          file.write("\n")


#-- Rotate the mesh to align the area to remesh with the vector (1, 1, 1)
def AlignBoundaryWithVector(mesh, iTrianglesToRemesh, vector=(1, 1, 1)):
    # Compute the centroid of the mesh
    centroid = mesh.get_center()
    
    #Compute the centroid of the area to remesh
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    area_vertices = vertices[triangles[list(iTrianglesToRemesh)]]
    area_vertices=area_vertices.reshape(-1, 3)  # Flatten to a 2D array
    area_centroid = np.mean(area_vertices, axis=0)
    tempDirection = area_centroid - centroid

    #Normalize vectors
    tempDirection /= np.linalg.norm(tempDirection)
    vector /= np.linalg.norm(vector)
    
    #Compute rotation matrix to align normal with the vector
    rotation_axis = np.cross(tempDirection, vector)
    rotation_angle = np.arccos(np.clip(np.dot(tempDirection, vector), -1.0, 1.0))
    if np.linalg.norm(rotation_axis) < 1e-6:
        return mesh  # No rotation needed if already aligned
    
    rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize the rotation axis
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
    
    #Rotate the mesh
    mesh.rotate(rotation_matrix, center=centroid)
    return mesh


def NormalizeSampleAndGridMesh(iInputMesh : o3d.geometry.TriangleMesh, iListOfTrianglesToTrack : list, iQuantBits : int = 7, iNbSamples : int = 8192) :
  #--- Normalize mesh
  vertices = np.asarray(iInputMesh.vertices)
  vertices = center_vertices(vertices)
  vertices = normalize_vertices_scale(vertices)
  vertices = np.clip(vertices, a_min=-0.5, a_max = 0.5)
  triangles = np.asarray(iInputMesh.triangles)

  # #--- Sampling -> before gridding
  # sampledPoints=o3d.geometry.PointCloud()
  # if iNbSamples > 0 :
  #   normalizedMesh=o3d.geometry.TriangleMesh()
  #   normalizedMesh.vertices=o3d.utility.Vector3dVector(vertices)
  #   normalizedMesh.triangles=o3d.utility.Vector3iVector(triangles)

  #   sampledPoints = normalizedMesh.sample_points_uniformly(number_of_points=iNbSamples)
  #   print(f"Sampled {len(sampledPoints.points)} points on the normalized mesh")

  #--- Prepare triangles tracking before gridding
  mapTriangleCenter=dict()
  for triangleIndex in iListOfTrianglesToTrack :
    mapTriangleCenter[triangleIndex]=np.mean(vertices[triangles[triangleIndex]], axis=0)

  #--- Grid mesh
  vertices = quantize_verts(vertices, n_bits = iQuantBits)
  vertices = dequantize_verts(vertices, n_bits= iQuantBits)

  griddedMesh=o3d.geometry.TriangleMesh()
  griddedMesh.vertices=o3d.utility.Vector3dVector(vertices)
  griddedMesh.triangles=o3d.utility.Vector3iVector(triangles)
  griddedMesh.remove_duplicated_vertices()
  griddedMesh.remove_duplicated_triangles()
  griddedMesh.remove_degenerate_triangles()
  griddedMesh.remove_unreferenced_vertices()

  if not griddedMesh.is_watertight() :
    raise Exception("@@@@ Gridded mesh is not watertight")
  if not griddedMesh.is_edge_manifold() :
    raise Exception("@@@@ Gridded mesh is not edge manifold")
  if not griddedMesh.is_vertex_manifold() :
    raise Exception("@@@@ Gridded mesh is not vertex manifold")
  
  #Debug
  #o3d.io.write_triangle_mesh("./output/griddedMesh.obj", griddedMesh)
  
  #--- Sampling -> after gridding
  sampledPoints=o3d.geometry.PointCloud()
  if iNbSamples > 0 :
    sampledPoints = griddedMesh.sample_points_uniformly(number_of_points=iNbSamples)
    print(f"Sampled {len(sampledPoints.points)} points on the normalized mesh")

  #--- Track triangles after gridding
  newListOfTrianglesToTrack=[]

  griddedMeshForRC=o3d.t.geometry.TriangleMesh.from_legacy(griddedMesh)
  scene = o3d.t.geometry.RaycastingScene()
  scene.add_triangles(griddedMeshForRC)
  for triangleIndex, triangleCenter in mapTriangleCenter.items() :
    query_point = o3d.core.Tensor([triangleCenter], dtype=o3d.core.Dtype.Float32)
    query_result= scene.compute_closest_points(query_point)
    closestTriangleIndex=query_result['primitive_ids'][0].item()
    newListOfTrianglesToTrack.append(closestTriangleIndex)

  #newListOfTrianglesToTrack=iListOfTrianglesToTrack

  return griddedMesh, sampledPoints, newListOfTrianglesToTrack








  