from model.treemeshgpt_inference import TreeMeshGPT
import os
import numpy as np
import open3d as o3d
import torch
from utils.utils import GenerateAreaToRemesh, SaveAreaToRemeshInOBJ, ExtractRingsAroundTriangles, SaveOBJ, NormalizeSampleAndGridMesh, AlignBoundaryWithVector
from tokenizer import prepare_halfedge_mesh
import trimesh
import time


def GenerateMeshToComplete(iMesh : o3d.geometry.TriangleMesh, iTriangleListToRemesh : list, iNbRingsAroundTrianglesToRemove = 1, iNbSamples = 8192, iAlignBoundaryWithVector = None, iDebugPrefixPath = "") :
    
    #--- Identify triangles to be removed
    triangleListToRemesh=ExtractRingsAroundTriangles(iMesh, iTriangleListToRemesh, iMaxRingSize=iNbRingsAroundTrianglesToRemove)

    #--- Rotate the mesh to align the area to remesh with a specified direction
    if iAlignBoundaryWithVector :
      iMesh = AlignBoundaryWithVector(iMesh, triangleListToRemesh, iAlignBoundaryWithVector)
      o3d.io.write_triangle_mesh(iDebugPrefixPath+"_AlignBoundaryWithVector.obj", iMesh)

    #--- Normalize and grid mesh
    normalizedMesh, sampledPoints, triangleListToRemesh=NormalizeSampleAndGridMesh(iMesh, list(triangleListToRemesh), iNbSamples=iNbSamples)

    #--- Reorder mesh elements
    he_mesh, _, _, triangleListToRemesh=prepare_halfedge_mesh(np.asarray(normalizedMesh.vertices), np.asarray(normalizedMesh.triangles, dtype=np.int32), triangleListToRemesh)
    # he_mesh = o3d.geometry.TriangleMesh()
    # he_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # he_mesh.triangles = o3d.utility.Vector3iVector(triangles)

    #Debug : save normalized and reorderedmesh
    localMesh = o3d.geometry.TriangleMesh()
    localMesh.vertices = he_mesh.vertices
    localMesh.triangles = he_mesh.triangles

    if iDebugPrefixPath :
      o3d.io.write_triangle_mesh(iDebugPrefixPath+"_normalized.obj", localMesh)
      SaveOBJ(he_mesh.vertices, he_mesh.triangles, iDebugPrefixPath+"_normalizedColored.obj", True)

    #--- Extract boundary to remesh
    CONTEXT_RING=-1
    submesh, remeshBoundary, _=GenerateAreaToRemesh(localMesh, triangleListToRemesh, iMaxRingSize=CONTEXT_RING, iNbSamples=0)

    #Debug : Save area to remesh in OBJ
    if iDebugPrefixPath :
      SaveAreaToRemeshInOBJ(submesh, remeshBoundary, iDebugPrefixPath+"_area2remesh.obj", iSampledPoints=sampledPoints)
      o3d.io.write_point_cloud(iDebugPrefixPath+"_Sampling.xyz", sampledPoints)

    return submesh, remeshBoundary, sampledPoints


def GenerateMeshToCompleteFromPath(iMeshPath : str, iTriangleListToRemesh : list, iNbRingsAroundTrianglesToRemove = 1, iNbSamples = 8192, iAlignBoundaryWithVector = None, iDebugPrefixPath = "") :
    #--- Load and normalize mesh
    mesh = o3d.io.read_triangle_mesh(iMeshPath)

    return GenerateMeshToComplete(mesh, iTriangleListToRemesh, iNbRingsAroundTrianglesToRemove, iNbSamples, iAlignBoundaryWithVector, iDebugPrefixPath)


def CompleteMesh(iTransformer : TreeMeshGPT, device,
                 iSubMesh : o3d.geometry.TriangleMesh, iBoundaryVertices : list, iSupportPC : o3d.geometry.PointCloud, iDebugPrefixPath = "") :
    
    #--- Check number of faces
    DECIMATION_TARGET_NFACES=5000

    print("Number of faces to remesh: ", len(iSubMesh.triangles))
    if len(iSubMesh.triangles) >= DECIMATION_TARGET_NFACES:
      raise Exception("@@@@ Number of faces to remesh is larger than target number of faces")

    #--- Point cloud sampling structures
    pc_array = np.asarray(iSupportPC.points)
    pc = torch.tensor(pc_array).unsqueeze(0).float().to(device)

    #--- Halfedge mesh
    halfEdgeTriangularMesh=o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(iSubMesh)

    #--- Generate completion
    t = time.time()
    #with torch.autocast(device_type=“cuda”, dtype=torch.float16), torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.float16), torch.no_grad():
        out_faces = iTransformer.generate_completion(halfEdgeTriangularMesh, iBoundaryVertices, pc, n = 0.25)
        #out_faces = iTransformer.generate_completion(halfEdgeTriangularMesh, remeshBoundary, pc, n = 0.0)
    elapsedTime=t = time.time()-t

    #--- Post process results
    vertices = out_faces.view(-1, 3).cpu().numpy()
    n = vertices.shape[0]
    faces = torch.arange(1, n + 1).view(-1, 3).numpy()

    if iDebugPrefixPath :
      with open(iDebugPrefixPath+"_GeneratedVertices.obj", "w") as file :
        for vertex in vertices :
          file.write(f"v  {vertex[0]}  {vertex[1]}  {vertex[2]}\n")

      #Debug : save generated geometry
      with open(iDebugPrefixPath+"_GeneratedFaces.obj", "w") as file :
        for vertex in vertices :
          file.write(f"v  {vertex[0]}  {vertex[1]}  {vertex[2]}\n")

        for face in faces :
          file.write(f"f  {face[0]}  {face[1]}  {face[2]}\n")

    if min(min(faces.tolist())) == 1:
        faces = (np.array(faces) - 1)

    #Remove collapsed triangles and duplicates
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    collapsed_mask = np.all(p0 == p1, axis=1) | np.all(p0 == p2, axis=1) | np.all(p1 == p2, axis=1)
    faces = faces[~collapsed_mask]
    faces = faces.tolist()
    scene_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, force="mesh",
                            merge_primitives=True)
    scene_mesh.merge_vertices()
    scene_mesh.update_faces(scene_mesh.nondegenerate_faces())
    scene_mesh.update_faces(scene_mesh.unique_faces())
    scene_mesh.remove_unreferenced_vertices()
    scene_mesh.fix_normals()

    del out_faces
    torch.cuda.empty_cache()

    #Create result mesh
    triangles = np.asarray(scene_mesh.faces)
    vertices = np.asarray(scene_mesh.vertices)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    torch.cuda.empty_cache()
    
    print(f"Completion done in {elapsedTime:.2f} seconds.")

    return mesh



