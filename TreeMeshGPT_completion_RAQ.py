from model.treemeshgpt_inference import TreeMeshGPT
import os
import numpy as np
import open3d as o3d
import torch
from accelerate import Accelerator
from pathlib import Path
from fns import center_vertices, normalize_vertices_scale, str2bool
import trimesh
import pyvista as pv
from utils.utils import GenerateAreaToRemesh, SaveAreaToRemeshInOBJ 

VERSION = "7bit"
CKPT_PATH = "./checkpoints/treemeshgpt_7bit.pt"

OUTPUT_DIR="./output"

DECIMATION_TARGET_NFACES = 5000
SAMPLING = "uniform" if VERSION == "7bit" else "fps"

TORCH_DEVICE="cuda:1"

if not os.path.exists("./output") :
  os.mkdir("./output")

# Set up model
torch.device(TORCH_DEVICE)

transformer = TreeMeshGPT(quant_bit = 7 if VERSION == "7bit" else 9, max_seq_len=13000) # can set higher max_seq_len if GPU is L4 or A100
transformer.load(CKPT_PATH)
accelerator = Accelerator(mixed_precision="fp16")
transformer = accelerator.prepare(transformer)

MESH_PATH = "demo/Mesh2.obj"
TRIANGLES_TO_REMESH=[ 107, 106, 105, 104, 103, 102,
                      117, 116, 115, 114, 113, 112,
                      127, 126, 125, 124, 123, 122 ]


#--- Load and normalize mesh
mesh = o3d.io.read_triangle_mesh(MESH_PATH)
vertices = np.asarray(mesh.vertices)
vertices = center_vertices(vertices)
vertices = normalize_vertices_scale(vertices)
vertices = np.clip(vertices, a_min=-0.5, a_max = 0.5)
triangles = np.asarray(mesh.triangles)

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(triangles)

#Debug : save normalized mesh
o3d.io.write_triangle_mesh(OUTPUT_DIR+"/"+"normalized_"+os.path.split(MESH_PATH)[1], mesh)

#--- Extract boundary to remesh
submesh, remeshBoundary, sampledPoints=GenerateAreaToRemesh(mesh, TRIANGLES_TO_REMESH, iNbSamples=2048)

#Debug : Save area to remesh in OBJ
SaveAreaToRemeshInOBJ(submesh, remeshBoundary, OUTPUT_DIR+"/"+"area2remesh_"+os.path.split(MESH_PATH)[1], iSampledPoints=sampledPoints)
o3d.io.write_point_cloud(OUTPUT_DIR+"/"+"Sampling_"+os.path.split(MESH_PATH)[1]+".ply", sampledPoints)

#Check number of faces
print("Number of faces to remesh: ", len(submesh.triangles))
if len(submesh.triangles) >= DECIMATION_TARGET_NFACES:
    raise Exception("@@@@ Number of faces to remesh is larger than target number of faces")


#Point cloud sampling structures
pc_array = np.asarray(sampledPoints.points)
pc = torch.tensor(pc_array).unsqueeze(0).float().cuda()

#Halfedge mesh
halfEdgeTriangularMesh=o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(submesh)

# Generation
with accelerator.autocast(), torch.no_grad():
    out_faces = transformer.generate_completion(halfEdgeTriangularMesh, remeshBoundary, pc, n = 0.25)

vertices = out_faces.view(-1, 3).cpu().numpy()
n = vertices.shape[0]
faces = torch.arange(1, n + 1).view(-1, 3).numpy()

with open(OUTPUT_DIR+"/"+"GeneratedVertices_"+os.path.split(MESH_PATH)[1], "w") as file :
  for vertex in vertices :
    file.write(f"v  {vertex[0]}  {vertex[1]}  {vertex[2]}\n")

with open(OUTPUT_DIR+"/"+"GeneratedFaces_"+os.path.split(MESH_PATH)[1], "w") as file :
  for vertex in vertices :
    file.write(f"v  {vertex[0]}  {vertex[1]}  {vertex[2]}\n")

  for face in faces :
    file.write(f"f  {face[0]}  {face[1]}  {face[2]}\n")

if min(min(faces.tolist())) == 1:
    faces = (np.array(faces) - 1)

# Remove collapsed triangles and duplicates
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


# del out_faces
# torch.cuda.empty_cache()


# # Plot mesh from: https://colab.research.google.com/drive/1CR_HDvJ2AnjJV3Bf5vwP70K0hx3RcdMb?usp=sharing#scrollTo=kXi90AcckMF5

# triangles = np.asarray(scene_mesh.faces)
# vertices = np.asarray(scene_mesh.vertices)
# colors = None

# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(vertices)
# mesh.triangles = o3d.utility.Vector3iVector(triangles)

# if not mesh.has_vertex_normals(): mesh.compute_vertex_normals()
# if not mesh.has_triangle_normals(): mesh.compute_triangle_normals()

# if mesh.has_triangle_normals():
#     colors = (0.5, 0.5, 0.5) + np.asarray(mesh.triangle_normals) * 0.5
#     colors = tuple(map(tuple, colors))
# else:
#     colors = (1.0, 0.0, 0.0)

# import plotly.graph_objects as go

# fig = go.Figure(
#     data=[
#         go.Mesh3d(
#             x=vertices[:,0],
#             y=vertices[:,1],
#             z=vertices[:,2],
#             i=triangles[:,0],
#             j=triangles[:,1],
#             k=triangles[:,2],
#             facecolor=colors,
#             opacity=0.50)
#     ],
#     layout=dict(
#         scene=dict(
#             xaxis=dict(visible=False),
#             yaxis=dict(visible=False),
#             zaxis=dict(visible=False)
#         )
#     )
# )
# fig.show()

# Save mesh if necessary
outputFilePath="./output/out_"+os.path.split(MESH_PATH)[1]
scene_mesh.export(outputFilePath)

