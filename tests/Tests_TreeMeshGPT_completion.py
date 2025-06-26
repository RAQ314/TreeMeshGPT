import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from multiprocessing import Process, Pipe, Queue
import numpy as np
import open3d as o3d
import torch
from pathlib import Path
from model.treemeshgpt_inference import TreeMeshGPT
from inference_completion import *
import time
import pandas as pd
import datetime
import random


#--- Init
USE_MULTIPROCESS = True
NB_SAMPLING_POINTS = 8192

VERSION = "7bit"
CKPT_PATH = "./checkpoints/treemeshgpt_7bit.pt"

TORCH_DEVICE="cuda:1"


def PerformCompletion(ioQueue, iMeshPath : str, iTriangleListToRemesh : list, iNbRingsAroundTrianglesToRemove, iTrialNumber : int, iKPILineToFill : list, iNbSamples = 8192, iDebugPrefixPath = "") :

  #--- Init
  random.seed(42)
  torch.manual_seed(42)
  np.random.seed(42)
  o3d.utility.random.seed(42)

  #Set cuda device
  device=torch.device(TORCH_DEVICE)

  #--- Debug only
  tempMesh=o3d.io.read_triangle_mesh(iMeshPath)
  nbInputVertices=len(tempMesh.vertices)
  nbInputTriangles=len(tempMesh.triangles)
  iKPILineToFill.append(nbInputVertices)
  iKPILineToFill.append(nbInputTriangles)
  iKPILineToFill.append(iTriangleListToRemesh[0])

  #--- Generate submesh to complete
  submesh, remeshBoundary, sampledPoints=GenerateMeshToCompleteFromPath(iMeshPath, iTriangleListToRemesh, iNbRingsAroundTrianglesToRemove, iNbSamples=iNbSamples, iDebugPrefixPath=iDebugPrefixPath)
  nbRemovedTriangles=nbInputTriangles-len(submesh.triangles)
  iKPILineToFill.append(nbRemovedTriangles)
  iKPILineToFill.append(iTrialNumber)

  #--- Set up model
  transformer = TreeMeshGPT(quant_bit = 7 if VERSION == "7bit" else 9, max_seq_len=13000).to(device) # can set higher max_seq_len if GPU is L4 or A100
  transformer.load(CKPT_PATH)

  #--- Complete submesh
  startTime=time.time()
  completedMesh=CompleteMesh(transformer, device, submesh, remeshBoundary, sampledPoints, iDebugPrefixPath)
  endTime=time.time()
  elapsedTime=endTime-startTime
  iKPILineToFill.append(elapsedTime)

  #Save mesh
  if iDebugPrefixPath :
    o3d.io.write_triangle_mesh(iDebugPrefixPath+"_COMPLETION.obj", completedMesh)

  #--- Dump stats :
  nbResultVertices=len(completedMesh.vertices)
  nbResultTriangles=len(completedMesh.triangles)
  nbAddedFaces=nbResultTriangles-len(submesh.triangles)

  iKPILineToFill.append(nbResultVertices)
  iKPILineToFill.append(nbResultTriangles)
  iKPILineToFill.append(nbAddedFaces)
  
  #Watertightness and free edges
  isWatertight=True if completedMesh.is_watertight() else False
  iKPILineToFill.append(isWatertight)
  if isWatertight :
    iKPILineToFill.append(0)
  else :
    #Get number of boundary edges, easy to get from halfedge structure but an exception can be raised
    #if the mesh is not manifold
    try :
      tempHEMesh=o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(completedMesh)
      
      nbFreeEdges=0
      for he in tempHEMesh.half_edges :
        if he.twin==-1 : nbFreeEdges+=1
      
      iKPILineToFill.append(nbFreeEdges)
    except :
      iKPILineToFill.append("NA")

  #Manifoldness
  iKPILineToFill.append(True if completedMesh.is_edge_manifold() else False)

  #return iKPILineToFill
  ioQueue.put(iKPILineToFill)
  
def RunAllTests() :
  
  #Set cuda device
  torch.device(TORCH_DEVICE)

  if not os.path.exists("./output") :
    os.mkdir("./output")

  class CompletionTestConfig(object) :
    meshPath=""
    listsOfTrianglesToRemesh=[]   #List of lists, each sub list containing the seed triangles defining an area to remesh
    listOfRingSizes=[]            #List of ring size around each seed triangles, each ring size will be sued for each seed triangles list.
    nbRunsPerTrial=1              #Nb runs per completion test

    def __init__(self, meshPath, listsOfTrianglesToRemesh, listOfRingSizes, nbRunsPerTrial = 1) :
      self.meshPath=meshPath
      self.listsOfTrianglesToRemesh=listsOfTrianglesToRemesh
      self.listOfRingSizes=listOfRingSizes
      self.nbRunsPerTrial=nbRunsPerTrial

  df=pd.DataFrame({ "run id":[], "model":[], "nb input vertices": [], "nb input faces": [], "seed face": [], "remesh size": [], "trial id": [], "run time": [],
                    "nb output vertices": [], "nb output faces": [], "nb added faces": [], "watertight":[], "nb free edges":[], "manifold":[]})


  allCompletionsTests=[]
  #allCompletionsTests.append(CompletionTestConfig("./demo/NewMesh1_Tri.obj", [[394]], [1], 3))
  allCompletionsTests.append(CompletionTestConfig("./demo/NewMesh1_Tri.obj", [[394], [174], [205], [275], [552], [85]], [1, 2, 3], 3))
  allCompletionsTests.append(CompletionTestConfig("./demo/NewMesh2_Tri.obj", [[242], [170], [276], [205], [45], [434]], [1, 2, 3], 3))
  allCompletionsTests.append(CompletionTestConfig("./demo/NewMesh3_Tri.obj", [[91], [41], [63], [52], [11], [137]], [1, 2, 3], 3))
  allCompletionsTests.append(CompletionTestConfig("./demo/115603_50901239_5.obj", [[662], [1986], [217]], [1, 2, 3], 3))
  allCompletionsTests.append(CompletionTestConfig("./demo/115603_50901239_23.obj", [[1981], [574]], [1, 2, 3], 3))
  allCompletionsTests.append(CompletionTestConfig("./demo/115615_d06fa061_6.obj", [[484], [1179], [1508]], [1, 2, 3], 3))
  allCompletionsTests.append(CompletionTestConfig("./demo/128043_372dc2bb_0.obj", [[2001], [473], [1559], [1988], [1687]], [1, 2, 3], 3))
  allCompletionsTests.append(CompletionTestConfig("./demo/objaverse_pig_CC0_Decim_2k.obj", [[1129], [1602], [783], [268]], [1, 2, 3], 3))


  #--- Run tests
  runCounter=0

  for currentTest in allCompletionsTests :

    #Debug : Extract mesh name
    meshName, _=os.path.splitext(os.path.basename(currentTest.meshPath))
    debugPrefixPathBase="./output/"+meshName

    for currentAreaToRemesh in currentTest.listsOfTrianglesToRemesh :
      for currentRingSize in currentTest.listOfRingSizes :
        for trialNumber in range(currentTest.nbRunsPerTrial) :

          print(f"Processing mesh : {currentTest.meshPath}", flush=True)
          
          runCounter+=1
          debugPrefixPath=debugPrefixPathBase+"_run"+f"{runCounter:3}"

          kpiLine=[]
          kpiLine.append(runCounter)
          kpiLine.append(meshName)

          if USE_MULTIPROCESS :
            subProcessQueue=Queue()
            subProcess=Process(target=PerformCompletion, args=(subProcessQueue, currentTest.meshPath, currentAreaToRemesh, currentRingSize, trialNumber, kpiLine, NB_SAMPLING_POINTS, debugPrefixPath,))
            subProcess.start()
            subProcess.join()
            if subProcess.exitcode==0 :
              kpiLine=subProcessQueue.get()
            else :
              kpiLineSize=len(kpiLine)
              for i in range(len(df.columns)-kpiLineSize) :
                kpiLine.append("NA")

          else :
            subProcessQueue=Queue()
            PerformCompletion(subProcessQueue, currentTest.meshPath, currentAreaToRemesh, currentRingSize, trialNumber, kpiLine, NB_SAMPLING_POINTS, debugPrefixPath)
            kpiLine=subProcessQueue.get()
          
          #Fill dataframe
          df.loc[len(df)] = kpiLine
          print("## ", *kpiLine, sep="\t")
          print("\n\n")


  #Save dataframe in a csv file named after date and time
  csvFileName="./output/KPI_Completion_"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".csv"
  df.to_csv(csvFileName, index=False)

        
if __name__ == '__main__':
  torch.multiprocessing.set_start_method('spawn')
  RunAllTests()



