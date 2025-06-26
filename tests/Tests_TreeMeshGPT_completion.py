import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from concurrent.futures import *
from multiprocessing import Process, Pipe, Queue
import numpy as np
import open3d as o3d
import torch
from pathlib import Path
from model.treemeshgpt_inference import TreeMeshGPT
from utils.utils import GenerateAreaToRemesh, SaveAreaToRemeshInOBJ 
from inference_completion import *
import time
import pandas as pd
import datetime


#--- Init
USE_MULTIPROCESS = True
NB_SAMPLING_POINTS = 8192

VERSION = "7bit"
CKPT_PATH = "./checkpoints/treemeshgpt_7bit.pt"

#Set cuda device
TORCH_DEVICE="cuda:1"
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
allCompletionsTests.append(CompletionTestConfig("./demo/NewMesh1_Tri.obj", [[394], [174], [205], [275], [552], [85]], [1, 2, 3], 3))
allCompletionsTests.append(CompletionTestConfig("./demo/NewMesh2_Tri.obj", [[242], [170], [276], [205], [45], [434]], [1, 2, 3], 3))
allCompletionsTests.append(CompletionTestConfig("./demo/NewMesh3_Tri.obj", [[91], [41], [63], [52], [11], [137]], [1, 2, 3], 3))
allCompletionsTests.append(CompletionTestConfig("./demo/115603_50901239_5.obj", [[662], [1986], [217]], [1, 2, 3], 3))
allCompletionsTests.append(CompletionTestConfig("./demo/115603_50901239_23.obj", [[1981], [574]], [1, 2, 3], 3))
allCompletionsTests.append(CompletionTestConfig("./demo/115615_d06fa061_6.obj", [[484], [1179], [1508]], [1, 2, 3], 3))
allCompletionsTests.append(CompletionTestConfig("./demo/128043_372dc2bb_0.obj", [[2001], [473], [1559], [1988], [1687]], [1, 2, 3], 3))
allCompletionsTests.append(CompletionTestConfig("./demo/objaverse_pig_CC0_Decim_2k.obj", [[1129], [1602], [783], [268]], [1, 2, 3], 3))


def PerformCompletion(ioQueue, iMeshPath : str, iTriangleListToRemesh : list, iNbRingsAroundTrianglesToRemove, iKPILineToFill : list, iNbSamples = 8192, iDebugPrefixPath = "") :
  #--- Debug only
  tempMesh=o3d.io.read_triangle_mesh(currentTest.meshPath)
  nbInputVertices=len(tempMesh.vertices)
  nbInputTriangles=len(tempMesh.triangles)
  iKPILineToFill.append(nbInputVertices)
  iKPILineToFill.append(nbInputTriangles)
  iKPILineToFill.append(currentAreaToRemesh[0])

  #--- Generate submesh to complete
  submesh, remeshBoundary, sampledPoints=GenerateMeshToCompleteFromPath(currentTest.meshPath, currentAreaToRemesh, currentRingSize, iNbSamples=iNbSamples, iDebugPrefixPath=iDebugPrefixPath)
  nbRemovedTriangles=nbInputTriangles-len(submesh.triangles)
  iKPILineToFill.append(nbRemovedTriangles)
  iKPILineToFill.append(trialNumber)

  #--- Set up model
  transformer = TreeMeshGPT(quant_bit = 7 if VERSION == "7bit" else 9, max_seq_len=13000) # can set higher max_seq_len if GPU is L4 or A100
  transformer.load(CKPT_PATH)
  accelerator = Accelerator(mixed_precision="fp16")
  transformer = accelerator.prepare(transformer)

  #--- Complete submesh
  startTime=time.time()
  completedMesh=CompleteMesh(transformer, accelerator, submesh, remeshBoundary, sampledPoints, iDebugPrefixPath)
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
    tempHEMesh=o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(completedMesh)
    
    nbFreeEdges=0
    for he in tempHEMesh.half_edges :
      if he.twin==-1 : nbFreeEdges+=1
    
    iKPILineToFill.append(nbFreeEdges)

  #Manifoldness
  iKPILineToFill.append(True if completedMesh.is_edge_manifold() else False)

  #return iKPILineToFill
  ioQueue.put(iKPILineToFill)


#--- Run tests
for currentTest in allCompletionsTests :
  print(f"Processing mesh : {currentTest.meshPath}", flush=True)
  print(f"Processing mesh : {currentTest.meshPath}", flush=True)
  
  #Debug : Extract mesh name
  meshName, _=os.path.splitext(os.path.basename(currentTest.meshPath))
  debugPrefixPathBase="./output/"+meshName

  runCounter=0

  for currentAreaToRemesh in currentTest.listsOfTrianglesToRemesh :
    for currentRingSize in currentTest.listOfRingSizes :
      for trialNumber in range(currentTest.nbRunsPerTrial) :

        runCounter+=1
        debugPrefixPath=debugPrefixPathBase+"_run"+f"{runCounter:2}"

        kpiLine=[]
        kpiLine.append(runCounter)
        kpiLine.append(meshName)

        if USE_MULTIPROCESS :
          # with ThreadPoolExecutor(max_workers=1) as executor:
          #   future = executor.submit(PerformCompletion, meshName, currentAreaToRemesh, currentRingSize, kpiLine, NB_SAMPLING_POINTS, debugPrefixPath)
          #   result=future.result()
          #   if isinstance(result, list) :
          #     kpiLine=result
          #   else :
          #     kpiLineSize=len(kpiLine)
          #     for i in range(len(df.columns)-kpiLineSize) :
          #       kpiLine.append("NA")

          subProcessQueue=Queue()
          subProcess=Process(target=PerformCompletion, args=(subProcessQueue, meshName, currentAreaToRemesh, currentRingSize, kpiLine, NB_SAMPLING_POINTS, debugPrefixPath,))
          subProcess.start()
          kpiLine=subProcessQueue.get()
          subProcess.join()

        else :
          subProcessQueue=Queue()
          PerformCompletion(meshName, currentAreaToRemesh, currentRingSize, kpiLine, NB_SAMPLING_POINTS, debugPrefixPath)
          kpiLine=subProcessQueue.get()
        
        #Fill dataframe
        df.loc[len(df)] = kpiLine
        print("## ", *kpiLine, sep="\t")
        print("\n\n")


#Save dataframe in a csv file named after date and time
csvFileName="./output/KPI_Completion_"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".csv"
df.to_csv(csvFileName, index=False)


        
        


