from numba import cuda
from grad3D import grad3D
import numpy as np
import math
import userInput as u
from visualizeSlice import contourPlot
try: TPB = u.TPB 
except: TPB = 8

@cuda.jit(device = True)
def distance(i,j,k,m,n,p):
    return math.sqrt((i-m)*(i-m)+(j-n)*(j-n)+(k-p)*(k-p))

@cuda.jit
def skeletonKernel(d_u,d_uGrad,gThresh,sThresh):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    if d_u[i,j,k]>sThresh or d_uGrad[i,j,k]>gThresh:
            d_u[i,j,k] = 1

def skeleton(u,gThresh=0.85,sThresh=1):
    #u = input voxel model, must be SDF
    #gThresh = gradient sensitivity, higher value = more skeletal voxels
    #sThresh = Distance sensitivity, higher value = fewer skeletal voxels
    #Outputs a matrix of the same shape as u, with negative voxels representing
    #the skeleton of u, with their magnitudes representing the radial
    #information needed to recreate the object.
    dims = u.shape
    d_u = cuda.to_device(u)
    uGrad = grad3D(u, maxVal = 1)
    d_uGrad = cuda.to_device(uGrad)
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    skeletonKernel[gridSize, blockSize](d_u,d_uGrad,gThresh,-sThresh)
    return d_u.copy_to_host()

@cuda.jit
def refleshKernel(d_r,d_w,template):
    i,j,k = cuda.grid(3)
    dims = d_r.shape
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    lowVal = d_w[i,j,k]
    for index in range(template**3):
        ni = int(i+((index//(template**2))%template-template//2))
        nj = int(j+((index//template)%template-template//2))
        nk = int(k+(index%template-template//2))
        if ni<dims[0] and nj<dims[1] and nk<dims[2] and min(ni,nj,nk)>=0:
            updated = d_r[ni,nj,nk]+((i-ni)**2+(j-nj)**2+(k-nk)**2)**(1/2)
            lowVal=min(lowVal,updated)
    d_w[i,j,k]=lowVal

def reflesh(skeleton,iteration = 0,template = 5):
    #skeleton = skeletal data of the object
    #iteration = number of passes taken by the algorithm
    #template = side length of the cube of checked values, must be odd number
    #Outputs a matrix of the same size as skeleton that represents the object
    #created by refleshing the skeleton.
    dims = skeleton.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    d_r = cuda.to_device(skeleton)
    d_w = cuda.to_device(skeleton)
    if iteration==0:
        iteration = int(max(dims)//template)
    for count in range(iteration):
        refleshKernel[gridSize, blockSize](d_r,d_w,template)
        d_r,d_w = d_w,d_r
    return d_r.copy_to_host()

@cuda.jit
def SWSetupKernel(d_u,d_r):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    if d_u[i,j,k]<0.0:
        d_r[i,j,k,:]=float(i),float(j),float(k),-float(d_u[i,j,k])
    else:
        d_r[i,j,k,:]=-1.0,-1.0,-1.0,0.0

@cuda.jit
def SWKernel(d_r,d_w,template):
    i,j,k = cuda.grid(3)
    dims = d_r.shape
    if i>=dims[0] or j>=dims[1] or k>=dims[2]: return
    cSX = d_r[i,j,k,0]
    cSY = d_r[i,j,k,1]
    cSZ = d_r[i,j,k,2]
    cSV = d_r[i,j,k,3]
    
    for index in range(template**3):
        ni = int(i+((index//(template**2))%template-template//2))
        nj = int(j+((index//template)%template-template//2))
        nk = int(k+(index%template-template//2))
        if ni<dims[0] and nj<dims[1] and nk<dims[2] and min(ni,nj,nk)>=0 and d_r[ni,nj,nk,3]>0:
            newSX = d_r[ni,nj,nk,0]
            newSY = d_r[ni,nj,nk,1]
            newSZ = d_r[ni,nj,nk,2]
            newSV = d_r[ni,nj,nk,3]
            newDist = distance(i,j,k,newSX,newSY,newSZ)
            if newDist<=newSV and newSV>cSV:
                cSX, cSY, cSZ, cSV = newSX, newSY, newSZ, newSV
    d_w[i,j,k,:]=cSX,cSY,cSZ,cSV

@cuda.jit
def unpackKernel(d_r,d_w):
    i,j,k = cuda.grid(3)
    dims = d_w.shape
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    if d_r[i,j,k,3]==0.0:
        d_w[i,j,k]=1.0
    else:
        d_w[i,j,k]=-d_r[i,j,k,3]

def skeletalWeight(skeleton,iteration = 0,template = 5):
    #skeleton = skeletal data of the object
    #iteration = number of passes taken by the algorithm
    #template = side length of the cube of checked values, must be an odd int
    #Outputs a matrix of the same size as skeleton that represents the object
    #created by refleshing the skeleton.
    dims = skeleton.shape
    skeleMin = np.amin(skeleton)
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    d_u = cuda.to_device(skeleton)
    d_r = cuda.to_device(np.ones([dims[0],dims[1],dims[2],4],np.float32))
    SWSetupKernel[gridSize, blockSize](d_u,d_r)
    d_w = cuda.to_device(1000*np.ones([dims[0],dims[1],dims[2],4],np.float32))
    if iteration==0:
        iteration = int(-skeleMin//2)+1
    for count in range(iteration):
        SWKernel[gridSize, blockSize](d_r,d_w,template)
        d_r,d_w = d_w,d_r
    unpackKernel[gridSize,blockSize](d_r,d_u)
    return d_u.copy_to_host()