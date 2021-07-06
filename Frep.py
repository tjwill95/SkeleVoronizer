from numba import cuda
import numpy as np
import math
import userInput as u
try: TPB = u.TPB 
except: TPB = 8

@cuda.jit
def halfSpaceKernel(d_u,h):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    if i >= dims[0] or j >= dims[1] or k >= dims[2]:
        return
    if k>h:
        d_u[i,j,k]=1
    else:
        d_u[i,j,k]=-1000
        
def halfSpace(u,h,inv=False):
    #Defines a half-space of u, where all values below a Z of h are negative
    #and all values above a Z of h are positive.
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    dims = u.shape
    d_u = cuda.to_device(u)
    gridDims = (dims[0]+TPBX-1)//TPBX, (dims[1]+TPBY-1)//TPBY, (dims[2]+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    halfSpaceKernel[gridDims, blockDims](d_u,h)
    u = d_u.copy_to_host()
    if inv: u = -u
    return u

@cuda.jit
def smoothKernel(d_u, d_v, buffer):
    i,j,k = cuda.grid(3)
    count = 0
    dims = d_u.shape
    if i>=dims[0]-buffer or j>=dims[1]-buffer or k>=dims[2]-buffer or min(i,j,k)<buffer:
        return
    else:
        d_v[i,j,k]=0
        for index in range(27):
            checkPos = (i+((index//9)%3-1),j+((index//3)%3-1),k+(index%3-1))
            if checkPos[0]<dims[0] and checkPos[1]<dims[1] and checkPos[2]<dims[2] and min(checkPos)>=0:
                d_v[i,j,k]+=d_u[checkPos]
                count+=1
        d_v[i,j,k] = d_v[i,j,k]/count

def smooth(u,iteration = 1,buffer=1):
    #u = input voxel model
    #iteration = number of times to run the algorithm
    #buffer = Layers of voxels on the boundaries of the box that are left untouched
    #Outputs a new matrix with each value set to the average of its neighbor's values.
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    dims = u.shape
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(u)
    gridDims = (dims[0]+TPBX-1)//TPBX, (dims[1]+TPBY-1)//TPBY, (dims[2]+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    for var in range(iteration):
        smoothKernel[gridDims, blockDims](d_u, d_v,buffer)
        d_u,d_v = d_v,d_u
    return d_u.copy_to_host()

@cuda.jit
def boolKernel(d_u,d_v):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    if i >= dims[0] or j >= dims[1] or k >= dims[2]:
        return
    d_u[i,j,k] = min(d_u[i,j,k],d_v[i,j,k])
    
def union(u,v):
    #u,v = voxel models that you want to union
    #Outputs the union of the models (fills in the resulting matrix such that 
    #if a cell is negative in either u or v, it is negative in the output).
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(v)
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    boolKernel[gridSize, blockSize](d_u,d_v)
    return d_u.copy_to_host()

def intersection(u,v):
    #u,v = voxel models that you want to intersect
    #Outputs the intersection of the models (fills in the resulting matrix such
    #that if a cell is positive in either u or v, it is positive in the output).
    d_u = cuda.to_device(-1*u)
    d_v = cuda.to_device(-1*v)
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    boolKernel[gridSize, blockSize](d_u,d_v)
    return -1*d_u.copy_to_host()

def subtract(u,v):
    #u = cutting tool model (Model that's removed)
    #v = base model
    #Outputs the subtraction of the models (fills in the resulting matrix such
    #that if a cell is negative in u, it's positive in the output)
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(-1*v)
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    boolKernel[gridSize, blockSize](d_u,d_v)
    return -1*d_u.copy_to_host()

@cuda.jit
def projectionKernel(d_u,X):
    j,k = cuda.grid(2)
    m,n,p = d_u.shape
    if j < n and k < p:
        if d_u[X+1,j,k]<=0:
            d_u[X,j,k]=-1    

def projection(u):
    #u = voxelized model, negative = internal
    #Assumes X is the vertical axis, projects entire part to lowest X value.
    TPBY, TPBZ = TPB, TPB
    m, n, p = u.shape
    minX = -1
    i = 0
    while minX<0:
        if np.amin(u[i,:,:])<0:
            minX = i
        else:
            i += 1
    X = m-1
    d_u = cuda.to_device(u)
    gridDims = (n+TPBY-1)//TPBY, (p+TPBZ-1)//TPBZ
    blockDims = TPBY, TPBZ
    while X>minX:
        X -= 1
        projectionKernel[gridDims, blockDims](d_u,X)
    return d_u.copy_to_host()

@cuda.jit
def translateKernel(d_u,d_v,x,y,z):
    i,j,k = cuda.grid(3)
    m,n,p = d_u.shape
    if i >= m or j >= n or k >= p:
        return
    d_v[i,j,k] = d_u[(i-x)%m,(j-y)%n,(k-z)%p]
    
def translate(u,x,y,z):
    #u = voxel model to translate
    #x,y,z = translation vector, integers in voxels
    #moves the model according to the translation vector.
    d_u = cuda.to_device(u)
    d_v = cuda.device_array(shape = u.shape, dtype = np.float32)
    dims = u.shape
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    translateKernel[gridSize, blockSize](d_u,d_v,x,y,z)
    return d_v.copy_to_host()

def thicken(u,weight):
    #u = voxel model to thicken, assumes SDF
    #origShape = outer bounds of model
    #weight = how much we're thickening the object (In voxels)
    return u - np.ones(u.shape)*weight
    
def shell(uSDF,sT):
    #u = voxel model to shell, assumes SDF
    #sT = thickness of the shell (In voxels)
    return intersection(uSDF,-uSDF-np.ones(uSDF.shape)*sT)

@cuda.jit
def condenseKernel(d_u,d_uCondensed,xSpace,ySpace,zSpace,minX,minY,minZ):
    i,j,k = cuda.grid(3)
    um,un,up = d_u.shape
    m,n,p = d_uCondensed.shape
    newi = i+xSpace
    newj = j+ySpace
    newk = k+zSpace
    if i+minX<um and j+minY<un and k+minZ<up and newi<m and newj<n and newk<p:
        d_uCondensed[newi,newj,newk] = d_u[i+minX,j+minY,k+minZ]
    
def condense(u,buffer,minDim=0,cube=False):
    #u = input voxel model
    #buffer = number of layers of voxels around the boundaries that are left empty
    #minDim = minimum dimension for one sidelength of the matrix
    #Outputs a new matrix that is fitted to the input voxel model, removing layers
    #that don't store geometry and centering the model in the resulting matrix
    m, n, p = u.shape
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    minX, maxX, minY, maxY, minZ, maxZ = -1,-1,-1,-1,-1,-1
    i, j, k = 0, 0, 0
    while minX<0:
        if np.amin(u[i,:,:])<0:     minX = i
        else:                       i += 1
    while minY<0:
        if np.amin(u[:,j,:])<0:     minY = j
        else:                       j += 1
    while minZ<0:
        if np.amin(u[:,:,k])<0:     minZ = k
        else:                       k += 1
    i, j, k = 1, 1, 1
    while maxX<0:
        if np.amin(u[m-i,:,:])<0:   maxX = m-i
        else:                       i += 1
    while maxY<0:
        if np.amin(u[:,n-j,:])<0:   maxY = n-j
        else:                       j += 1
    while maxZ<0:
        if np.amin(u[:,:,p-k])<0:   maxZ = p-k
        else:                       k += 1
    xSize = max(2*buffer+maxX-minX+TPB-1,minDim+TPB-1)//TPB*TPB
    ySize = max(2*buffer+maxY-minY+TPB-1,minDim+TPB-1)//TPB*TPB
    zSize = max(2*buffer+maxZ-minZ+TPB-1,minDim+TPB-1)//TPB*TPB
    if cube: 
        size=max(xSize,ySize,zSize)
        xSize,ySize,zSize = size, size, size
    d_u = cuda.to_device(u)
    d_uCondensed = cuda.to_device(np.ones([xSize,ySize,zSize]))
    gridDims = (m+TPBX-1)//TPBX, (n+TPBY-1)//TPBY, (p+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    xSpace, ySpace, zSpace = (xSize-maxX+minX)//2,(ySize-maxY+minY)//2,(zSize-maxZ+minZ)//2
    condenseKernel[gridDims, blockDims](d_u, d_uCondensed,xSpace,ySpace,zSpace,minX,minY,minZ)
    return d_uCondensed.copy_to_host()

@cuda.jit
def heartKernel(d_u, d_x, d_y, d_z,cx,cy,cz):
    i,j,k = cuda.grid(3)
    m,n,p = d_u.shape
    if i < m and j < n and k < p:
        x = d_x[i]-cx
        y = d_y[j]-cy
        z = d_z[k]-cz
        d_u[i,j,k] = (x**2+9*(y**2)/4+z**2-1)**3-(x**2)*(z**3)-9*(y**2)*(z**3)/80
        
def heart(x,y,z,cx,cy,cz):
    #x,y,z = coordinate domain that we want the shape to live in, vectors
    #cx,cy,cz = coordinates of the center of the heart shape.
    #Outputs a 3D matrix with negative values showing the inside of our shape, 
    #positive values showing the outside, and 0s to show the surfaces.
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    m = x.shape[0]
    n = y.shape[0]
    p = z.shape[0]
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_z = cuda.to_device(z)
    d_u = cuda.device_array(shape = [m, n, p], dtype = np.float32)
    gridDims = (m+TPBX-1)//TPBX, (n+TPBY-1)//TPBY, (n+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    heartKernel[gridDims, blockDims](d_u, d_x, d_y, d_z,cx,cy,cz)
    return d_u.copy_to_host()

@cuda.jit
def eggKernel(d_u, d_x, d_y, d_z,cx,cy,cz):
    i,j,k = cuda.grid(3)
    m,n,p = d_u.shape
    if i < m and j < n and k < p:
        x = d_x[i]-cx
        y = d_y[j]-cy
        z = d_z[k]-cz
        d_u[i,j,k] = 9*x**2+16*(y**2+z**2)+2*x*(y**2+z**2)+(y**2+z**2)-144
        
def egg(x,y,z,cx,cy,cz):
    #x,y,z = coordinate domain that we want the shape to live in, vectors
    #cx,cy,cz = coordinates of the center of the egg shape.
    #Outputs a 3D matrix with negative values showing the inside of our shape, 
    #positive values showing the outside, and 0s to show the surfaces.
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    m = x.shape[0]
    n = y.shape[0]
    p = z.shape[0]
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_z = cuda.to_device(z)
    d_u = cuda.device_array(shape = [m, n, p], dtype = np.float32)
    gridDims = (m+TPBX-1)//TPBX, (n+TPBY-1)//TPBY, (n+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    eggKernel[gridDims, blockDims](d_u, d_x, d_y, d_z,cx,cy,cz)
    return d_u.copy_to_host()

@cuda.jit
def rectKernel(d_u, d_x, d_y, d_z, xl, yl, zl, origin):
    i,j,k = cuda.grid(3)
    m,n,p = d_u.shape
    if i < m and j < n and k < p:
        sx = abs(d_x[i]-origin[0]) - xl/2
        sy = abs(d_y[j]-origin[1]) - yl/2
        sz = abs(d_z[k]-origin[2]) - zl/2
        d_u[i,j,k]=max(sx,sy,sz)
        
def rect(x,y,z,xl,yl,zl,origin = [0,0,0]):
    #x,y,z = coordinate domain that we want the shape to live in, vectors
    #xl,yl,zl = sidelengths of the rectangular prism.
    #origin = coordinates for the center of the prism
    #Outputs a 3D matrix with negative values showing the inside of our shape, 
    #positive values showing the outside, and 0s to show the surfaces.
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    m = x.shape[0]
    n = y.shape[0]
    p = z.shape[0]
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_z = cuda.to_device(z)
    d_origin = cuda.to_device(origin)
    d_u = cuda.device_array(shape = [m, n, p], dtype = np.float32)
    gridDims = (m+TPBX-1)//TPBX, (n+TPBY-1)//TPBY, (n+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    rectKernel[gridDims, blockDims](d_u, d_x, d_y, d_z, xl, yl, zl, d_origin)
    return d_u.copy_to_host()

@cuda.jit
def sphereKernel(d_u, d_x, d_y, d_z, rad):
    i,j,k = cuda.grid(3)
    m,n,p = d_u.shape
    if i < m and j < n and k < p:
        d_u[i,j, k] = math.sqrt(d_x[i]**2+d_y[j]**2+d_z[k]**2)-rad
        
def sphere(x,y,z,rad):
    #x,y,z = x,y,z coordinate domain that we want the shape to live in, centered on [0,0,0]
    #rad = radius of the sphere.
    #Outputs a 3D matrix with negative values showing the inside of our shape, 
    #positive values showing the outside, and 0s to show the surfaces.
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    m = x.shape[0]
    n = y.shape[0]
    p = z.shape[0]
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_z = cuda.to_device(z)
    d_u = cuda.device_array(shape = [m, n, p], dtype = np.float32)
    gridDims = (m+TPBX-1)//TPBX, (n+TPBY-1)//TPBY, (n+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    sphereKernel[gridDims, blockDims](d_u, d_x, d_y, d_z, rad)
    return d_u.copy_to_host()

@cuda.jit
def cylinderXKernel(d_u, d_x, d_y, d_z, start, stop, rad):
    i,j,k = cuda.grid(3)
    m,n,p = d_u.shape
    if i < m and j < n and k < p:
        height = (d_x[i]-start)*(d_x[i]-stop)
        width = math.sqrt(d_y[j]**2+d_z[k]**2)-rad
        d_u[i,j,k] = max(height,width)

def cylinderX(x,y,z,start,stop,rad):
    #x,y,z = x,y,z coordinate domain that we want the shape to live in.
    #start, stop = highest and lowest X coordinates (Order irrelevant).
    #rad = radius of the cylinder.
    #Outputs a 3D matrix with negative values showing the inside of our shape, 
    #positive values showing the outside, and 0s to show the surfaces.
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    m = x.shape[0]
    n = y.shape[0]
    p = z.shape[0]
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_z = cuda.to_device(z)
    d_u = cuda.device_array(shape = [m, n, p], dtype = np.float32)
    gridDims = (m+TPBX-1)//TPBX, (n+TPBY-1)//TPBY, (n+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    cylinderXKernel[gridDims, blockDims](d_u, d_x, d_y, d_z, start, stop, rad)
    return d_u.copy_to_host()

@cuda.jit
def cylinderYKernel(d_u, d_x, d_y, d_z, start, stop, rad):
    i,j,k = cuda.grid(3)
    m,n,p = d_u.shape
    if i < m and j < n and k < p:
        height = (d_y[j]-start)*(d_y[j]-stop)
        width = math.sqrt(d_x[i]**2+d_z[k]**2)-rad
        d_u[i,j,k] = max(height,width)

def cylinderY(x,y,z,start,stop,rad):
    #x,y,z = x,y,z coordinate domain that we want the shape to live in.
    #start, stop = highest and lowest Y coordinates (Order irrelevant).
    #rad = radius of the cylinder.
    #Outputs a 3D matrix with negative values showing the inside of our shape, 
    #positive values showing the outside, and 0s to show the surfaces.
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    m = x.shape[0]
    n = y.shape[0]
    p = z.shape[0]
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_z = cuda.to_device(z)
    d_u = cuda.device_array(shape = [m, n, p], dtype = np.float32)
    gridDims = (m+TPBX-1)//TPBX, (n+TPBY-1)//TPBY, (n+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    cylinderYKernel[gridDims, blockDims](d_u, d_x, d_y, d_z, start, stop, rad)
    return d_u.copy_to_host()