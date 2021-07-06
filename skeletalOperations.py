from numba import cuda
import Frep as f
import math
import numpy as np
import userInput as u
try: TPB = u.TPB 
except: TPB = 8

@cuda.jit
def addKernel(d_u,t):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    if d_u[i,j,k]<0:
            d_u[i,j,k] +=t

def thickenSkeletonAdd(u,t):
    #u = input skeleton
    #t = added constant
    #Outputs the skeleton after each skeletal voxel's value has had t added to
    #it.
    dims = u.shape
    d_u = cuda.to_device(u)
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    addKernel[gridSize, blockSize](d_u,t)
    return d_u.copy_to_host()

@cuda.jit
def multKernel(d_u,t):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    if i>=dims[0] or j>=dims[1] or k>=dims[2]:
        return
    if d_u[i,j,k]<0:
            d_u[i,j,k] *=t

def thickenSkeletonMult(u,t):
    #u = input skeleton
    #t = multiplied constant
    #Outputs the skeleton after each skeletal voxel's value has been multiplied
    #by t.
    dims = u.shape
    d_u = cuda.to_device(u)
    gridSize = [(dims[0]+TPB-1)//TPB, (dims[1]+TPB-1)//TPB,(dims[2]+TPB-1)//TPB]
    blockSize = [TPB, TPB, TPB]
    multKernel[gridSize, blockSize](d_u,t)
    return d_u.copy_to_host()

@cuda.jit
def softenKernel(d_u,d_v,thresh):
    i,j,k = cuda.grid(3)
    dims = d_v.shape
    if i >= dims[0] or j >= dims[1] or k >= dims[2]:
        return
    if d_u[i,j,k]>-thresh:
        d_v[i,j,k]=1

def soften(u,thresh):
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    dims = u.shape
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(u)
    gridDims = (dims[0]+TPBX-1)//TPBX, (dims[1]+TPBY-1)//TPBY, (dims[2]+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    softenKernel[gridDims, blockDims](d_u,d_v,thresh)
    return d_v.copy_to_host()
    
def normalize(v):
    return v/np.linalg.norm(v)

@cuda.jit    
def sharpStretchKernel(d_u,d_v,cutStart,distance):
    i,j,k = cuda.grid(3)
    dims = d_v.shape
    udims = d_u.shape
    if i >= dims[0] or j >= dims[1] or k >= dims[2] or i-distance>=udims[0]:
        return
    if i<cutStart:
        d_v[i,j,k]=d_u[i,j,k]
    elif i<cutStart+distance:
        d_v[i,j,k]=d_u[cutStart,j,k]
    else:
        d_v[i,j,k]=d_u[i-distance,j,k]

@cuda.jit
def conStretchKernel(d_u, d_v,cutStart,cutEnd,distance):
    i,j,k = cuda.grid(3)
    dims = d_v.shape
    udims = d_u.shape
    if i >= dims[0] or j >= dims[1] or k >= dims[2] or i-distance+cutEnd-cutStart>=udims[0]:
        return
    if i<cutStart:
        d_v[i,j,k]=d_u[i,j,k]
    elif i<cutStart+distance:
        inew = int((i-cutStart)*(cutEnd-cutStart)/distance+cutStart+0.5)
        d_v[i,j,k]=d_u[inew,j,k]
    else:
        d_v[i,j,k]=d_u[i-distance+cutEnd-cutStart,j,k]
        
@cuda.jit
def tanStretchKernel(d_u, d_v,cutStart,cutEnd,distance):
    i,j,k = cuda.grid(3)
    dims = d_v.shape
    udims = d_u.shape
    if i >= dims[0] or j >= dims[1] or k >= dims[2] or i-distance+cutEnd-cutStart>=udims[0]:
        return
    if i<cutStart:
        d_v[i,j,k]=d_u[i,j,k]
    elif i<cutStart+distance:
        L1 = cutEnd-cutStart
        L2 = distance
        inew = int(i+(L1-L2)*(1-math.cos((i-cutStart)*math.pi/L2))/2)
        d_v[i,j,k]=d_u[inew,j,k]
    else:
        d_v[i,j,k]=d_u[i-distance+cutEnd-cutStart,j,k]

def stretch(u,direction,cutStart,cutEnd,distance,boundaries="con"):
    #u = voxel model
    #direction = length 3 vector that defines direction
    #cutStart = Where the stretch starts (measured in the direction given)
    #cutEnd = Where the stretch ends (measured in the given direction)
    #distance = New length of stretched section
    #behavior = End conditions of the stretched sections, matches contact or tangent
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    aavect = findAxisAngle([1,0,0],direction)
    urot = axisAngleRot(u,aavect)
    dims = urot.shape
    v = np.ones([((dims[0]+distance+TPB)//TPB)*TPB,dims[1],dims[2]])
    vdims = v.shape
    d_u = cuda.to_device(urot)
    d_v = cuda.to_device(v)
    gridDims = (vdims[0]+TPBX-1)//TPBX, (vdims[1]+TPBY-1)//TPBY, (vdims[2]+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    if cutStart==cutEnd: sharpStretchKernel[gridDims, blockDims](d_u,d_v,cutStart,distance)
    elif boundaries == "con": conStretchKernel[gridDims, blockDims](d_u,d_v,cutStart,cutEnd,distance)
    elif boundaries == "tan": tanStretchKernel[gridDims, blockDims](d_u,d_v,cutStart,cutEnd,distance)
    else: print("Boundaries must be con or tan")
    v = d_v.copy_to_host()
    v = f.condense(v,2,cube=True)
    v = axisAngleRot(v,aavect,inv=True)
    return v

@cuda.jit
def transformKernel(d_u, d_v, d_r, d_c, d_t):
    i,j,k = cuda.grid(3)
    dims = d_u.shape
    if i >= dims[0] or j >= dims[1] or k >= dims[2]:
        return
    ci = i-d_c[0]
    cj = j-d_c[1]
    ck = k-d_c[2]
    m = int(d_r[0,0]*ci+d_r[0,1]*cj+d_r[0,2]*ck+d_c[0]+d_t[0]+0.5)
    n = int(d_r[1,0]*ci+d_r[1,1]*cj+d_r[1,2]*ck+d_c[1]+d_t[1]+0.5)
    p = int(d_r[2,0]*ci+d_r[2,1]*cj+d_r[2,2]*ck+d_c[2]+d_t[2]+0.5)
    if min(m,n,p)<0 or m>=dims[0] or n>=dims[1] or p>=dims[2] or d_u[m,n,p]>=0:
        return
    else:
        d_v[i,j,k]=d_u[m,n,p]

def transform(u,translation=[0,0,0],rotation=[0,0,0],center=[0,0,0],units="deg",inv=False):
    #u = voxel model
    #translation = translation of model.  If stationary, leave as [0,0,0].  Measured in voxels
    #rotation = rotation of model about center point, measured in roll, pitch, yaw.
    #center = centerpoint of rotation, if [0,0,0] it is rotated about the midpoint of the matrix, measured in voxels
    #units = angular units, defaults to degrees.
    #inv = boolean. If true, then it inverts the rotation matrix to undo the transformation
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    dims = u.shape
    rotation = np.array(rotation)
    translation = np.array(translation)
    center = np.array(center)
    if units == "deg": rotation = rotation*np.pi/180
    elif units == "rad": rotation = rotation
    else: 
        print("Unit selection invalid, input string rad or deg")
        return
    su,sv,sw = np.sin(rotation)
    cu,cv,cw = np.cos(rotation)
    Rx = np.array([[1,0,0],[0,cu,-su],[0,su,cu]])
    Ry = np.array([[cv,0,sv],[0,1,0],[-sv,0,cv]])
    Rz = np.array([[cw,-sw,0],[sw,cw,0],[0,0,1]])
    R = np.matmul(np.matmul(Rx,Ry),Rz)
    if not inv: R, translation = np.transpose(R), -translation
    if max(center)<1:
        center = np.array([dims[0]//2,dims[1]//2,dims[2]//2])
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(np.ones(dims,np.float32))
    d_r = cuda.to_device(R)
    d_c = cuda.to_device(center)
    d_t = cuda.to_device(translation)
    gridDims = (dims[0]+TPBX-1)//TPBX, (dims[1]+TPBY-1)//TPBY, (dims[2]+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    transformKernel[gridDims, blockDims](d_u, d_v, d_r, d_c, d_t)
    return d_v.copy_to_host()

def findAxisAngle(v1,v2):
    #v1 = starting orientation
    #v2 = ending orientation
    #Ouptuts a vector where the first three indices are the axis of rotation
    #and the last index is the angle of rotation (in degrees)
    v1 = normalize(v1)
    v2 = normalize(v2)
    if np.allclose(v1,v2):
        v=[1,0,0]
    else:
        v = normalize(np.cross(v1,v2))
    theta = math.acos(np.dot(v1,v2))*180/np.pi
    return np.append(v,theta)

def axisAngleRot(u,aavect,center=[0,0,0],units="deg",inv=False):
    #u = input voxel model
    #aavect = [kx,ky,kz,theta]
    #center = where the model should be rotated about
    #units = angular units, defaults to degrees.
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    dims = u.shape
    kx,ky,kz,theta = aavect
    if units == "deg": theta = theta*np.pi/180
    elif units == "rad": theta = theta
    else: 
        print("Unit selection invalid, input string rad or deg")
        return
    if max(center)<1:
        center = np.array([dims[0]//2,dims[1]//2,dims[2]//2])
    st = np.sin(theta)
    ct = np.cos(theta)
    R = np.array([[ct+kx**2*(1-ct),kx*ky*(1-ct)-kz*st,kx*kz*(1-ct)+ky*st],
                  [ky*kx*(1-ct)+kz*st,ct+ky**2*(1-ct),ky*kz*(1-ct)-kx*st], 
                  [kz*kx*(1-ct)-ky*st,ky*kz*(1-ct)+kx*st,ct+kz**2*(1-ct)]])
    if not inv: R = np.transpose(R)
    d_u = cuda.to_device(u)
    d_v = cuda.to_device(np.ones(dims,np.float32))
    d_r = cuda.to_device(R)
    d_c = cuda.to_device(center)
    d_t = cuda.to_device(np.array([0,0,0]))
    gridDims = (dims[0]+TPBX-1)//TPBX, (dims[1]+TPBY-1)//TPBY, (dims[2]+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    transformKernel[gridDims, blockDims](d_u, d_v, d_r, d_c, d_t)
    return d_v.copy_to_host()

@cuda.jit
def circKernel(d_rp,cy,cz,y,theta):
    i,j,k = cuda.grid(3)
    dims = d_rp.shape
    if i > dims[0] or j > dims[1] or k > dims[2]:
        return
    elif int(math.sqrt(float((j-cy)**2+(k-cz)**2))+0.5)==int(y-cy) and -math.atan2(float(k-cz),-float(j-cy))+math.pi<=theta and d_rp[i,y,cz]<0:
        d_rp[i,j,k]= d_rp[i,y,cz]
        
def revolve(u,axisHeight,axisDepth,theta):
    #Revolves the planar geometry at the slice given at axisHeight (here Z) 
    #around an axis parallel with the X axis at axisDepth (y value) by an angle
    #theta
    TPBX, TPBY, TPBZ = TPB, TPB, TPB
    dims = u.shape
    v = f.halfSpace(u,axisHeight)
    newU =f.intersection(u,v)
    w = f.halfSpace(u,axisHeight-1,inv=True)
    revProfile = f.intersection(newU,w)
    gridDims = (dims[0]+TPBX-1)//TPBX,(dims[1]+TPBY-1)//TPBY, (dims[2]+TPBZ-1)//TPBZ
    blockDims = TPBX, TPBY, TPBZ
    theta = theta*np.pi/180 #Converts to radians
    d_rp = cuda.to_device(revProfile)
    for j in range(axisDepth,dims[1]):
        circKernel[gridDims, blockDims](d_rp,axisDepth,axisHeight,j,theta)
    return f.union(newU,d_rp.copy_to_host())

def bend(u,axisHeight,axisDepth,theta):
    bottomAndCorner = revolve(u,axisHeight,axisDepth,theta)
    top = f.halfSpace(u,axisHeight-1,inv=True)
    top = f.intersection(top,u)
    top = transform(top,rotation=[theta,0,0],center=[0,axisDepth,axisHeight])
    return f.union(top,bottomAndCorner)   