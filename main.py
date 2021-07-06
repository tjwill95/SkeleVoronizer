import os #Just used to set up file directory
import userInput as u
import numpy as np
import Frep as f
from SDF3D import SDF3D
from skeleton import skeleton, skeletalWeight
from visualizeSlice import multiPlot, slicePlot
from meshExport import generateMesh
from voxelize import voxelize
from pointGen import genRandPoints
from voronize import voronize

def main():
    scale = [1,1,1]
    try: os.mkdir(os.path.join(os.path.dirname(__file__),'Output')) #Creates an output folder if there isn't one yet
    except: pass
    try:    FILE_NAME = u.FILE_NAME #Checks to see if a file name has been set
    except: FILE_NAME = ""
    try:    PRIMITIVE_TYPE = u.PRIMITIVE_TYPE #Checks to see if a primitive type has been set
    except: PRIMITIVE_TYPE = ""

    if FILE_NAME != "":
        #This section retrieves the input file and voxelizes the input STL
        shortName = FILE_NAME[:-4]
        filepath = os.path.join(os.path.dirname(__file__), 'Input',FILE_NAME)
        res = u.RESOLUTION-u.BUFFER*2
        origShape, objectBox = voxelize(filepath, res, u.BUFFER)
        gridResX, gridResY, gridResZ = origShape.shape
        scale[0] = objectBox[0]/(gridResX-u.BUFFER*2)
        scale[1] = max(objectBox[1:])/(gridResY-u.BUFFER*2)
        scale[2] = scale[1]
        print("Model Imported")
    elif PRIMITIVE_TYPE != "":
        #This section generates the desired primitive
        shortName = PRIMITIVE_TYPE
        if PRIMITIVE_TYPE == "Heart":
            x0 = np.linspace(-1.5,1.5,u.RESOLUTION)
            y0, z0 = x0, x0
            origShape = f.heart(x0,y0,z0,0,0,0)
        elif PRIMITIVE_TYPE == "Egg":
            x0 = np.linspace(-5,5,u.RESOLUTION)
            y0, z0 = x0, x0
            origShape = f.egg(x0,y0,z0,0,0,0)
            #eggknowledgement to Molly Carton for this feature.
        else:
            x0 = np.linspace(-50,50,u.RESOLUTION)
            y0, z0 = x0, x0
            if PRIMITIVE_TYPE == "Cube":
                origShape = f.rect(x0,y0,z0,80,80,80)
            elif PRIMITIVE_TYPE == "Box":
                origShape = f.rect(x0,y0,z0,80,60,40)
            elif PRIMITIVE_TYPE == "Sheet":
                origShape = f.rect(x0,y0,z0,1,80,60)
            elif PRIMITIVE_TYPE == "Silo":
                origShape = f.union(f.sphere(x0,y0,z0,40),f.cylinderY(x0,y0,z0,-40,0,40))
            elif PRIMITIVE_TYPE == "Cylinder":
                origShape = f.cylinderX(x0,y0,z0,-40,40,40)
            elif PRIMITIVE_TYPE == "Sphere":
                origShape = f.sphere(x0,y0,z0,40)
            else:
                print("Given primitive type has not yet been implemented.")
                return
        print("Primitive Generated")
    else:
        print("Provide either a file name or a desired primitive.")
        return

    initModel = origShape
    origShape = SDF3D(f.smooth(SDF3D(origShape)))
    oSkele = skeleton(origShape,gThresh = u.GTHRESH,sThresh = u.STHRESH)
    print("Skeleton Computed")
    if u.PLOTTING: multiPlot(oSkele,shortName+' Medial Surface',u.SAVE_PLOTS)
    weightField = skeletalWeight(oSkele)
    print("Weight Field Computed")
    if u.PLOTTING: multiPlot(weightField,shortName+' Weight Field',u.SAVE_PLOTS)
    skelePointField = genRandPoints(weightField, u.PDIST)
    voronoiModel = voronize(initModel,skelePointField,u.WALL_THICKNESS,u.SHELL_THICKNESS,name=shortName,smoothing=True)
    print("Voronoi Cells Generated")
    if u.PLOTTING: slicePlot(voronoiModel, voronoiModel.shape[2]//2, titlestring=shortName+' Medial Surface Voronoi Infill',axis = "Z")
    generateMesh(voronoiModel,scale,modelName = shortName+"MSVoronoiInfill")
    
if __name__ == '__main__':
    main()