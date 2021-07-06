TPB = 8         #Threads per block, should be 8
BUFFER = 3     #Number of empty layers around the object
RESOLUTION = 400    #Maximum number of voxels in the X/Y directions
#Gradient function sensitivity (Higher value = more gradient voxels)
GTHRESH = 0.85
#SDF Skeletal Sensitivity (Higher magnitude = more skeletal voxels)
STHRESH = 2
#Point Distribution Weighting Value (Higher value = more voronoi cells)
PDIST = 2.0
#Voronoi Cell Thickness (In voxels, higher value = thicker cell walls)
WALL_THICKNESS = 3.0
#Shell Thickness (In voxels, higher value = thicker outer wall/more perimeters)
SHELL_THICKNESS = 3.0

PLOTTING = True     #Print plots as it goes through the skeletizing process
SAVE_PLOTS = True   #Save the afore-printed plots to the Output folder

#Put the name of the desired file below, or uncomment one of the example files
#This file must be in the Input folder, set to be in the same directory as the
#python files.

#FILE_NAME = "Bird.stl"
#FILE_NAME = "E.stl"
#FILE_NAME = "3DBenchy.stl"
#FILE_NAME = "bust_low.stl"
#FILE_NAME = "wavySurface.stl"
#FILE_NAME = "hand_low.stl"

#If you would prefer a simple geometric object, uncomment the one you want and
#make sure that all FILE_NAME options are commented out.
#PRIMITIVE_TYPE = "Heart"
#PRIMITIVE_TYPE = "Egg"
#PRIMITIVE_TYPE = "Cube"
#PRIMITIVE_TYPE = "Sheet"
PRIMITIVE_TYPE = "Box"
#PRIMITIVE_TYPE = "Sphere"
#PRIMITIVE_TYPE = "Cylinder"
#PRIMITIVE_TYPE = "Silo"