# This code is part of:
# 
#   COMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
# 
# Evaluation code for photometric stereo
# 
# Your goal is to implement the three functions prepareData(), 
# photometricStereo() and getSurface() to estimate the albedo and shape of
# the objects in the scene from multiple images. 
# 
# Start with setting subjectName='debug' which sets up a toy scene with
# known albedo and height which you can compare against. After you have a
# good implementation of this part, set the subjectName='yaleB01', etc. to
# run your code against real images of people. 
# 
# Credits: The homework is adapted from a similar one developed by
# Shvetlana Lazebnik (UNC/UIUC)


import os
import time
import numpy as np
import matplotlib.pyplot as plt 
import skimage.io as io

from utils import *
from getSurface import *
from photometricStereo import *
from loadFaceImages import *
from toyExample import *
from prepareData import *
from displayOutput import *
from displaySurfaceNormals import *

#library for colorbar ticker
from matplotlib import ticker
#library for testing
from toyExample import toyExample

#subjectName = debug, yaleB01, yaleB02, yaleB05, yaleB07
for subjectName in ['yaleB01', 'yaleB02', 'yaleB05', 'yaleB07']: 
    numImages = 128
    data_dir = os.path.join('..', 'data')
    out_dir = os.path.join('..', 'output', 'photometricStereo')
    image_dir = os.path.join(data_dir, 'photometricStereo', subjectName)
    integrationMethod = 'random' #column-row, row-column, average, random

    # Load images
    print('Loading images.')
    if subjectName == 'debug':
        imageSize = (64, 64) # Make this smaller to run your code faster for debugging
        (ambientImage, imArray, lightDirs, trueAlbedo, trueSurfaceNormals, trueHeightMap) = toyExample(imageSize, numImages)
    else:
        (ambientImage, imArray, lightDirs) = loadFaceImages(image_dir, subjectName, numImages)

    # Prepare data
    print('Prepaing data.')
    imArray = prepareData(imArray, ambientImage)

    # Estimate albedo and normals
    print('Estimating albedo and normals.')
    (albedoImage, surfaceNormals) = photometricStereo(imArray, lightDirs)

    # Estimate surface
    print('Estimating surface height map.')
    heightMap = getSurface(surfaceNormals, integrationMethod)

    # Display outputs in different view points
    print("Method={}".format(integrationMethod))    
    for view_point in [(-60,20),(90,90),(135,90),(180,90),(90,55),(135,55),(180,55),(90,20),(135,20),(180,20)]:
        azim, elev = view_point 
        print("View point: Azim_angle={}, Elev_angle={}".format(azim, elev))
        displayOutput(albedoImage, heightMap, azim_angle=azim, elev_angle=elev)    
    
    #displayOutput(albedoImage, heightMap)
    displaySurfaceNormals(surfaceNormals)

    # Display the true answer for debug
    if subjectName == 'debug':
        displayOutput(trueAlbedo, trueHeightMap)
        displaySurfaceNormals(trueSurfaceNormals)

    # Pause for input
    x = input('[Done] Press any key to quit.')
