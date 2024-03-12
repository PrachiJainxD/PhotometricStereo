import numpy as np

# importing library to obtain the least-squares solution of a linear system
from scipy.linalg import lstsq

def photometricStereo(imarray, lightdirs):
    h, w, c = imarray.shape
    reshape_array = np.zeros((c, h * w))
    for i in range(c):
        reshape_array[i,:] = np.reshape(imarray[:,:,i],[1,h*w])
    
    #Setting up least-squares solution of a linear system with the assumption of using a perfectly Lambertian surface
    gxy = lstsq(lightdirs, reshape_array)
    
    gxy = np.array(gxy[0])

    albedo1 = np.zeros((1,h * w))
    normal1 = np.zeros((3,h * w))
    
    #computing albedo and normals 
    for i in range(h * w):
        albedo1[0, i] = np.linalg.norm(gxy[:,i])
        normal1[:, i] = gxy[:,i]/np.linalg.norm(gxy[:,i])
        
    #resizing albedo and normals 
    albedo2 = np.reshape(albedo1,(h, w))
    normal2 = np.dstack([np.reshape(normal1[0,:], (h, w))[...,np.newaxis], np.reshape(normal1[1,:],(h, w))[...,np.newaxis], np.reshape(normal1[2,:],(h, w))[...,np.newaxis]]
                    )
    return albedo2, normal2