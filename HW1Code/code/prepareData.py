import numpy as np

def prepareData(imArray, ambientImage):
    
    h, w, c = imArray.shape
    data = np.zeros((h, w, c))
    
    #removing the affect of ambient lighting by linearity of light
    for i in range(c):
        data[:,:,i] = imArray[:,:,i] - ambientImage
        #renormalize images
        data[data < 0] = 0
    data = data / np.max(imArray)
    
    return data
