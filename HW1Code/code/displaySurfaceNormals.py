import numpy as np
import matplotlib.pyplot as plt

#importing library to update tick in colorbar
from matplotlib import ticker

def displaySurfaceNormals(surfaceNormals):
    #Updated file to match output in 670: Computer Vision Homework Fall 2022
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    axlist = [ax1,ax2,ax3]
    
    first = ax1.imshow(surfaceNormals[:, :,0],  cmap ="jet", vmin =-1, vmax =1)
    ax1.set_title('X')
    ax1.axis('off') 
     
    second = ax2.imshow(surfaceNormals[:, :,1],  cmap ="jet", vmin =-1, vmax =1)
    ax2.set_title('Y')
    ax2.axis('off')           
   
    third = ax3.imshow(surfaceNormals[:, :,2],  cmap ="jet", vmin =-1, vmax =1)
    ax3.set_title('Z')
    ax3.axis('off')           

    cb = fig.colorbar(third, ax=axlist, shrink = 0.4, aspect=10) 
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb.locator = tick_locator
    cb.update_ticks() 
    
    plt.show(block=False)