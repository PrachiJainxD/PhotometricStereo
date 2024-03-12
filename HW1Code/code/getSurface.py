import numpy as np

#library for performing random integration
import random

def getSurface(surfaceNormals, method):

    #partial derivatives computation wrt z
    partial_x = surfaceNormals[:, :, 0] / surfaceNormals[:, :, 2]
    partial_y = surfaceNormals[:, :, 1] / surfaceNormals[:, :, 2]
    
    row_sum = np.cumsum(partial_x, axis=1)
    col_sum = np.cumsum(partial_y, axis=0)
    
    if method == 'row-column':
        '''
        Integrating first the rows, then the columns. That is, your path first goes along the same row as
        the pixel along the top, and then goes vertically down to the pixel. Call this option “row-column”
        '''    
        return col_sum[:, 0][:, np.newaxis] + row_sum
    
    if method == 'column-row':
        '''
        Integrating first along the columns, then the rows. Call this option “column-row”.
        '''
        return row_sum[0] + col_sum
    
    if method == 'average':
        '''
        Average of the first two options.
        '''
        row_column_ = getSurface(surfaceNormals, method="row-column")
        column_row_ = getSurface(surfaceNormals, method="column-row")
        heightMap = (row_column_ + column_row_)//2
        return heightMap
    
    if method == 'random':
        '''
        Average of multiple random paths. There are several ways to sample random paths, 
        and we leave it up to you to figure out what works best.Determine the number of required paths experimentally 
        by visualizing the results for different number of paths. You might find switching back to the toy example and 
        running the algorithm on smaller images useful to debugging as this approach is relatively slower.
        '''    
        #flipping coins to 50 generate random paths        
        h = surfaceNormals.shape[0]
        w = surfaceNormals.shape[1]
        heightMap = np.zeros((h, w))
        
        #configure number of random paths
        num_paths = 50
        
        #loop through each pixel.
        for y in range(h):#Row
            for x in range(w):#Column       
                #exclude the start pixel(0, 0)
                if x != 0 or y != 0:   
                    for path in range(num_paths):
                        #guarantees #0 = x, #1 = y in coins
                        zeros = [0] * x
                        ones = [1] * y
                        coin = np.array(zeros + ones)
                        #randomly shuffle coins to create sudo-random path
                        np.random.shuffle(coin)
                        
                        #start pixel
                        xCurrent = 0
                        yCurrent = 0
                        step = 0
                        cumsum = 0
                        
                        #move along path until reach target pixel
                        while xCurrent < x or yCurrent < y:
                            #right movement
                            if coin[step] == 0:
                                cumsum += partial_x[yCurrent, xCurrent]
                                xCurrent += 1
                            else:
                                cumsum += partial_y[yCurrent, xCurrent]
                                yCurrent += 1
                            
                            step += 1
                        
                        #add cumsum along path to heightMap
                        heightMap[y, x] += cumsum
                    
                    #compute average over paths
                    heightMap[y, x] = heightMap[y, x]/num_paths
        return heightMap