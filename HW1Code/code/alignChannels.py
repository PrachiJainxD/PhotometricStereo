import numpy as np

# importing libraries for saving images in lighter memory format and edge detection
import skimage as sk
from scipy.ndimage import sobel

def alignChannels(img, max_shift):
    #convert to double to save memory   
    img = sk.img_as_float(img)
    
    #separate color channels
    #Blue channel
    channel1 = img[:, :, 2]
    #Green channel
    channel2 = img[:, :, 1] 
    #Red channel
    channel3 = img[:, :, 0]
    
    #aligning central crop of the images to ignore borders
    img_c1 = channel1[int(0.15 * len(channel1)):-int(0.15 * len(channel1)), int(0.15 * len(channel1[0])):-int(0.15 * len(channel1[0]))]
    img_c2 = channel2[int(0.15 * len(channel2)):-int(0.15 * len(channel2)), int(0.15 * len(channel2[0])):-int(0.15 * len(channel2[0]))]
    img_c3 = channel3[int(0.15 * len(channel3)):-int(0.15 * len(channel3)), int(0.15 * len(channel3[0])):-int(0.15 * len(channel3[0]))]
    
    
    best_score1 = -float('inf')
    best_score2 = -float('inf')
    best_shift1 = [0, 0]
    best_shift2 = [0, 0]
    
    #using cosine similarity and ignoring edge
    method = 'CS'
    use_edge = False
    
    #perfroming green shift by looping over all different displacement permutations wrt to blue channel
    for i in range(-max_shift[0], max_shift[0] + 1):
        for j in range(-max_shift[1], max_shift[1] + 1):
            current_score1 = compute_score(np.roll(img_c2, (i, j), (0, 1)), img_c1, method, use_edge)
            if current_score1 > best_score1:
                best_score1 = current_score1
                best_shift1 = [i, j]
    
    shifted_img1 = np.roll(channel2, best_shift1, (0, 1))

    #perfroming red shift by looping over all the different displacement permutations wrt to blue channel
    for i in range(-max_shift[0], max_shift[1] + 1):
        for j in range(-max_shift[0], max_shift[1] + 1):
            current_score2 = compute_score(np.roll(img_c3, (i, j), (0, 1)), img_c1, method, use_edge)
            if current_score2 > best_score2:
                best_score2 = current_score2
                best_shift2 = [i, j]
    
    shifted_img2 = np.roll(channel3, best_shift2, (0, 1))    
    
    #creates color image in RGB format - red shift, green shift & blue channel
    new_shifted_img = np.dstack([shifted_img2,shifted_img1,channel1])
    
    #crops final result to remove boundary artifacts
    colour_img = new_shifted_img[int(0.1 * len(new_shifted_img)):-int(0.1 * len(new_shifted_img)), int(0.1 * len(new_shifted_img[0])):-int(0.1 * len(new_shifted_img[0]))]    

    #returns best displaced image along with the displacement vector
    return colour_img, np.vstack((best_shift1,best_shift2))
    
def compute_score(image1, image2, method, use_edge):
    #returns similarity score of image1 and image2    
    if use_edge:
    #computes sobel filter by considering boundaries 
        image1 = np.abs(sobel(image1))
        image2 = np.abs(sobel(image2))
        
    if method == 'SSD':
    #computes Sum of Squared Differences(SSD) between two images
        return np.sum((np.array(image1, dtype=np.float32) - np.array(image2, dtype=np.float32))**2)
        
    elif method == 'CS':
    #computes Cosine Similarity(CS) between two images
        image1 = np.ndarray.flatten(image1)
        image2 = np.ndarray.flatten(image2)
        return np.dot(image1 / np.linalg.norm(image1), image2 / np.linalg.norm(image2))