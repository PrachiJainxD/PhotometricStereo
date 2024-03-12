
**1. Photometric Stereo**
   - Implement a basic “shape from shading” algorithm described in 2.2 in Forsyth and Ponce book.
   - The input to the algorithm is a set of photographs of a static scene taken with known lighting directions, and the output of the algorithm is the albedo (paint), normal directions, and the height map.

**2. Aligning Prokudin-Gorskii images**
   - The goal is to take photographs of each plate and generate a color image by aligning them.
   - One way to align the plates is to exhaustively search over a window of possible displacements, say [-15,15] pixels, score each one using some image matching metric, and take the displacement with the best score.
   - This method works because we expect the pixel intensity across R,G,B channels to be correlated.
