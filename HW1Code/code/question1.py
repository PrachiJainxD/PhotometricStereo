import matplotlib.pyplot as plt
import numpy as np
from numpy.lib import scimath as SM
np.seterr(divide='ignore', invalid='ignore')
import os
import warnings

def lambertian_sphere(radius, albedo):
        xx = np.linspace(-1, 1, 1501)
        yy = np.linspace(-1, 1, 1501)
        x, y = np.meshgrid(xx, yy)

        #p = -x/sqrt(r^2 - (x^2 + y^2)) where r is radius - 1 meter
        px = np.power(x,2)
        py = np.power(y,2)
        pz = SM.sqrt(radius ** 2 - (px + py))


        p = -x / pz
        q = -y / pz
        mask = np.copy(1 ** 2 - (px + py))

        mask[mask >= 0] = 1
        mask[mask < 0] = 0

        I = np.array([1,1,1])
        I = np.transpose(I)

        N_dot_S = (-I[0] * p - I[1] * q + I[2]) / (SM.sqrt(1 + np.power(p,2) + np.power(q,2)))
        #For a perfect Lambertian surface : Radiosity = Albedo * Unit Normal * Source vector 
        R = (albedo * N_dot_S )

        mask = np.reshape(mask, (1501,1501))
        R = np.multiply (R, mask)
        R*=-1
        
        E = np.copy(R)
        E[E < 0] = 0
        where_are_NaNs = np.isnan(E)
        E[where_are_NaNs] = 0
        E = E / np.amax(E)
        plt.imshow(np.real(E), 'gray')
        plt.title('1.a A perfect Lambertian surface.')
        plt.show()

def spec_sphere(r=1, z=2, specular=True, f=128, h=256, w=384, cx=192, cy=128):
  #Refrences and articles used for understanding:
  #https://www.robots.ox.ac.uk/~att/index.html
  #https://github.com/amanrajdce/ComputerVision-CSE-252A/blob/master/HW2/HW2.ipynb
  # Assume that camera is at origin looking in the +Z direction. X axis to the right, Y axis downwards.
  # Also sphere is at (0, 0, z) and of radius r.
  x, y = np.meshgrid(np.arange(w), np.arange(h))
  x = x - cx
  y = y - cy

  phi = np.arccos(f / np.sqrt(x**2 + y**2 + f**2))
  # equation is rho*2 (1+tan^2(phi)) - 2 rho * z + z*z - r*r = 0
  a = np.tan(phi)**2 + 1
  b = -2 * z
  c = z**2 - r**2
  
  rho = -b - np.sqrt(b**2 - 4*a*c)
  rho = rho / 2 / a
  depth = rho

  # Calculate the point on the sphere surface
  X = x * depth / f
  Y = y * depth / f
  Z = depth
  
  N = np.array([X, Y, Z])
  N = np.transpose(N, [1,2,0])
  N = N - np.array([[[0, 0, z]]])
  N = N / np.linalg.norm(N, axis=2, keepdims=True)
  Z[np.isnan(Z)] = np.inf

  #compute k_d, k_a
  A = (np.sign(X*Y) > 0) * 0.5 + 0.1

  #set specularity
  if specular:
    S = np.invert(np.isinf(depth)) + 0
  else:
    S = A-A
  return Z, N, A, S
  
ke = 50

def render(Z, N, A, S, point_light_location, point_light_strength, directional_light, directional_light_strength, ambient_light, ke):
  #Assume (cx, cy) denote the center of the image, f is the focal length.
  h, w = A.shape
  cx, cy = w / 2, h /2
  f = 128.

  #Get the camera coordinates of all the points in the pixel coordinates
  coordinates = np.zeros((h, w, 3))
  for y in range(h):
      for x in range(w):
          depth = Z[y, x]
          coord = [(x - cx) * depth/f, (y - cy) * depth/f, depth]
          coordinates[y, x, :] = np.array(coord)

  #Normalize the directional light
  directional_light = np.array(directional_light)/np.linalg.norm(np.array(directional_light))

  #Ambient Term
  I = A * ambient_light

  #Diffuse Term
  point_light_direction = np.array(point_light_location) - coordinates
  point_light_direction = (point_light_direction.T/np.linalg.norm(point_light_direction, axis=2).T).T

  #Dot product of direction vectors with the surface normal vectors
  point_diffuse = np.sum(point_light_direction * N, axis=2)
  directional_diffuse = np.sum(directional_light * N, axis=2)
  
  #Filter out the values that are negative in the diffuse matrix
  filter = point_diffuse < 0
  point_diffuse[filter] = 0
  filter = directional_diffuse < 0
  directional_diffuse[filter] = 0

  #Specular Term  
  I += A*(point_diffuse * np.array(point_light_strength) + directional_diffuse * np.array(directional_light_strength))

  #Get the viewing directions and normalize
  vr = np.zeros(3) - coordinates
  vr = (vr.T/np.linalg.norm(vr, axis=2).T).T

  #Reflection is the reverse of the incident light
  point_reflection = (2*np.sum(point_light_direction * N, axis=2).T*N.T).T - point_light_direction
  directional_reflection = (2*np.sum(directional_light * N, axis=2).T*N.T).T - directional_light

  #Get the specular matrices
  point_specular = np.sum(point_reflection *vr, axis=2)
  directional_specular = np.sum(directional_reflection * vr, axis=2)
  
  #Filter out the negative values in the specular matrices
  filter = point_specular < 0
  point_specular[filter] = 0
  filter = directional_specular < 0
  directional_specular[filter] = 0
  
  I += S*(np.array(point_light_strength) * (point_specular)**ke + np.array(directional_light_strength) * (directional_specular)** ke)

  I = np.minimum(I, 1)*255
  I = I.astype(np.uint8)
  I = np.repeat(I[:,:,np.newaxis], 3, axis=2)
  return I

def main():
  lambertian_sphere(1, 1)
  for specular in [True]:
    # spec_sphere function returns:
    # - Z (depth image: distance to scene point from camera center, along the Z-axis)
    # - N is the per pixel surface normals
    # - A is the per pixel ambient and diffuse reflection coefficient per pixel
    # - S is the per pixel specular reflection coefficient
    Z, N, A, S = spec_sphere(specular=True)

    # Strength of the ambient light.
    ambient_light = 0.5

    #No directional light, only point light source that moves around the object.
    #point_light_strength = [1.5]
    directional_light_dirn = [[1, 0, 0]]
    directional_light_strength = [0.0]
    exponents = [0.0, 0.25, 0.5 , 0.75, 1, 1.25, 1.5, 1.75, 2.00, 2.5, 5, 10, 15, 20, 25, 30]

    fig, axes = plt.subplots(4, 4, figsize=(15,10))
    axes = axes.ravel()[::-1].tolist()
    theta =  5.86430629
    for point_light_strength in exponents:
      point_light_loc = [[10*np.cos(theta), 10*np.sin(theta), -3]]
      I = render(Z, N, A, S, point_light_loc, point_light_strength, directional_light_dirn, directional_light_strength, ambient_light, ke)
      ax = axes.pop()
      ax.imshow(I)
      ax.set_axis_off()
 
    #Specular surface rendered using the Phong reflection model with varying exponents
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()