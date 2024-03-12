import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

def set_aspect_equal_3d(ax):
    """https://stackoverflow.com/questions/13685386"""
    """Fix equal aspect bug for 3D plots."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)
    plot_radius = max([
        abs(lim - mean_)
        for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
        for lim in lims
    ])
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])
    


def displayOutput(albedo, heightMap, azim_angle=45, elev_angle=20):

    fig = plt.figure()
    plt.imshow(albedo, cmap='gray')
    plt.title('Estimated Albedo')
    plt.axis('off')
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(20, 20)
    X = np.arange(albedo.shape[0])
    Y = np.arange(albedo.shape[1])
    
    X, Y = np.meshgrid(Y, X)
    H = np.flipud(np.fliplr(heightMap))
    A = np.flipud(np.fliplr(albedo))
    A = np.stack([A, A, A], axis=-1)
    ax.xaxis.set_ticks([])
    ax.xaxis.set_label_text('Z')
    ax.yaxis.set_ticks([])
    ax.yaxis.set_label_text('X')
    ax.zaxis.set_ticks([])
    ax.yaxis.set_label_text('Y')
    
    min = 0.0
    max = 1.0
    scalarMap = cm.ScalarMappable(norm=Normalize(vmin=min, vmax=max), cmap='gray')
    A_color = scalarMap.to_rgba(A)
    ax.view_init(azim=azim_angle, elev=elev_angle)
    
    surf = ax.plot_surface(H, X, Y, cmap='gray', facecolors=A_color, linewidth=0, antialiased=False)
    set_aspect_equal_3d(ax)
    plt.title('Integrated Height Map')
    plt.show(block=False)
    
if __name__ == '__main__':
    displayOutput(np.zeros((50, 50)), np.zeros((50, 50)))    