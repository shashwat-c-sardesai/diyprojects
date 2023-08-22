# -*- coding: utf-8 -*-
"""
To create imaginary and real slices of the Mandelbrot set 4D object
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
%matplotlib qt

LEN = 2500
FRAMES=240
X = np.linspace(-2.1,0.6,LEN)
Y = np.linspace(-1.2,1.2,LEN)
x,y = np.meshgrid(X,Y)

#The 'z' used for the 2D slice
low = -1.4
high = 1.4
z_0 = np.linspace(low,high,FRAMES)

def mandel_slice(num,type='real'):
    """
    Animation function

    Parameters
    ----------
    num : slice of linspace z_0
    type: imag or real
    
    Returns
    -------
    Image of plot.

    """
    if type='real':
        z = recur(x,y,a=z_0[num])
        mask = (np.abs(z) <= 2.)
        im.set_array(mask)
        plt.title('c = {0}'.format(np.round(z_0[num],4)))
        
    if type='imag':
        z = recur(x,y,b=z_0[num])
        mask = (np.abs(z) <= 2.)
        im.set_array(mask)
        plt.title('c = {0}i'.format(np.round(z_0[num],4)))
    
    return [im]

def recur(x,y,a=0,b=0,iter=25):
    """
    Parameters
    ----------
    x : Meshgrid x.
    y : Meshgrid y.
    a : coefficient of real part.
    b : coefficient of imaginary part.
    iter : How many runs. The default is 25.

    Returns
    -------
    z : Final iteration of mandelbrot recursion.

    """
    z = a*np.ones_like(x)+b*1j*np.ones_like(y)
    c = (x+1j*y)
    for i in range(iter):
        z = z**2 + c
    return z

z_init = recur(x, y)
mask_init = (np.abs(z_init) <= 2.)

fig = plt.figure(figsize=(10,9))
im = plt.imshow(mask_init,cmap=plt.get_cmap('jet'),)
plt.xlim(-2.1,0.6)
plt.ylim(-1.2,1.2)
plt.yticks(np.linspace(0,LEN-1,7),labels=np.round(np.linspace(-1.2,1.2,7),2))
plt.xticks(np.linspace(0,LEN-1,7),labels=np.round(np.linspace(-2.1,0.6,7),2))

line = animation.FuncAnimation(fig, mandel_slice, interval=100,frames=FRAMES, )
writer = animation.PillowWriter(fps=30) 
line.save('mandel_slices_imag.gif', writer=writer)
#ax.set_xticklabels(['','-2.2','-1.78','-1.37','-0.95','-0.53','-0.12','0.3',])
#ax.set_yticklabels(['','-1.1','-0.73','-0.37','0.0','0.37','0.73','1.1',])
#ax.set_title('colorMap')

#ax.imshow(mask)
plt.show()
#ax.set_aspect('equal')