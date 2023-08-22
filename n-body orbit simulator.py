# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:39:15 2023

@author: Sasqwhat
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:10:17 2023

@author: Sasqwhat
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
%matplotlib qt
plt.style.use('dark_background')

silvr = np.sqrt(2)

G = 6.011e-11
c = 299792458

bodies = ['sun','earth','moon','mars','venus','mercury','jupiter']

m1 = 2e+30
m2 = 6e+24
m3 = 7e+22
m4 = 6.4e+23
m5 = 4.9e+24
m6 = 3.3e+23
m7 = 1.9e+27

#OBJ = [x_pos,x_vel,y_pos,y_vel]
obj1 = [0,0,0,0]                        #Sun at origin for starters
obj2 = [1.5e+11,0,0,30000]              #Earth with radial velocity 30km/s
obj3 = [1.5e+11 + 384400000,0,0,31027]  #Moon
obj4 = [2.4e+11,0,0,24100]              #Mars
obj5 = [1.09e+11,0,0,35020]             #Venus
obj6 = [5.7e+10,0,0,47000]              #Mercury
obj7 = [7.4e+11,0,0,13070]              #Jupiter

m = np.array([m1,m2,m3,m4,m5,m6])
z = np.array([obj1,obj2,obj3,obj4,obj5,obj6])

eps =0.1

#INPUTS
t = 0
dt = 1*24*3600
LEN = 4*365.25*24*3600

def com(m,z):
    """
    Parameters
    ----------
    m : Mass of objs 1D array.
    z : 2D position of objs [[x1,vx1,y1,vy1],[x2,vx2,y2,vy2]....].

    Returns
    -------
    Center of mass of the system [x,y]
    """
    x_com = np.sum(m*z[:,0])/np.sum(m)
    y_com = np.sum(m*z[:,2])/np.sum(m)
    
    return np.array([x_com,y_com])

def newt(m,z):
    """

    Parameters
    ----------
    m : Mass of objs 1D array.
    z : 2D position of objs [[x1,vx1,y1,vy1],[x2,vx2,y2,vy2]....].

    Returns
    -------
    Acceleration 2D array [[ax1,ay1].[ax2,ay2],....].

    """
    acc = []
    for j in range(len(m)):
        ax, ay = 0, 0
        for i in range(len(m)):
            if j != i:
                #Using PN approx
                lorentz = 1 - (z[i,1]**2 + z[i,3]**2)/c**2
                
                a_m = G*m[i]/((z[j,0]-z[i,0])**2 + (z[j,2]-z[i,2])**2)**(3/2) * 1/lorentz
                ax += -a_m*(z[j,0] - z[i,0])
                ay += -a_m*(z[j,2] - z[i,2])
        acc.append([ax,ay])
    
    return np.array(acc)

def diff(t,m,z):
    """

    Parameters
    ----------
    t : Time (float).
    m : Mass of objs 1D array.
    z : 2D position of objs [[x1,vx1,y1,vy1],[x2,vx2,y2,vy2]....].

    Returns
    -------
    diff : Returns 2D array of differential of z
           [[vx1,ax1,vy1,ay1],[vx2,ax2,vy2,ay2]...].

    """

    acc = newt(m,z)
    
    diff = np.zeros_like(z)
    
    diff[:,0] = z[:,1]
    diff[:,1] = acc[:,0]
    diff[:,2] = z[:,3]
    diff[:,3] = acc[:,1]
    
    return diff


def rk4(t,dt,m,z):
    """

    Parameters
    ----------
    t : Time (float).
    dt : Time step (float).
    m : Mass of objs 1D array.
    z : 2D position of objs [[x1,vx1,y1,vy1],[x2,vx2,y2,vy2]....].

    Returns
    -------
    TYPE
        Integrated z via RK4, over time step dt.

    """
    k1 = diff(t,m,z) * dt
    k2 = diff(t + 0.5*dt, m, z+0.5*k1) * dt
    k3 = diff(t + 0.5*dt, m, z+0.5*k2) * dt
    k4 = diff(t + dt,m, z+k3) * dt
    
    return z + 1/6*(k1+2*k2+2*k3+k4)


def loop(t,dt,LEN,m,z):
    
    t0 = t
    z_super = []
    while (t < t0+LEN):
        z = rk4(t,dt,m,z)
        t += dt
        z_super.append(z)
    return z_super


#ANIMATION FUNCTIONS

def anim(frames,Z, fig, ax):
    z0 = Z[frames]
    fig.suptitle('Time elapsed: {0} days'.format(frames))
    ax.scatter(z0[0,0],z0[0,2],alpha=0.4,marker='*',c='yellow', label='Sun')
    ax.scatter(z0[1,0],z0[1,2],alpha=0.4,s=2,c='blue', label='Earth')
    ax.scatter(z0[2,0],z0[2,2],alpha=0.5,s=3,c='grey', label='Moon')
    ax.scatter(z0[3,0],z0[3,2],alpha=0.4,s=2,c='red', label='Mars')
    ax.scatter(z0[4,0],z0[4,2],alpha=0.4,s=2,c='orange', label='Venus')
    ax.scatter(z0[5,0],z0[5,2],alpha=0.4,s=2,c='brown', label='Mercury')
    #plt.scatter(z[6,0],z[6,2],alpha=0.4,s=5,c='brown', label='Jupiter')
    ax.set_xlim(-3.5e11,3.5e11), ax.set_ylim(-3.5e11,3.5e11)
    #plt.legend()
    return ax

def anim_geo(frames,Z, fig, ax):
    z0 = Z[frames][:,:] - Z_SUPER[frames][1,:]
    fig.suptitle('Time elapsed: {0} days'.format(frames))
    ax.scatter(z0[0,0],z0[0,2],alpha=0.4,marker='*',c='yellow', label='Sun')
    ax.scatter(z0[1,0],z0[1,2],alpha=0.4,s=2,c='blue', label='Earth')
    ax.scatter(z0[2,0],z0[2,2],alpha=0.5,s=3,c='grey', label='Moon')
    ax.scatter(z0[3,0],z0[3,2],alpha=0.4,s=2,c='red', label='Mars')
    ax.scatter(z0[4,0],z0[4,2],alpha=0.4,s=2,c='orange', label='Venus')
    ax.scatter(z0[5,0],z0[5,2],alpha=0.4,s=2,c='brown', label='Mercury')
    #plt.scatter(z[6,0],z[6,2],alpha=0.4,s=5,c='brown', label='Jupiter')
    ax.set_xlim(-5e11,5e11), ax.set_ylim(-5e11,5e11)
    return ax

def anim_EM(frames,Z, fig, ax):
    z0 = Z[frames][:,:]
    ax.scatter(z0[1,0],z0[1,2],alpha=0.4,c='blue', label='Earth')
    ax.scatter(z0[2,0],z0[2,2],alpha=0.5,c='grey', label='Moon')
    ax.set_xlim(z0[1,0]-1e9,z0[1,0]+1e9), ax.set_ylim(z0[1,2]-1e9,z0[1,2]+1e9)
    return ax

def anim_comb(frames, Z, fig, ax):
    z0 = Z[frames]
    fig.suptitle('Time elapsed: {0} days'.format(frames))
    ax[0].scatter(z0[0,0],z0[0,2],alpha=0.4,marker='*',c='yellow', label='Sun')
    ax[0].scatter(z0[1,0],z0[1,2],alpha=0.4,s=2,c='blue', label='Earth')
    ax[0].scatter(z0[2,0],z0[2,2],alpha=0.5,s=3,c='grey', label='Moon')
    ax[0].scatter(z0[3,0],z0[3,2],alpha=0.4,s=2,c='red', label='Mars')
    ax[0].scatter(z0[4,0],z0[4,2],alpha=0.4,s=2,c='orange', label='Venus')
    ax[0].scatter(z0[5,0],z0[5,2],alpha=0.4,s=2,c='brown', label='Mercury')
    #plt.scatter(z[6,0],z[6,2],alpha=0.4,s=5,c='brown', label='Jupiter')
    ax[0].set_title('Heliocentric')
    ax[0].set_xlim(-5e11,4e11), ax[0].set_ylim(-3e11,3e11)
    
    ax[1].clear()
    ax[1].scatter(z0[1,0],z0[1,2],alpha=0.4,c='blue', label='Earth')
    ax[1].scatter(z0[2,0],z0[2,2],alpha=0.5,c='grey', label='Moon')
    ax[1].set_xlim(z0[1,0]-1e9,z0[1,0]+1e9), ax[1].set_ylim(z0[1,2]-1e9,z0[1,2]+1e9)
    ax[1].set_xticks([]), ax[1].set_yticks([])
    ax[1].set_title('Moon orbit')
    return ax


def earth_plus_moon():
    t = 0
    dt = 1*24*3600
    LEN = 4*365.25*24*3600
    Z_SUPER = loop(t,dt,LEN,m,z)

    fig, ax = plt.subplots(1,2,gridspec_kw={'width_ratios': [2,1]}, figsize=(10*silvr,5))
    ax = np.ravel(ax)
    line = animation.FuncAnimation(fig, anim_comb, interval=100,frames=len(Z_SUPER), fargs=(Z_SUPER, fig, ax))
    writer = animation.PillowWriter(fps=60)
    line.save('Orbits.gif', writer=writer)
    plt.show()
    return


def geocentric():
    t = 0
    dt = 1*24*3600
    LEN = 4*365.25*24*3600
    Z_SUPER = loop(t,dt,LEN,m,z)
    
    fig, ax = plt.subplots(1,1)
    line = animation.FuncAnimation(fig, anim_geo, interval=100,frames=len(Z_SUPER), fargs=(Z_SUPER, fig, ax))
    writer = animation.PillowWriter(fps=60)
    line.save('Solar system rocky geocentric.gif', writer=writer)
    plt.show()
    return


def heliocentric():
    t = 0
    dt = 1*24*3600
    LEN = 4*365.25*24*3600
    Z_SUPER = loop(t,dt,LEN,m,z)
    
    fig, ax = plt.subplots(1,1)
    line = animation.FuncAnimation(fig, anim, interval=100,frames=len(Z_SUPER), fargs=(Z_SUPER, fig, ax))
    writer = animation.PillowWriter(fps=60)
    line.save('Solar system rocky.gif', writer=writer)
    plt.show()
    return

