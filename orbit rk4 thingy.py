# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:10:17 2023

@author: Sasqwhat
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sci

G = 6.011e-11
m1 = 2e+30
m2 = 6e+24
m3 = 7e+22
#OBJ = [[x_pos,y_pos],[x_vel,y_vel]]
obj1 = [[0,0],[0,0]]                      #Sun at origin for starters
obj2 = [[1.5e+11,0],[0,30000]]           #Earth with radial velocity 30km/s
obj3 = [[1.5e+11 + 384400000,0],[0,301027]]
eps =0.1

#INPUTS
t = 0
dt = 24*3600
LEN = 10*365.25*24*3600

def com(m1,m2,a1,a2):
    """
    Parameters
    ----------
    m1 : Mass of obj 1.
    m2 : Mass of obj 2.
    a1 : 2D position of obj 1 [x1,y1].
    a2 : 2D position of obj 12 [x2,y2].

    Returns
    -------
    Center of mass of the system [x,y]
    """
    x_com = (m1*a1[0] + m2*a2[0])/(m1+m2)
    y_com = (m1*a1[1] + m2*a2[1])/(m1+m2)
    
    return np.array([x_com,y_com])

def newt(a1,a2):
    """
    Newtonian gravity, a = -GM/a^3 * a->
    Acceleration needs to be negative, vector faces towards center of mass
    
    Parameters
    ----------
    a1 : 2D position of obj 1 [x1,y1].
    a2 : 2D position of obj 12 [x2,y2].
    
    Returns
    -------
    Newtonian acceleration due to gravity [[a_x1,a_y1],[a_x2,a_y2]]
    """
    xcom, ycom = com(m1,m2,a1,a2)
    a = G/((a1[0]-a2[0])**2 + (a1[1]-a2[1])**2)**(3/2)
    acc1 = [-a*m2*(a1[0]-xcom),-a*m2*(a1[1]-ycom)]
    acc2 = [-a*m1*(a2[0]-xcom),-a*m1*(a2[1]-ycom)]
    
    return np.array([acc1,acc2])

def diff(t,obj1,obj2):
    """
    Differential function for RK4 integrator
    
    dy/dt = f(t,y)
    [[pos1_xy,vel1_xy],[pos2_xy,vel2_xy]] -f()-> [[vel1_xy,acc1_xy],[vel2_xy,acc2_xy]]

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    obj1 : TYPE
        position and velocity of obj1
        [pos1_xy,vel1_xy]
    obj2 : TYPE
        position and velocity of obj2
        [pos2_xy,vel2_xy]

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    vel_1 = obj1[1]
    vel_2 = obj2[1]
    
    pos_1 = obj1[0]
    pos_2 = obj2[0]

    a1,a2 = newt(pos_1,pos_2)
    
    obj1_df = [vel_1,a1]
    obj2_df = [vel_2,a2]
    
    return np.array([obj1_df,obj2_df])


def rk4(t,dt,obj1,obj2):
    k1 = diff(t,obj1,obj2) * dt
    k2 = diff(t + 0.5*dt, obj1+0.5*k1[0],obj2+0.5*k1[1]) * dt
    k3 = diff(t + 0.5*dt, obj1+0.5*k2[0],obj2+0.5*k2[1]) * dt
    k4 = diff(t + dt, obj1+k3[0],obj2+k3[1]) * dt
    
    return np.array([obj1,obj2]) + 1/6*(k1+2*k2+2*k3+k4)


def loop(t,dt,LEN,obj1,obj2):
    t0 = t
    while (t+dt < t0+LEN):
        obj1, obj2 = rk4(t,dt,obj1,obj2)
        t += dt
        plt.scatter(obj1[0][0],obj1[0][1], color='orange',alpha=0.4,marker='*',s=1)
        plt.scatter(obj2[0][0],obj2[0][1],color='blue',alpha=0.4,s=1)
       
    return np.array([obj1,obj2])

loop(t,dt,LEN,obj1,obj2)
