# -----------------------------------------------------------------------------
# Implementation of the "intensive" integration algorithm used to calculate
# spectra. Use to perform electric field calculations.
# -----------------------------------------------------------------------------

cdef extern from "math.h":
    double sin(double x)
    double cos(double x)
    double sqrt(double x)
    double atan(double x)
    double exp(double x)
    double log(double x)

from libc.stdlib cimport malloc, free
from cpython cimport array
import array
import random
cimport numpy as np
import numpy as np
cimport cython
import multiprocessing as mp
import time
import sys
from scipy.integrate import dblquad, nquad
import matplotlib.pyplot as plt

#----------------------------------------------------------------
# Define physical constants
#----------------------------------------------------------------
cdef double pi = np.pi
cdef double e = np.e
cdef double hbar = 6.582e-16          # Planck's constant (eV*s)
cdef double e0 = 8.85e-12             # epsilon 0 (C^2/Nm^2)
cdef double c = 2.998e8               # speed of light (m/s)
cdef double r = 2.818e-15             # electron radius (m)
cdef double ec = 511e3                # electron rest energy (eV)
cdef double q = 1.602e-19             # electron charge (C)
cdef double eVtoJ = 1.602176565e-19   # eV to Joules conversion
cdef double me  = 9.10938356e-31      # electron mass (kg)
cdef double mu0 = 4*pi*1e-7			  # [N/A^2]

#================================================================
# COMPTON SCATTERING PARAMETERS: OK to edit
#----------------------------------------------------------------
cdef int Ncurve = 1
#================================================================

#================================================================

# ELECTRIC FIELD: OK to edit

#----------------------------------------------------------------

# E of omega` is the equation for the modified electric field
# of the wave packet.  The E function defined here is for a
# Gaussian wave front, but it can be re-coded to solve the
# equation for other wave shapes. x is the angular frequency
# of incident photon.

#=========E=======================================================

#----------------------------------------------------------------

# W is the angular frequency of the incident photon as a function
# of the theta and phi angles of the scattered photon (t,p), and
# the angular frequency of the scattered photon. dW is the
# partial derivative of W in terms of the scattered frequency

#----------------------------------------------------------------

cdef double W(double x,double y,double z,double tc,double ph,double wp,double pm,double g,double B):
    cdef double ts=sqrt(1.0-tc*tc)
    return ((wp*(1-(B/pm)*(x*ts*cos(ph)+ y*ts*sin(ph)+ z
                *tc)))/(1+B*(z/pm)-((wp*hbar)/(g*ec))*(1+tc)))

cdef double dW(double x,double y,double z,double tc,double ph,double wp,double pm,double g,double B):
    cdef double ts=sqrt(1.0-tc*tc)
    cdef double phc=cos(ph)
    cdef double phs=sin(ph)
    return (1+(B*z/pm)-((B/pm)*(x*ts*phc+y*ts*phs+z*tc))
                -((((B**2)*z)/(pm**2))*(x*ts*phc+y*ts*
                phs+z*tc)))/((1+((B*z)/pm)-((wp*hbar)/(g*ec))*
                (1+tc))**2)

#-----------------------------------------------------------------

# DSX is the cross section of the scattering event as a function
# of the theta and phi angles of the scattered photon (t,p), and
# the angular frequency of the scattered photon

#-----------------------------------------------------------------

cdef double dsx0(double x,double y,double z,double tc,double ph,double wp,double w,double pm,double g,double B):
    cdef double ts=sqrt(1.0-tc*tc)
    return ((1+tc-((wp*hbar)/(g*ec))*(1+tc))/
                        (1-(B/pm)*(x*ts*cos(ph)+y*ts*
                        sin(ph)+z*tc)))**2

cdef double dsx1(double tc,double ph,double wp,double g,double B):
    return (1+tc-((wp*hbar)/(g*ec))*(1+tc))/(1+B)

cdef double dsx2(double x,double y,double z,double tc,double ph,double g):
    cdef double ts=sqrt(1.0-tc*tc)
    cdef double phc=cos(ph)
    return (2*(me*ts*phc)**2)/((g*me
                    -(1/c)*(x*ts*phc+
                    y*ts*sin(ph)+z*tc))**2)

cdef double dsx3(double x,double y,double z,double tc,double ph,double g):
    cdef double ts=sqrt(1.0-tc*tc)
    cdef double phc=cos(ph)
    return (((2*me)**2)*c*x*ts*(1+tc)*phc)/((me*
                    g+(z/c))*(g*me*c-x*ts*phc-y*
                    ts*sin(ph)-z*tc)**2)

cdef double dsx4(double x,double y,double z,double tc,double ph,double g):
    cdef double ts=sqrt(1.0-tc*tc)
    return (2*(me*x*(1+tc))**2)/((g*me*c+z/c)*(g*me*
                    c-x*ts*cos(ph)-y*ts*sin(ph)-z*
                    tc))**2

cdef double DSX(double x0, double x1, double x2, double x3, double x4, double g, double B):
    return (0.5*(r/(g*(1+B)))**2)*x0*(x1+(1/x1)-x2+x3-x4)

#-----------------------------------------------------------------

# dE is the energy density spectrum of the scattering event as a
# function of the electric field of thelaser pulse (EF), the
# differential cross section (DX), scattered angular frequency (wp),
# incident angular frequency (w), and the first derivative of the
# incident angular frequency function in terms of the scattered
# angular frequency dw).

#-----------------------------------------------------------------

cdef double dE1 = ((e0*1.602*(1e-7))/(2*pi*c*hbar))

# Final expression of the energy density to be integrated to calculate spectra
cdef double dE(double EF, double DX, double wp, double w, double dw):

    return dE1*(abs(EF)**2)*DX*((wp/w)*dw)

# Calculates the total energy spectra by integrating the energy density over a 
# specified number of the particles
def loopParticles(delta_x, delta_y, sigmaPmag, pmag0, dOmegap, Nomegap, Aper, start, E1, E2, E3, E4):

    E = lambda x: x*E1*e**(E2*(x-E3)**2)*(e**(x*E4)+1)

    p0 = pmag0*random.gauss(0,1) * delta_x
    p1 = pmag0*random.gauss(0,1) * delta_y
    p2 = sqrt((pmag0+sigmaPmag*random.gauss(0,1))**2-(p0**2)-(p1**2))

    Y = [0 for x in range(0, Nomegap)]

    pmag1 = sqrt(p0**2+p1**2+p2**2)  # electron momentum magnitude[kg*m/s]

    g1 = sqrt(1+(pmag1/(me*c))**2)   # Lorentz factor of electron frame

    B1 = pmag1/(g1*me*c)             # relativistic speed of electron frame

		                 # ^^ Electron momentum magnitude

    options = {'limit':100}

    for n in range(0, Nomegap):

        x2 = n * dOmegap + start

        Ws2 = ((x2 * (10 ** 6)) / hbar)

        yp = nquad(lambda ctheta, phi: dE(E(W(p0,p1,p2,ctheta,phi,Ws2,pmag1,g1,B1))

		                    ,DSX(dsx0(p0,p1,p2,ctheta,phi,Ws2

		                    ,W(p0,p1,p2,ctheta,phi,Ws2

		                    ,pmag1,g1,B1),pmag1,g1,B1)

		                    ,dsx1(ctheta,phi,Ws2,g1,B1)

		                    ,dsx2(p0,p1,p2,ctheta,phi,g1)

		                    ,dsx3(p0,p1,p2,ctheta,phi,g1)

		                    ,dsx4(p0,p1,p2,ctheta,phi,g1),g1,B1)

		                    ,Ws2, W(p0,p1,p2,ctheta,phi,Ws2,pmag1,g1,B1)

		                    ,dW(p0,p1,p2,ctheta,phi,Ws2,pmag1,g1,B1))

		                    ,[[0,2*pi],[cos(Aper)]],opts=[options,options])

        Y[n] += 2*yp[0]
    
    return Y

# Calculates the total energy spectra by integrating the energy density over a 
# specified number of the particles
def loopParticles2(p0, p1, p2, dOmegap, Nomegap, Aper, start, E1, E2, E3, E4):

    E = lambda x: x*E1*e**(E2*(x-E3)**2)*(e**(x*E4)+1) # Auxiliary electric field

    Y = [0 for x in range(0, Nomegap)]

    pmag1 = sqrt(p0**2+p1**2+p2**2)  # electron momentum magnitude[kg*m/s]

    g1 = sqrt(1+(pmag1/(me*c))**2)   # Lorentz factor of electron frame

    B1 = pmag1/(g1*me*c)             # relativistic speed of electron frame

		                 # ^^ Electron momentum magnitude

    for n in range(0, Nomegap):

        x2 = n * dOmegap + start

        Ws2 = ((x2 * (10 ** 6)) / hbar)

        yp = dblquad(lambda ctheta, phi: dE(E(W(p0,p1,p2,ctheta,phi,Ws2,pmag1,g1,B1))

		                    ,DSX(dsx0(p0,p1,p2,ctheta,phi,Ws2

		                    ,W(p0,p1,p2,ctheta,phi,Ws2

		                    ,pmag1,g1,B1),pmag1,g1,B1)

		                    ,dsx1(ctheta,phi,Ws2,g1,B1)

		                    ,dsx2(p0,p1,p2,ctheta,phi,g1)

		                    ,dsx3(p0,p1,p2,ctheta,phi,g1)

		                    ,dsx4(p0,p1,p2,ctheta,phi,g1),g1,B1)

		                    ,Ws2, W(p0,p1,p2,ctheta,phi,Ws2,pmag1,g1,B1)

		                    ,dW(p0,p1,p2,ctheta,phi,Ws2,pmag1,g1,B1))

		                    ,0,2*pi,lambda ctheta:cos(Aper),lambda ctheta:1.0)

        Y[n] += 2*yp[0]
    
    return Y
