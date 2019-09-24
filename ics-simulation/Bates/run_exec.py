# --------------------------------------------------------------------------------
# Driver for the simulations. Utilizes execute.pyx for low-level implementation
# of computationally expensive methods.
# --------------------------------------------------------------------------------

import execute
import array
import random
import numpy as np
import multiprocessing as mp
import time
import sys
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from math import pi,sqrt,cos,sin,atan,e,log,exp

#----------------------------------------------------------------
# Define physical constants
#----------------------------------------------------------------
hbar = 6.582e-16          # Planck's constant (eV*s)
e0 = 8.85e-12             # epsilon 0 (C^2/Nm^2)
c = 2.998e8               # speed of light (m/s)
me = 9.109e-31            # electron mass (kg)
r = 2.818e-15             # electron radius (m)
ec = 511e3                # electron rest energy (eV)
q = 1.602e-19             # electron charge (C)
eVtoJ = 1.602176565e-19   # eV to Joules conversion

#----------------------------------------------------------------
# Read in momentum values from file
#----------------------------------------------------------------
def mewantlist (Location, Column1, Column2, Column3):
        """mewantlist produces a list of lists, where the lists are the 
        chosen columns of data
        takes three indexes for desired columns"""
        column1=[]
        crs=open(Location,"r")
        Np = 0
        for columns in (raw.strip().split() 
                for raw in crs):
                        number1=float(columns[Column1])
                        number2=float(columns[Column2])
                        number3=float(columns[Column3])
                        column1.append([number1,number2,number3])
                        Np = Np + 1
        return column1, Np

arg_file = open("config.in", "r")
args = []
first = arg_file.readline()
first = float(first[3:4])
for line in arg_file:
    i = 0
    while (line[i:i+1] != " "):
        i += 1
    num = float(line[0:i])
    args.append(num)
args.insert(0, first)
args[0] = int(args[0])

# Collect Arguments into List

sigmaEn = -1.0
if (args[0] == 0):
    args[1] = int(args[1])
    op = args[0]
    Npart = args[1]    
    L = args[2]           
    a0 = args[3]   
    s = args[4]
    En0 = args[5]
    sigmaEn = args[6]
    ex = args[7]
    ey = args[8]
    Aper = args[9]
    betastar = args[10]
    start1 = args[11]
    Range1 = args[12]
elif (args[0] == 1):
    p, Npart = mewantlist("data.in", 0, 1, 2)
    En0 = 0.0
    for n in range(0, Npart):
        En0 += sqrt((c**2)*(float(p[n][0])**2 + float(p[n][1])**2 + float(p[n][2])**2 + (me*c)**2))/eVtoJ
    En0 /= Npart
    print "E_0 = ", En0
    op = args[0]
    L = args[1]
    a0 = args[2]
    s = args[3]
    Aper = args[4]
    betastar = args[5]
    start1 = args[6]
    Range1 = args[7]
    ex = 0.0
    ey = 0.0

def eScat(g, X, E):
    fac = 4*(g**2)/(1.0+X)
    Escat = fac*E*1e-6
    return Escat

#----------------------------------------------------------------
# Compute mean energy of the distribution
#----------------------------------------------------------------
Ncurve = 1
gamma0=En0/ec
#================================================================
# COMPTON SCATTERING PARAMETERS: OK to edit
#----------------------------------------------------------------
E_l = hbar*2*pi*c/L
X = 4*E_l*En0/ec/ec
if (Range1 == 0):
  Range = 1/sqrt(Npart)*eScat(En0/ec, X, E_l)*3   # range of E_gamma [MeV]
else:
  Range = Range1
if (start1 == 0):
  start = eScat(En0/ec, X, E_l)-Range/2.0
else:
  start = start1
pmag0 = (eVtoJ/c)*sqrt(En0**2-ec**2)              # (kg m/s)
delta_x = sqrt(ex/betastar)
delta_y = sqrt(ey/betastar)
if (sigmaEn != -1.0):
    sigmaPmag = (eVtoJ/c)*sqrt(((En0*(1.0+sigmaEn))**2)-ec**2)-pmag0
#----------------------------------------------------------------
#----------------------------------------------------------------
# Define resolution of the simulation
#----------------------------------------------------------------
Nomegap = 400         # Iterations for omega prime
#================================================================

#----------------------------------------------------------------
# Predict the amplitude of the peak (if aperture is big enough)
#----------------------------------------------------------------
alpha = 1.0/137.0
gamma = En0/ec
beta = sqrt(1.0-1.0/(gamma**2))
om0 = 2.0*pi*c/L
mt = om0*(hbar*eVtoJ)/(gamma*me*(c**2))
omega_max = om0*(1+beta)/(1-beta+2*mt)
E_max = hbar*omega_max
Amp = pi*alpha*(a0**2)*sqrt(pi)*s/E_max
print "Aperture: ", Aper
#print "Expected maximum amplitude Amp = ", Amp
#================================================================
E1 = (a0*c*me*s*L*sqrt(pi))/(q*(1.602e-13)*sqrt(2.0)) # [MeVs/C]
E2 = (-((s*L)**2)/(2*c**2))                               # [s^2]
E3 = ((2*pi*c)/L)                                         # [1/s]
E4 = (-(4*pi*(s**2)*L)/c)                                 # [s]
#-----------------------------------------------------------------
# Main program
#-----------------------------------------------------------------
def logResult(result):
    results.append(result)

dOmegap = Range/(Nomegap-1)

if __name__ == '__main__':
    cpu = mp.cpu_count()
    if (op == 0):
        Yall = [[0 for x in range(0, Nomegap)] for y in range(0, Ncurve)]
        for j in range(0, Ncurve):
            results = []
            pool = mp.Pool(processes=cpu)  # parallel execute on cpu CPUs
            for x in range(0, Npart):      # paralellize over Npart
                pool.apply_async(execute.loopParticles, args=(delta_x, delta_y, sigmaPmag, pmag0, dOmegap, Nomegap, Aper, start, E1, E2, E3, E4), callback=logResult)
            pool.close()
            pool.join()                    # ends processes when completed
            for i in range(0, Nomegap):
                for n in range(0, len(results)):
                    Yall[j][i] += results[n][i]
            ## Collect x-y coordinates
            xplotN=[]
            yplotN=[]
            yplotNn=[]

            for i in range(0,Nomegap):
                xval=1e3*((i*Range)/(Nomegap-1)+start)
                xplotN.append(xval)
                Yall[0][i]=Yall[0][i]/(1e3*xval)       # [1/eV] for dE/d\omega, comment this line out
            Ynorm=1.0 #max(Y)

            for i in range(0,Nomegap):
                yplotN.append(Yall[0][i]/Npart)
                yplotNn.append(Yall[0][i]/Ynorm)       # NOT normalized to 1

#-----------------------------
# Dump data to a file
#-----------------------------
            f  = open('output.txt','w')
            for i in range(0, Nomegap):
                line = r'%15.8e  %15.8e  %15.8e' % (xplotN[i],yplotN[i],yplotNn[i]) 
                f.write(line)
                f.write('\n')
            f.close()
            Ynorm=max(yplotN)

    elif (op == 1):
        Yall = [[0 for x in range(0, Nomegap)] for y in range(0, Ncurve)]
        for j in range(0, Ncurve):
            results = []
            pool = mp.Pool(processes=cpu)  # parallel execute on cpu CPUs
            for x in range(0, Npart):      # paralellize over Npart
                pool.apply_async(execute.loopParticles2, args=(p[x][0], p[x][1], p[x][2], dOmegap, Nomegap, Aper, start, E1, E2, E3, E4), callback=logResult)
            pool.close()
            pool.join()  # ends processes when completed
            for i in range(0, Nomegap):
                for n in range(0, len(results)):
                    Yall[j][i] += results[n][i]
            ## Collect x-y coordinates
            xplotN=[]
            yplotN=[]
            yplotNn=[]

            for i in range(0,Nomegap):
                xval=1e3*((i*Range)/(Nomegap-1)+start)
                xplotN.append(xval)
                Yall[0][i]=Yall[0][i]/(1e3*xval)       # [1/eV]for dE/d\omega, comment this line out
            Ynorm=1.0 #max(Y)

            for i in range(0,Nomegap):
                yplotN.append(Yall[0][i]/Npart)
                yplotNn.append(Yall[0][i]/Ynorm) # NOT normalized to 1

#-----------------------------
# Dump data to a file
#-----------------------------
            f  = open('output.txt','w')
            for i in range(0, Nomegap):
                line = r'%15.8e  %15.8e  %15.8e' % (xplotN[i],yplotN[i],yplotNn[i]) 
                f.write(line)
                f.write('\n')
            f.close()
            Ynorm=max(yplotN)
