# --------------------------------------------------------------------------------
# Driver for the simulations. Utilizes scale.pyx for low-level implementation
# of computationally expensive methods.
# --------------------------------------------------------------------------------

import scale
import array
import random
import numpy as np
import multiprocessing as mp
import time
import sys
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from math import pi,sqrt,cos,sin,atan,e,log,exp

def eScat(g, X, E):
    fac = 4 * (g ** 2) / (1.0 + X)
    Escat = fac * E * 1e-6
    return Escat

#----------------------------------------------------------------
# Define physical constants
#----------------------------------------------------------------
pi = 3.1415926535897932384626433832795028841971693993751
e = 2.71828182845904523536028747135266249775724709369995
hbar = 6.582e-16          # Planck's constant (eV*s)
e0 = 8.85e-12             # epsilon 0 (C^2/Nm^2)
c = 2.998e8               # speed of light (m/s)
eVtoJ = 1.602176565e-19   # eV to Joules conversion
ec = 511e3                # electron rest energy (eV)
me  = 9.10938356e-31      # electron mass (kg)

Ncurve = 1
En0 = 500e6        # mean electron beam energy (eV)
pmag0 = (eVtoJ/c)*sqrt(En0**2-ec**2)  # (kg m/s)
gamma = En0/ec     # relativistic factor
#L = 132.94701e-9    # lambda: peak laser wavelength [m]
L = 800e-9
El = c / L * hbar * 2 * np.pi #Energy of Photon Beam [eV]
Lc = 600.0         # distance from collision pt to collimator [m]
#R = float(sys.argv[3])  # aperture radius [meters]
R  = 0.16         # radius of aperture [m]
Aper = atan(R/Lc)  # aperture [no units]
betastar = 5.0     # beta * [m]
ex = float(sys.argv[3])*1.0e-9           # emittance [m rad]
ey = float(sys.argv[4])*1.0e-9
X = 4 * En0 * El / (ec ** 2) #Electron Recoil Parameter [no units]
Gcm = gamma/np.sqrt(1 + X)  #Gamma at CM
a0 = 0.026
psi = Gcm * Aper
Range = 3.0                           #Range in sampling
start = eScat(gamma,X,El)-Range+.1     # starting point in sampling of E_gamma [MeV]
sg = 0.00225

# Command-line arguments
pz = float(sys.argv[1])
sigma_pz = float(sys.argv[2])
#---------------------------

s = 1.0/(2.0*sqrt(2.0)*pi*sg)

sigma_px = sqrt(ex/betastar)*pmag0
sigma_py = sqrt(ey/betastar)*pmag0

#----------------------------------------------------------------
# Define resolution of the simulation
#----------------------------------------------------------------
Npart = 40           # number of photon-electron collisions
Nomegap = 150         # Iterations for omega prime
#================================================================

#-----------------------------------------------------------------

# Main program

#-----------------------------------------------------------------

p_list = []
e_list = []
p2_list = []
e2_list = []

def logResult(result):
    array=[0 for x in range(0,Nomegap)]
    for x in range(Nomegap):
        array[x]+=result[x]
    results.append(array)
    p_list.append(result[Nomegap])
    p2_list.append(result[Nomegap+1])
    e_list.append(result[Nomegap+2])
    e2_list.append(result[Nomegap+3])

dOmegap = Range/(Nomegap-1)

if __name__ == '__main__':

    Yall = [[0 for x in range (0,Nomegap)] for y in range (0,Ncurve)]
    cpu = mp.cpu_count()
    print cpu, "cpus"

    for j in range(0, Ncurve):

        Y = [0 for x in range (0,Nomegap)]

        results = []

        pool = mp.Pool(processes=cpu)  # parallel execute on cpu CPUs

        for x in range(0, Npart):      # paralellize over Npart

            pool.apply_async(scale.loopParticles,args=(sigma_px,sigma_py

                             ,sigma_pz,pz,dOmegap,Aper,s,Nomegap)

                             ,callback=logResult)

        pool.close()
        pool.join()                    # ends processes when completed

        for i in range(0, Nomegap):
            for n in range(0, len(results)):
                Yall[j][i] += results[n][i]

## Collect x-y coordinates

    xplotN=[]
    yplotN0=[]
    yplotN0n=[]

    for i in range(0,Nomegap):
        xval=((i*Range)/(Nomegap-1))+start
        xplotN.append(xval)

        for j in range(0,Ncurve):
            Yall[j][i]=Yall[j][i]/xval

    Ynorm0=max(Yall[0])

    for i in range(0,Nomegap):
        yplotN0.append(Yall[0][i]*1e4/Npart)
        yplotN0n.append(Yall[0][i]/Ynorm0)

    Ynorm0 = max(yplotN0)

    for i in range(0,Nomegap):
        yplotN0.append(Yall[0][i]*1e4/Npart)
        yplotN0n.append(Yall[0][i]/Ynorm0)

    Ynorm0 = max(yplotN0)
    Ynmax = Ynorm0

#-----------------------------

# Dump data to a file

#-----------------------------
    f  = open('out_fig4b.txt','w')
    fn = open('out_fig4b_n.txt','w')

    for i in range(0, Nomegap):
        line = r'%15.8e  %15.8e' % (xplotN[i],yplotN0[i])
        f.write(line)
        f.write('\n')
        line2 = r'%15.8e  %15.8e' % (xplotN[i],yplotN0n[i])
        fn.write(line2)
        fn.write('\n')

    f.close()
    fn.close()
#-----------------------------

# Read data from a file

#-----------------------------

    xplotN=[]
    yplotN0=[]
    yplotN1=[]
    yplotN2=[]
    yplotN3=[]
    yplotN0n=[]
    yplotN1n=[]
    yplotN2n=[]
    yplotN3n=[]

    d  = np.loadtxt('out_fig4b.txt')
    dn = np.loadtxt('out_fig4b_n.txt')

    for i in range(0, Nomegap):
        xplotN.append(d[i,0])
        yplotN0.append(d[i,1])
        yplotN0n.append(dn[i,1])

    Ynorm0 = max(yplotN0)
    Ynmax = Ynorm0

    m0 = 0
    m1 = 0
    m2 = 0
    m3 = 0

    for i in range(0, Nomegap):

        m0 = m0 + xplotN[i]*yplotN0[i]

    m0 = m0/sum(yplotN0)
    s0 = 0

    for i in range(0, Nomegap):
        s0 = s0 + yplotN0[i]*(xplotN[i]-m0)**2

    s0 = sqrt(s0/sum(yplotN0))
#-----------------------------
    sump = 0.0
    sump2 = 0.0
    energy = 0.0
    energy2 = 0.0
    for i in range(0,len(p_list)):
        sump += p_list[i]
        sump2 += p2_list[i]
        energy += e_list[i]
        energy2 += e2_list[i]
    sump /= len(p_list)
    sump2 /= len(p_list)
    energy /= len(p_list)
    energy2 /= len(p_list)
    var_p = sump2 - sump**2
    var_e = energy2 - energy**2
    sigmaEn = sqrt(var_e)/energy
    sigmap_p = sqrt(var_p)/sump
    print 'Expected Value (E0);', energy
    print 'Expected Value (momentum):', sump
    print 'Sigma (energy):', sqrt(var_e)
    print 'Sigma (momentum):', sqrt(var_p)
    print 'sigma_E/E_e', sigmaEn

#-----------------------------
    beta = sqrt(1.0-1.0/(gamma**2))
    om0 = 2.0*pi*c/L
    mt = om0*(hbar*eVtoJ)/(gamma*me*(c**2))
    om_max = om0*(1+beta)/(1-beta+2*mt)
    om_min = om0*(1+beta)/(1-beta*cos(R/Lc)+mt*(1+cos(R/Lc)))
    om_mid = 0.5*(om_max+om_min)
    sig_aper = (om_max-om_min)/(sqrt(12.0)*om_mid)
    sig_p = 1.0/(2.0*pi*sqrt(2.0)*s)
    pr0 = sqrt((2*sigmaEn)**2 + sig_p**2 + sig_aper**2)
    m_2 = 1
    sigma_tx = ex/betastar
    sigma_ty = ey/betastar
    x1 = ((psi ** 2)/sqrt(12)/(1 + (psi ** 2)/2))
    x1p = (psi ** 2)/sqrt(12)/(1 + (psi ** 2))
    x2 = ((2 + X)/(1 + X+ gamma**2*Aper**2)) * sigmap_p * (1-sigma_tx**2/2-sigma_ty**2/2)
    x3 = (1/(1+X+gamma**2*Aper**2)) * sig_p
    x4 = m_2 * L / (2 * np.pi) / (El / hbar)
    x5 = (a0 ** 2)/3/(1 + (a0 ** 2)/2)
    print "gammathetasquared", gamma**2*Aper**2
    pr1 = sqrt((x1 ** 2) + (x2 ** 2) + (x3 ** 2) + (x4 ** 4) + (x5 ** 2))
    pr2 = sqrt((x1p ** 2) + (x2 ** 2) + (x3 ** 2) + (x4 ** 4) + (x5 ** 2))
    #pr0 = sqrt((sig_aper ** 2) + (x2 ** 2) + (x3 ** 2) + (x4 ** 4) + (x5 ** 2))
    print "pr1 =", pr1
    print sig_aper, x1
    print x1, x2, x3, x4, x5
#-----------------------------
    f = open('width.txt','w')
    #f2 = open('width2.txt', 'w')
    line = r'%15.8e  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e  %15.8e' % (sg, sigma_px, sigma_py, sigma_pz, pz, sqrt(var_p), sigmaEn, R, s0/m0, pr1)
    #line2 = r'%15.8e  %15.8e  %15.8e  %15.8e' % (sg, sigmaEn, s0/m0, pr1)
    f.write(line)
    #f2.write(line2)
    print line
    #print line2
    f.write('\n')
    #f2.write('\n')
    f.close()
    #f2.close()
#-----------------------------
#--------
# Plot
#--------
#    print "Generating figure..."
    p1 = plt.subplot(111)
    p1.plot(xplotN,yplotN0,'b',lw=1.5)
    #p1.set_title('Sun et al. 2009: Figure 4b')
    p1.set_xlabel(r"$E'$ [MeV]", size=16)
    #p1.set_ylabel('Scattered photon energy density dE/d\omega [a.u.]')
    p1.set_ylabel(r'$dN/d\omega \times 10^{-4}$ [s per electron]', size=16)
    #p1.grid()
    p1.set_xlim([start, start+Range])
    p1.set_ylim([0, 1.04*Ynmax])

    plt.subplots_adjust(left=0.14,bottom=0.14,right=0.9,top=0.9
                       ,wspace=0.5,hspace=0.35)
    plt.tick_params(axis='both', labelsize=14, pad=7)
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig('Sun_et_al_2009_Fig_4b_sg=' + str(sg) + '_sig='+ str(sigmaEn) + '_R=' + str(R) + 'emit=' + str(ex) + '.eps',format='eps')
    plt.close()

#---------------
    p2 = plt.subplot(111)
    p2.plot(xplotN,yplotN0n,'b',lw=1.5)
    #p2.set_title('Sun et al. 2009: Figure 4b')
    p2.set_xlabel(r"$E'$ [MeV]", size=16)
    #p2.set_ylabel('Scattered photon energy density dE/d\omega [a.u.]')
    p2.set_ylabel(r'$dN/d\omega$ [a.u.]', size=16)
    #p2.grid()
    p2.set_xlim([start, start+Range])
    p2.set_ylim([0, 1.04])

    plt.subplots_adjust(left=0.14,bottom=0.14,right=0.9,top=0.9
                       ,wspace=0.5,hspace=0.35)
    plt.tick_params(axis='both', labelsize=14, pad=7)
    figure = plt.gcf()
    figure.set_size_inches(8, 6)
    plt.savefig('Sun_et_al_2009_Fig_4b_sg='+str(sg)+'_sig='+str(sigmaEn)+'_' + 'R=' + str(R) + 'emit=' + str(ex) + '_norm.eps',format='eps')
