# -----------------------------------------------------------------------------
# Implementation of the "intensive" integration algorithm used to calculate
# spectra over 3D cross-section. Use to perform electric field calculations.
# -----------------------------------------------------------------------------

cdef extern from "math.h":
    double sin(double x)
    double cos(double x)
    double sqrt(double x)
    double atan(double x)
    double exp(double x)
    double log(double x)
    double fabs(double x)

cdef extern from "string.h":
    char *strcat(char *dest, const char *src)
    void *memset(void *str, int c, size_t n)

cdef extern from "stdio.h":
    int sprintf(char *str, const char *format, ...)

import array
import random
#cimport numpy as np
import numpy as np
#cimport cython

from libc.stdio cimport FILE, fopen, fprintf, fclose
from libc.stdlib cimport malloc, free

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

cdef double alpha_x(double aPeak, double xi,double sign,int iTypeEnv):
    cdef double xis2 = 0.0
    cdef int n = 20
    cdef double value
    if (iTypeEnv == 1):
        xis2 = (xi ** 2)/(2.0 * (sign ** 2))
        value = aPeak * (e ** (-xis2))
    elif (iTypeEnv == 2):
        if ((xi > 0.0) & (xi < n)):
            value = aPeak/1.176
        else:
            value = 0.0
    else:
        xis2 = (xi ** 2)/(2.0 * (sign **2))
        value = aPeak * (e ** (-xis2))
    return value

cdef double fmod (double xi, double alphax, double aPeak, int modType, double sign, int iTypeEnv):
    cdef double fmod = 1.0
    cdef double z = 0.0
    cdef double cons1
    cdef double x
    cdef double alphax_1
    if (modType == 1):
        fmod = 1.0
    elif (modType == 2):
        fmod = (2.0/3.0)*(1.0 + 0.5 * ((alphax/aPeak) ** 2))
    elif (modType == 3):
        fmod = (1.0 + 0.5 * (alphax **2))/(1.0 + 0.5 * (aPeak ** 2))
    elif (modType == 4):
        if (fabs(xi) < 0.01):
            fmod = 1.0 - 2.0 * (aPeak ** 2) * (xi ** 2)/(3.0 * (2.0 + aPeak ** 2))
        else:
            z = xi + sqrt(pi/2.0) * (aPeak ** 2) * erf(sqrt(2.0) * xi)/4.0
            fmod = z/(xi * (1.0 + (aPeak ** 2)/2.0))
    elif (modType == 5):
        fmod = 1.0 + 0.5 * (alphax ** 2)
    elif (modType == 6):
        fmod = 1.0 + 0.5 * ((alphax/aPeak) ** 2)
    elif (modType == 7):
        fmod = 1.5
    elif (modType == 8):
        fmod = 0.5
    elif (modType == 9):
        fmod = 1.0 + 0.5 * (aPeak ** 2)
    elif (modType == 10):
        fmod = sqrt(1.0 + 0.5 * ((alphax/aPeak) ** 2))
    elif (modType == 11):
        fmod = (1.0 + 0.5 * (alphax ** 2))/(1.0 + 0.5 * (aPeak ** 2))
    elif (modType == 20):
        fmod = sqrt(1.0 + 0.5 * (alphax **2))
    elif (modType == 36):
        cons1 = 1.0/(1.0 + 0.5 * (aPeak ** 2))
        if (xi == 0):
            fmod = 1.0
        else:
            fmod = cons1 + (cons1 * (aPeak ** 2) * sqrt(pi) * sign/(4.0 * xi)) * erf(xi/sign)
    elif (modType == 37):
        fmod = 2.0 / 3.0 * (sqrt(pi) * sign / (4.0 * xi)) * erf(xi/sign)
        if (xi == 0):
            fmod = 1
    elif (modType == 38):
        x = xi / 2.0
        alphax_1 = alpha_x(aPeak,xi,sign,iTypeEnv)
        fmod = 2.0 / 3.0 * (1.0 + 0.5 * ((alphax_1/aPeak) ** 2))
    else:
        fmod = 1.0
    return fmod

cdef double erf (double x):
    cdef double a1 = 0.254829592
    cdef double a2 = -0.284496736
    cdef double a3 = 1.421413741
    cdef double a4 = -1.453152027
    cdef double a5 = 1.061405429
    cdef double p = 0.3275911
    cdef int sign = 1
    if (x < 0):
        sign = -1
    x = fabs(x)
    cdef double t = 1.0/(1.0+p*x)
    cdef double y = 1.0 - (((((a5*t + a4)*t)+a3)*t + a2)*t +a1)*t*exp(-x*x)
    return sign*y;

cdef void fm(double gamma, double aPeak, double omega_omega0_min, double omega_omega0_max, int nout, int ntot, int iTypeEnv, int modType, double theta, double phi, double sign, str filename):

    cdef int r = 0
    if (ntot > nout):
        r = ntot
    else:
        r = nout
    cdef double[:] dx = np.empty(r, dtype=np.double)
    cdef double[:] dz = np.empty(r, dtype=np.double)
    cdef double[:] ax = np.empty(r, dtype=np.double)
    cdef double[:] axi = np.empty(r, dtype=np.double)
    cdef double[:] axi2 = np.empty(r, dtype=np.double)
    cdef double[:] axp = np.empty(r, dtype=np.double)
    cdef double[:] n1x = np.empty(r, dtype=np.double)
    cdef double[:] n2x = np.empty(r, dtype=np.double)
    cdef double[:] n1z = np.empty(r, dtype=np.double)
    cdef double[:] n2z = np.empty(r, dtype=np.double)

    cdef double betaz = sqrt(1.0 - (1.0 / (gamma ** 2)))
    cdef double c1 = gamma * (1.0 + betaz)
    cdef double c2 = (1.0 - betaz * cos(theta)) / (1 + betaz)
    cdef double c3 = sin(theta) * cos(phi) / c1
    cdef double c4 = (1.0 + cos(theta)) / (2.0 * (c1 ** 2))
    cdef double c5 = 1.0 / (sqrt(2.0 * pi) * (1.0 - (e ** (-2.0 * pi * pi * sign * sign))))
    cdef double c6 = (cos(theta) - betaz)/(1 - betaz*cos(theta))
    cdef double omega0 = 2.0 * pi * c1 * c1

    cdef double nLambda = nout - 1
    cdef double xip = <double>ntot
    xip = xip / -2 / nLambda
    cdef double alphax = alpha_x(aPeak, xip, sign, iTypeEnv)
    cdef double f = fmod(xip, alphax, aPeak, modType, sign, iTypeEnv)
    ax[0] = alphax * cos(2.0 * pi * xip * f)
    axi[0] = 0.0
    axi2[0] = 0.0
    cdef int i = 0
    for i in range(0, ntot - 1):
        xip = (i - <double>ntot / 2) / nLambda
        alphax = alpha_x(aPeak, xip, sign, iTypeEnv)
        f = fmod(xip, alphax, aPeak, modType, sign, iTypeEnv)
        ax[i+1] = alphax*cos(2.0*pi*f*xip)
        alphax = alpha_x(aPeak, xip, sign, iTypeEnv)
        f = fmod(xip - 0.5, alphax, aPeak, modType, sign, iTypeEnv)
        axp[i] = alphax*cos(2.0*pi*f*(xip-0.5))
        axi[i+1] = axi[i] + (ax[i] + ax[i+1] + 4.0 * (axp[i]**2))/6.0
        axi2[i+1] = axi2[i] + ((ax[i]**2) + (ax[i+1]**2) + 4.0*(axp[i]**2))/6.0

    cdef double d_omega = (omega_omega0_max - omega_omega0_min) / <double>nout
    cdef int j = 0
    cdef double arg = 0.0
    cdef double omega = 0.0
    cdef int k
    cdef double om = 0.0
    cdef double Esigma = 0.0
    cdef double Epi = 0.0
    cdef double num = 0.0
    cdef FILE *spectrumFile
    spectrumFile = fopen(filename, 'w')
    #cdef char* writeString = <char*>malloc(sizeof(char) * 180)
    #cdef char* c_string
    #cdef char[300] w1
    #cdef char[30] w2
    #cdef char[30] w3
    #cdef char[30] w4
    #cdef char[30] w5
    #cdef char[30] w6
    #cdef char[30] w7
    #cdef char[30] w8
    #cdef char[30] w9
    for j in range(0, nout):
        dx[j] = 0.0
        dz[j] = 0.0
        n1x[j] = 0.0
        n2x[j] = 0.0
        n1z[j] = 0.0
        n2z[j] = 0.0
        omega = omega0 * (omega_omega0_min + j * d_omega)
        for k in range(0, ntot):
            xi = float(k - ntot / 2 - 1)
            arg = omega * (xi * c2 - axi[k] * c3 + axi2[k] * c4) / nLambda
            n1x[j] += ax[k] * (cos(arg)) / nLambda
            n2x[j] += ax[k] * (sin(arg)) / nLambda
            n1z[j] += (ax[k]**2) * (cos(arg)) / nLambda
            n2z[j] += (ax[k]**2) * (sin(arg)) / nLambda
        dx[j] = sqrt(n1x[j]**2 + n2x[j] **2) / c1
        dz[j] = sqrt(n1z[j]**2 + n2z[j] **2) /(2*c1*c1*c2)
        om = omega/omega0
        Esigma = (om*dx[j]*sin(phi))**2
        Epi = (om**2)*(dx[j]*c6*cos(phi) + dz[j]*sin(theta))**2
        num = Epi + Esigma
        fprintf(spectrumFile,"%e  %e  %e  %e  %e  %e  %e  %e  %e\n", aPeak, theta, phi, om, dx[j], dz[j], Esigma, Epi, num)
    fclose(spectrumFile)

def function(a, b, c, d, e, f, g, h, i, j, k, filename):
    fm(a,b,c,d,e,f,g,h,i,j,k, filename)
    return 1
   