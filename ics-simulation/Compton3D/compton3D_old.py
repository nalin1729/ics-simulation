# -----------------------------------------------------------------------------
# Implementation of the non-optimized integration algorithm used to calculate
# spectra. Use to perform electric field calculations.
# -----------------------------------------------------------------------------

import multiprocessing
import scipy
import math
import sys
import numpy as np

gamma, betaz, c1, c2, c3, c4, c5, nLambda, alphax = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
sign, aPeak, arg = 0.0, 0.0, 0.0
omega0, d_omega = 0.0, 0.0
f, xi, xip, lambda0, obj1, obj2, omega = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
pi = math.pi
e = math.e
dx, dz, ax, axi, axi2, axp = [], [], [], [], [], []
i, j, modType, iTypeEnv, simID = 0, 0, 0, 0, 0
cpu = multiprocessing.cpu_count()
processList = []
results = []

def alpha_x(aPeak,xi,sign,iTypeEnv):
    xis2 = 0.0
    n = 20
    if (iTypeEnv == 1):
        xis2 = (xi ** 2)/(2.0 * (sign ** 2))
        value = aPeak * (e ** (-xis2))
    elif (iTypeEnv) == 2:
        if ((xi > 0.0) & (xi < n)):
            value = aPeak/1.176
        else:
            value = 0.0
    else:
        xis2 = (xi ** 2)/(2.0 * (sign **2))
        value = aPeak * (e ** (-xis2))
    return value

def fmod (xi,alphax,aPeak,modType,sign,iTypeEnv):
    fmod = 1.0
    z = 0.0
    if (modType == 1):
        fmod = 1.0
    elif (modType == 2):
        fmod = (2.0/3.0)*(1.0 + 0.5 * ((alphax/aPeak) ** 2))
    elif (modType == 3):
        fmod = (1.0 + 0.5 * (alphax **2))/(1.0 + 0.5 * (aPeak ** 2))
    elif (modType == 4):
        if (math.fabs(xi) < 0.01):
            fmod = 1.0 - 2.0 * (aPeak ** 2) * (xi ** 2)/(3.0 * (2.0 + aPeak ** 2))
        else:
            z = xi + math.sqrt(pi/2.0) * (aPeak ** 2) * math.erf(math.sqrt(2.0) * xi)/4.0
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
        fmod = math.sqrt(1.0 + 0.5 * ((alphax/aPeak) ** 2))
    elif (modType == 11):
        fmod = (1.0 + 0.5 * (alphax ** 2))/(1.0 + 0.5 * (aPeak ** 2))
    elif (modType == 20):
        fmod = math.sqrt(1.0 + 0.5 * (alphax **2))
    elif (modType == 36):
        cons1 = 1.0/(1.0 + 0.5 * (aPeak ** 2))
        if (xi == 0):
            fmod = 1.0
        else:
            fmod = cons1 + (cons1 * (aPeak ** 2) * math.sqrt(pi) * sign/(4.0 * xi)) * math.erf(xi/sign)
    elif (modType == 37):
        fmod = 2.0 / 3.0 * (math.sqrt(pi) * sign / (4.0 * xi)) * math.erf(xi/sign)
        if (xi == 0):
            fmod = 1
    elif (modType == 38):
        x = xi / 2.0
        alphax_1 = alpha_x(aPeak,xi,sign,iTypeEnv)
        fmod = 2.0 / 3.0 * (1.0 + 0.5 * ((alphax_1/aPeak) ** 2))
    else:
        fmod = 1.0
    return fmod

def fm(aPeak, omega_omega0_min, omega_omega0_max, nout, ntot, iTypeEnv, modType, theta, phi, filename):
    global gamma, betaz, c1, c2, c3, c4, c5, nLambda, alphax, sign, arg, omega0, d_omega
    global f, xi, xip, lambda0, obj1, obj2, omega, pi, e
    global dx, dz, ax, axi, axi2, axp, i, j
    global sign, gamma

    betaz = math.sqrt(1.0 - (1.0 / (gamma ** 2)))
    c1 = gamma * (1.0 + betaz)
    c2 = (1.0 - betaz * math.cos(theta)) / (1 + betaz)
    c3 = math.sin(theta) * math.cos(phi) / c1
    c4 = (1.0 + math.cos(theta)) / (2.0 * (c1 ** 2))
    c5 = 1.0 / (math.sqrt(2.0 * pi) * (1.0 - (e ** (-2.0 * pi * pi * sign * sign))))
    c6 = (math.cos(theta) - betaz)/(1 - betaz*math.cos(theta))
    omega0 = 2.0 * pi * c1 * c1

    if (ntot > nout):
        for a in range(0, ntot):
            ax.append(0.0)
            axi.append(0.0)
            axi2.append(0.0)
            axp.append(0.0)
            dx.append(0.0)
            dz.append(0.0)
    else:
        for a in range(0, nout):
            ax.append(0.0)
            axi.append(0.0)
            axi2.append(0.0)
            axp.append(0.0)
            dx.append(0.0)
            dz.append(0.0)

# Inner integral
    open('fmod.out', 'w')
    nLambda = float(nout - 1)
    xip = float(-ntot / 2) / nLambda
    alphax = alpha_x(aPeak, xip, sign, iTypeEnv)
    f = fmod(xip, alphax, aPeak, modType, sign, iTypeEnv)
    ax[0] = alphax * math.cos(2.0 * pi * xip * f)
    axi[0] = 0.0
    axi2[0] = 0.0
    for i in range(0, ntot - 1):
        xip = float(i - ntot / 2) / nLambda
        alphax = alpha_x(aPeak, xip, sign, iTypeEnv)
        f = fmod(xip, alphax, aPeak, modType, sign, iTypeEnv)
        ax[i+1] = alphax*math.cos(2.0*pi*f*xip)
        alphax = alpha_x(aPeak, xip, sign, iTypeEnv)
        f = fmod(xip - 0.5, alphax, aPeak, modType, sign, iTypeEnv)
        axp[i] = alphax*math.cos(2.0*pi*f*(xip-0.5))
        axi[i+1] = axi[i] + (ax[i] + ax[i+1] + 4.0 * (axp[i]**2))/6.0
        axi2[i+1] = axi2[i] + ((ax[i]**2) + (ax[i+1]**2) + 4.0*(axp[i]**2))/6.0

# Outer integral
    spectrumFile = open(filename, 'w')
    d_omega = (omega_omega0_max - omega_omega0_min) / float(nout)
#    for j in range(0, nout - 1):
    for j in range(0, nout):
        dx[j] = complex(0.0, 0.0)
        dz[j] = complex(0.0, 0.0)
        omega = omega0 * (omega_omega0_min + j * d_omega)
        for k in range(0, ntot):
            xi = float(k - ntot / 2 - 1)
            arg = omega * (xi * c2 - axi[k] * c3 + axi2[k] * c4) / nLambda
            dx[j] += ax[k] * (e**(complex(0.0, 1.0) * arg)) / nLambda
            dz[j] += (ax[k]**2) * (e**(complex(0.0, 1.0) * arg)) / nLambda
        dx[j] = abs(dx[j]/c1)
        dz[j] = abs(dz[j]/(2*c1*c1*c2))
        om = omega/omega0
        Esigma = (om*dx[j]*math.sin(phi))**2
        Epi = (om**2)*(dx[j]*c6*math.cos(phi) + dz[j]*math.sin(theta))**2
        writeString = str(aPeak) + ' ' + str(theta) + ' ' + str(phi) + ' ' + str(om) + ' ' + str(dx[j]) + ' ' + str(dz[j]) + ' ' + str(Esigma) + ' ' + str(Epi) + ' ' + str(Esigma+Epi) + '\n'
        spectrumFile.write(writeString)
    spectrumFile.close()

def logResult(result):
    results.append(result) 

def get_filename(n,t,p):
    filename = 'spectrum.0x'
    if (n < 10):
       char = str(chr(n + 48))
       filename = filename[0:10] + char
    elif (n >= 10):
       char1 = str(chr((n / 10) + 48))
       char2 = str(chr((n % 10) + 48))
       filename = filename[0:9] + char1 + char2

    filename = filename[0:11] + '.0t'
    if (t < 10):
       char = str(chr(t + 48))
       filename = filename[0:13] + char
    elif (t >= 10):
       char1 = str(chr((t / 10) + 48))
       char2 = str(chr((t % 10) + 48))
       filename = filename[0:12] + char1 + char2

    filename = filename[0:14] + '.0p'
    if (p < 10):
       char = str(chr(p + 48))
       filename = filename[0:16] + char
    elif (p >= 10):
       char1 = str(chr((p / 10) + 48))
       char2 = str(chr((p % 10) + 48))
       filename = filename[0:15] + char1 + char2
    return filename

def spectrum1(Na,Nt,Np,amin,amax,a,theta_max):
    da = (amax - amin)/(Na-1)
    dt = (1-math.cos(theta_max))/(Nt-1)
    dp = 2*math.pi/(Np-1)
    cpu = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu)
    print "Running on", cpu, "cores."
    results = []
    for n in range(0,Na):
        for t in range(0,Nt):
            theta = math.acos(1-t*dt)
            for p in range(0,Np):
                phi = p*dp
                filename = get_filename(n,t,p)
                temp = (amin+da*n,a[1],a[2],a[3],a[4],a[5],a[6],theta,phi,filename)
                pool.apply_async(fm,args=temp,callback=logResult)
    pool.close()
    pool.join()

#    Yall = [0 for x in range (0, Nout)]
#    for i in range(0, Nout):
#        for n in range(0, Na):
#            Yall[i] += results[n][i]

def n_pdf(a,a0,sigma_e,sigma_p,Ne,lasType):
    if (lasType == 1):      # Gaussian laser pulse
       if (a == 0):
           nfunc = 0.0
       else:
           nfunc = (Ne/a0)*((sigma_p/sigma_e)**2)*((a/a0)**((sigma_p/sigma_e)**2-1)) 
    else:                   # flat-top laser pulse
       nfunc = 1.0
    return nfunc

#-------------------------
# Main program
#-------------------------
arg_file = open("config.in", "r")
args = []
for line in arg_file:
    i = 0
    while (line[i:i+1] != " "):
        i += 1
    num = float(line[0:i])
    args.append(num)
#-------------------------
Ee = args[0]
sigma_e = args[1]

sign = args[2]
sigma_p = args[3]
a0 = args[4]
iTypeEnv = int(args[5])
modType = int(args[6])

theta_max = args[7]
wtilde_min = args[8]
wtilde_max = args[9]

Nout = int(args[10])
Ntot = int(args[11])
Na = int(args[12])
Nt = int(args[13])
Np = int(args[14])
#-------------------------
print "***********************************"
print "****   Simulation Parameters   ****"
print "***********************************"
print "Ee        =  ", Ee
print "sigma_e   =  ", sigma_e
print "sig       =  ", sign
print "sigma_p   =  ", sigma_p
print "a0        =  ", a0
print "iTypeEnv  =  ", iTypeEnv
print "modType   =  ", modType
print "theta_max =  ", theta_max
print "w_min     =  ", wtilde_min
print "w_max     =  ", wtilde_max
print "Nout      =  ", Nout
print "Ntot      =  ", Ntot
print "Na        =  ", Na
print "Nt        =  ", Nt
print "Np        =  ", Np
print "***********************************"
#-------------------------
mc2 = 511e3
Ne = 1
#-------------------------
amin = 0.0
amax = a0
gamma = Ee/mc2
a = [0 for x in range(0,7)]
a[0] = a0
a[1] = wtilde_min
a[2] = wtilde_max
a[3] = Nout
a[4] = Ntot
a[5] = iTypeEnv
a[6] = modType

plotOnly = int(sys.argv[1])   # comm. lin. arg: 1: plot only, <> compute & plot
if (plotOnly <> 1):
   out = spectrum1(Na,Nt,Np,amin,amax,a,theta_max)

X = [0 for x in range(0,Nout)]
Y = [0 for x in range(0,Nout)]
Y_oax = [0 for x in range(0,Nout)]
print "Nout = ", Nout
ne_out = open('ne_pdf.txt','w')
dt = (1-math.cos(theta_max))/(Nt-1)
for n in range(0,Na):
    Ftp = [0 for x in range(0,Nout)]
    Ftp_oax = [0 for x in range(0,Nout)]
    for t in range(0,Nt):
        for p in range(0,Np):
            filename = get_filename(n,t,p)
            d = np.loadtxt(filename)
            if (n == 0) and (t == 0) and (p == 0):
                line = r'%15.8e  %15.8e' % (d[n,3],n_pdf(d[n,3],a0,sigma_e,sigma_p,Ne,1))
                X[:] = d[:,3]
                ne_out.write(line)
                ne_out.write('\n')
            theta = math.acos(1-t*dt)
            Ftp[:] += d[:,8]/(Np*Nt)
            if (t == 0) and (p == 0):
                Ftp_oax[:] += d[:,8]
    
    Nomegap = len(d[:,0])
    for k in range(0, Nomegap):
        Y[k] += Ftp[k]*n_pdf(d[k,3],a0,sigma_e,sigma_p,Ne,1)
        Y_oax[k] += Ftp_oax[k]*n_pdf(d[k,3],a0,sigma_e,sigma_p,Ne,1)
ne_out.close()

f  = open('all_spectrum.txt','w')
for k in range(0, Nomegap):
    line = r'%15.8e  %15.8e  %15.8e' % (X[k],Y[k],Y_oax[k]) 
    f.write(line)
    f.write('\n')
f.close()
