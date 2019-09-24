import multiprocessing
import scipy
import math
import sys
import numpy as np
import compton3D

pi = math.pi
e = math.e
i, j, = 0,0
cpu = multiprocessing.cpu_count()
processList = []

def spectrum1(Na,Nt,Np,amin,amax,a,theta_max,gamma,sign):
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
                temp = (gamma,amin+da*n,a[1],a[2],a[3],a[4],a[5],a[6],theta,phi,sign,filename)
                pool.apply_async(compton3D.function,args=temp,)
    pool.close()
    pool.join()

def n_pdf(a,a0,sigma_e,sigma_p,Ne,lasType):
    if (lasType == 1):      # Gaussian laser pulse
       if (a == 0):
           nfunc = 0.0
       else:
           nfunc = (Ne/a0)*((sigma_p/sigma_e)**2)*((a/a0)**((sigma_p/sigma_e)**2-1))
    else:                   # flat-top laser pulse
       nfunc = 1.0
    return nfunc

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

arg_file = open("config.in", "r")
args = []
for line in arg_file:
    i = 0
    while (line[i:i + 1] != " "):
        i += 1
    num = float(line[0:i])
    args.append(num)
# -------------------------
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
# -------------------------
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
# -------------------------
mc2 = 511e3
Ne = 1
# -------------------------
amin = 0.0
amax = a0
gamma = Ee / mc2
a = [0 for x in range(0, 7)]
a[0] = a0
a[1] = wtilde_min
a[2] = wtilde_max
a[3] = Nout
a[4] = Ntot
a[5] = iTypeEnv
a[6] = modType

plotOnly = int(sys.argv[1])  # comm. lin. arg: 1: plot only, <> compute & plot
if (plotOnly <> 1):
    spectrum1(Na, Nt, Np, amin, amax, a, theta_max,gamma,sign)

X = [0 for x in range(0, Nout)]
Y = [0 for x in range(0, Nout)]
Y_oax = [0 for x in range(0, Nout)]
print "Nout = ", Nout
#ne_out = open('ne_pdf.txt', 'w')
dt = (1 - math.cos(theta_max)) / (Nt - 1)
da = a0/(Na-1)
for n in range(0, Na):
    Ftp = [0 for x in range(0, Nout)]
    Ftp_oax = [0 for x in range(0, Nout)]
    for t in range(0, Nt):
        for p in range(0, Np):
            filename = get_filename(n, t, p)
            d = np.loadtxt(filename)
#            if (n == 0) and (t == 0) and (p == 0):
#                line = r'%15.8e  %15.8e' % (d[n, 3], n_pdf(d[n, 3], a0, sigma_e, sigma_p, Ne, 1))
#                ne_out.write(line)
#                ne_out.write('\n')
            theta = math.acos(1 - t * dt)
            Ftp[:] += d[:, 8] / (Np * Nt)
            if (t == 0) and (p == 0):
                Ftp_oax[:] += d[:, 8]

    X[:] = d[:, 3]
    Nomegap = len(d[:, 0])
    for k in range(0, Nomegap):
        if (n == 0) or (n == Na-1):
            Y[k] += 0.5*Ftp[k]*n_pdf(d[k,0], a0, sigma_e, sigma_p, Ne, 1)*da
            Y_oax[k] += 0.5*Ftp_oax[k]*n_pdf(d[k,0], a0, sigma_e, sigma_p, Ne, 1)*da
        else:
            Y[k] += Ftp[k]*n_pdf(d[k,0], a0, sigma_e, sigma_p, Ne, 1)*da
            Y_oax[k] += Ftp_oax[k]*n_pdf(d[k,0], a0, sigma_e, sigma_p, Ne, 1)*da
#ne_out.close()

f = open('all_spectrum.txt', 'w')
for k in range(0, Nomegap):
    line = r'%15.8e  %15.8e  %15.8e' % (X[k], Y[k], Y_oax[k])
    f.write(line)
    f.write('\n')
f.close()

data = np.loadtxt('all_spectrum.txt')
f2 = open('moments.txt', 'w')

N = len(d[:,0])
fmax  = -999.0
xbar  = 0
x2bar = 0
fnorm = 0
for j in range(0, N):
    if (data[j,1]>fmax):
        fmax  =  data[j,1]
        xloc  =  data[j,0]
    xbar  = xbar + data[j,1]*data[j,0]
    x2bar = x2bar + data[j,1]*(data[j,0]**2)
    fnorm = fnorm + data[j,1]
xbar = xbar/fnorm
x2bar = x2bar/fnorm
width = (x2bar - xbar**2)**0.5
line2 = r'%15.8e  %15.8e  %15.8e  %15.8e  %15.8e' % (a0, theta_max, xbar, xloc, width)
f2.write(line2)
f2.close()

