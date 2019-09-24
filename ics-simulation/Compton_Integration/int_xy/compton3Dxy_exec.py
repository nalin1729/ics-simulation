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

def spectrum1(Nx,Ny,Nt,Np,amin,amax,a,theta_max,gamma,sign,sigma_p,sigma_e_x,sigma_e_y):
    dt = (1-math.cos(theta_max))/(Nt-1)
    dp = 2*pi/(Np-1)
    x_min =-3*sigma_e_x
    x_max = 3*sigma_e_x
    y_min =-3*sigma_e_y
    y_max = 3*sigma_e_y
    a_max = a[0]
    dx = (x_max - x_min)/(Nx-1)
    dy = (y_max - y_min)/(Ny-1)
    cpu = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cpu)
    print "Running on", cpu, "cores."
    results = []
    for nx in range(0,Nx):
        x = x_min + nx*dx
        for ny in range(0,Ny):
            y = y_min + ny*dy
            aval = a_max*math.exp(-((x**2)/(2*(sigma_p**2)))
                                  -((y**2)/(2*(sigma_p**2))))
            for t in range(0,Nt):
                theta = math.acos(1-t*dt)
                for p in range(0,Np):
                    phi = p*dp
                    filename = get_filename(nx,ny,t,p)
                    temp = (gamma,aval,a[1],a[2],a[3],a[4],a[5],a[6],theta,phi
                           ,sign,sigma_p,sigma_e_x,sigma_e_y,filename)
                    pool.apply_async(compton3D.function,args=temp,)
    pool.close()
    pool.join()

def n_xy(x,y,sig_x,sig_y):
    fun = (1.0/(2*pi*sig_x*sig_y))*math.exp(-((x**2)/(2*(sig_x**2))) 
                                            -((y**2)/(2*(sig_y**2))))
    return fun

def get_filename(nx,ny,t,p):
    filename = 'spectrum.0x'
    if (nx < 10):
       char = str(chr(nx + 48))
       filename = filename[0:10] + char
    elif (nx >= 10):
       char1 = str(chr((nx / 10) + 48))
       char2 = str(chr((nx % 10) + 48))
       filename = filename[0:9] + char1 + char2

    filename = filename[0:11] + '.0t'
    if (ny < 10):
       char = str(chr(ny + 48))
       filename = filename[0:13] + char
    elif (ny >= 10):
       char1 = str(chr((ny / 10) + 48))
       char2 = str(chr((ny % 10) + 48))
       filename = filename[0:12] + char1 + char2

    filename = filename[0:14] + '.0p'
    if (t < 10):
       char = str(chr(t + 48))
       filename = filename[0:16] + char
    elif (t >= 10):
       char1 = str(chr((t / 10) + 48))
       char2 = str(chr((t % 10) + 48))
       filename = filename[0:15] + char1 + char2

    filename = filename[0:17] + '.0p'
    if (p < 10):
       char = str(chr(p + 48))
       filename = filename[0:19] + char
    elif (p >= 10):
       char1 = str(chr((p / 10) + 48))
       char2 = str(chr((p % 10) + 48))
       filename = filename[0:18] + char1 + char2
    return filename

#-------------------------
# Main program
#-------------------------
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
sigma_e_x = args[1]
sigma_e_y = args[2]

sign = args[3]
sigma_p = args[4]
a0 = args[5]
iTypeEnv = int(args[6])
modType = int(args[7])

theta_max = args[8]
wtilde_min = args[9]
wtilde_max = args[10]

Nout = int(args[11])
Ntot = int(args[12])
Nt = int(args[13])
Np = int(args[14])
Nx = int(args[15])
Ny = int(args[16])
# -------------------------
print "***********************************"
print "****   Simulation Parameters   ****"
print "***********************************"
print "Ee        =  ", Ee
print "sigma_e_x =  ", sigma_e_x
print "sigma_e_y =  ", sigma_e_y
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
print "Nt        =  ", Nt
print "Np        =  ", Np
print "Nx        =  ", Nx
print "Ny        =  ", Ny
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
    spectrum1(Nx,Ny,Nt,Np,amin,amax,a,theta_max,gamma,sign,sigma_p,sigma_e_x,sigma_e_y)

X = [0 for n in range(0, Nout)]
Y = [0 for n in range(0, Nout)]
Y_oax = [0 for n in range(0, Nout)]
print "Nout = ", Nout
dt = (1 - math.cos(theta_max)) / (Nt - 1)
x_min =-3*sigma_e_x
x_max = 3*sigma_e_x
y_min =-3*sigma_e_y
y_max = 3*sigma_e_y
dx = (x_max - x_min)/(Nx-1)
dy = (y_max - y_min)/(Ny-1)
for nx in range(0,Nx):
    x = x_min + nx*dx
    for ny in range(0,Ny):
        y = y_min + ny*dy
        Ftp = [0 for n in range(0,Nout)]
        Ftp_oax = [0 for n in range(0,Nout)]
        for t in range(0,Nt):
            for p in range(0,Np):
                filename = get_filename(nx,ny,t,p)
                d = np.loadtxt(filename)
                theta = math.acos(1-t*dt)
                Ftp[:] += d[:,8]/(Np*Nt)
                if (t == 0) and (p == 0):
                    Ftp_oax[:] += d[:,8]
        Nomegap = len(d[:,0])
        if (nx == 0) or (nx == Nx-1):
            if (ny == 0) or (ny == Ny-1):
                for k in range(0, Nomegap):
                    Y[k] += 0.25*Ftp[k]*n_xy(x,y,sigma_e_x,sigma_e_y)*dx*dy
                    Y_oax[k] += 0.25*Ftp_oax[k]*n_xy(x,y,sigma_e_x,sigma_e_y)*dx*dy
            else:
                for k in range(0, Nomegap):
                    Y[k] += 0.5*Ftp[k]*n_xy(x,y,sigma_e_x,sigma_e_y)*dx*dy
                    Y_oax[k] += 0.5*Ftp_oax[k]*n_xy(x,y,sigma_e_x,sigma_e_y)*dx*dy
        else:
            if (ny == 0) or (ny == Ny-1):
                for k in range(0, Nomegap):
                    Y[k] += 0.5*Ftp[k]*n_xy(x,y,sigma_e_x,sigma_e_y)*dx*dy
                    Y_oax[k] += 0.5*Ftp_oax[k]*n_xy(x,y,sigma_e_x,sigma_e_y)*dx*dy
            else:
                for k in range(0, Nomegap):
                    Y[k] += Ftp[k]*n_xy(x,y,sigma_e_x,sigma_e_y)*dx*dy
                    Y_oax[k] += Ftp_oax[k]*n_xy(x,y,sigma_e_x,sigma_e_y)*dx*dy
    X[:] = d[:,3]


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
