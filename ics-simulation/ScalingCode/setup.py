from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("scale2.pyx")
)

# def loopParticles(double sigmax, double sigmay, double sigmaz, double pzbar, double dOmegap1, double Aper, double s, int Nomegap1):
#
#     #cdef double array[154]
#     array = [0 for i in range(154)]
#     #cdef double y[150]      # sum of A indexed over Ws2
#     y=[]
#     cdef double p0 = sigmax*random.gauss(0,1)
#
#     cdef double p1 = sigmay*random.gauss(0,1)
#
#     cdef double p2 = sigmaz*random.gauss(0,1) + pzbar
#
#     cdef double pmag1 = sqrt(p0**2+p1**2+p2**2)  # electron momentum magnitude[kg*m/s]
#
#     cdef double pmag12 = p0**2+p1**2+p2**2
#
#     cdef double energy = sqrt(pmag1**2*c**2+me**2)
#
#     cdef double energy2 = energy**2
#
#     cdef double g1 = sqrt(1+(pmag1/(me*c))**2)   # Lorentz factor of electron frame
#
#     cdef double B1 = pmag1/(g1*me*c)             # relativistic speed of electron frame
#
#     cdef double x2
#
#     cdef double Ws2
#
#     cdef double[2] yp
#
#     cdef double yp1
#
#     cdef int x
#
#     #cdef double param1
#     #cdef double cp1
#     #cdef double ep1
#     #cdef double cp2
#     #cdef double cp21
#     #cdef double cp22
#     #cdef double cp23
#     #cdef double cp24
#     #cdef double ep2
#     #cdef double ep3
#     #cdef double ep4
#     #cdef double ep5
#
#     for n in range(Nomegap1):
#
#         x2 = n * dOmegap1 + start
#
#         Ws2 = ((x2 * (10 ** 6)) / hbar)
#
#         def integ(ctheta,phi):
#             return dE(E(W(p0,p1,p2,ctheta,phi,Ws2,pmag1,g1,B1))
#                                             ,DSX(dsx0(p0,p1,p2,ctheta,phi,Ws2
#                                             ,W(p0,p1,p2,ctheta,phi,Ws2
#                                             ,pmag1,g1,B1),pmag1,g1,B1)
#                                             ,dsx1(ctheta,phi,Ws2,g1,B1)
#                                             ,dsx2(p0,p1,p2,ctheta,phi,g1)
#                                             ,dsx3(p0,p1,p2,ctheta,phi,g1)
#                                             ,dsx4(p0,p1,p2,ctheta,phi,g1),g1,B1)
#                                             ,Ws2, W(p0,p1,p2,ctheta,phi,Ws2,pmag1,g1,B1)
#                                             ,dW(p0,p1,p2,ctheta,phi,Ws2,pmag1,g1,B1))
#
#         #cp1 = lambda ctheta,phi: W(p0,p1,p2,ctheta,phi,Ws2,pmag1,g1,B1)
#         #ep1 = E(cp1)
#         #cp2 = lambda ctheta,phi: dsx0(p0,p1,p2,ctheta,phi,Ws2,W(p0,p1,p2,ctheta,phi,Ws2,pmag1,g1,B1),pmag1,g1,B1)
#         #cp21 = lambda ctheta,phi: dsx1(ctheta,phi,Ws2,g1,B1)
#         #cp22 = lambda ctheta,phi: dsx2(p0,p1,p2,ctheta,phi,g1)
#         #cp23 = lambda ctheta,phi: dsx3(p0,p1,p2,ctheta,phi,g1)
#         #cp24 = lambda ctheta,phi: dsx4(p0,p1,p2,ctheta,phi,g1)
#         #ep2 = lambda ctheta,phi: DSX(cp2,cp21,cp22,cp23,cp24,g1,B1)
#         #ep4 = lambda ctheta,phi: W(p0,p1,p2,ctheta,phi,Ws2,pmag1,g1,B1)
#         #ep5 = lambda ctheta,phi: dW(p0,p1,p2,ctheta,phi,Ws2,pmag1,g1,B1)
#
#         #param1 = lambda ctheta, phi: dE(ep1,ep2,Ws2,ep4,ep5)
#
#         yp = dblquad(integ,0,2*pi,lambda ctheta:cos(Aper),lambda ctheta:1.0)
#
#         yp1 = yp[0]
#
#         y[n] += yp1
#         for x in range(Nomegap1):
#             array[x]=y[x]
#         array[Nomegap1] = pmag1
#         array[Nomegap1+1] = pmag12
#         array[Nomegap1+2] = energy
#         array[Nomegap1+3] = energy2
#
#     return array