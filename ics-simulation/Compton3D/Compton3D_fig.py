#================================================================
# PURPOSE: Compute backscattered spectra for the Duke Compton 
# source as shown in Sun et al. 2009 (PR STAB 12, 062801) Fig. 4b
#================================================================
from math import pi,sqrt,cos,sin,atan,e,log,exp
import matplotlib.pyplot as plt
import numpy as np

#-----------------------------
# Read data from a file
#-----------------------------
d  = np.loadtxt('all_moments.txt')
#--------
# Plot
#--------
print "Generating figure..."
p1 = plt.subplot(111)
#p1.scatter(d[:,1],d[:,3],color='red',marker='o',s=50,label='analytic')
p1.plot(d[:,0],d[:,5]*100,c='r',linewidth=1.5)
x_kramer = [0.05337, 0.4691, 0.96067, 1.5534]
y_kramer = [13.175, 11.215, 16.487, 27.568]
x_error = [0.0, 0.0660, 0.14324, 0.2290]
y_error = [1.0145, 0.945, 1.0815, 1.0815]
plt.scatter(x_kramer, y_kramer, color='green',marker='+',s=66,lw=3,label='Kramer 2017')
plt.errorbar(x_kramer, y_kramer, y_error, capsize=0, ls='none', color='black', elinewidth=2, label='Kramer 2017')
p1.scatter(d[:,0],d[:,5]*100,color='blue',marker='+',s=100,lw=3,label='Computed')
p1.set_xlabel(r"$a_0$", size=16)
p1.set_ylabel(r"$\sigma_{E'}/E'$", size=16)
#p1.grid()
#p1.set_xlim([start, start+Range])
p1.set_xlim([0.0, 2.0])
p1.set_ylim([0.0, 30.0])
leg = plt.legend(('Kramer et al.','Computed'),'upper left', shadow=False,scatterpoints=1)
plt.subplots_adjust(left=0.14,bottom=0.14,right=0.9,top=0.9,wspace=0.5,hspace=0.35)
plt.tick_params(axis='both', labelsize=14, pad=7)
p1.legend(loc='bottom right')
figure = plt.gcf()
figure.set_size_inches(8, 6)
plt.savefig('compton3D_fig.eps',format='eps')
plt.show()
