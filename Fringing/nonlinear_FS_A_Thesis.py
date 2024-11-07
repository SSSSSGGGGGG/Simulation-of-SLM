# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
"""In this section, we only simulate the influence of parameters AL and AR on the phase profile keeping other parameters constant, and how they affect the normalized intensities of the 0th order and the 1st orders."""


"""
1. AL and AR are in the range of (1000, 2, 50), and they can be explored seperatedly.
2. M and B are initialled with 1.007 and 0.1. respectively.
3. Some other necessary parameters are defined and initialled: 
    Fill factor a, 
    Roundness A for linear phase funtion,
    Diffraction order m,
    Period p,
    gray level gray (phase value).
"""
AL=[1000,2,50]
AR=AL
M=1.007
B=0.1
a=[0.964,1,0.7]
A=1000
m=[0,1,-1]
p=1
gray=np.arange(0,256,1)
x=np.linspace(-0.5, 0.5,640)  # For linear
x_=np.linspace(0, 0.5,320)   # For nonlinear
ac=[]   # An empty array to save the normalized intensities associated with different fill factors.
# A loop that is for going through different fill factors.
for a_ in range (len(a)):      
    # The linear f(x) in Eq. (8.21).
    f_linear=2*x/p
    # Define a nonlinear f(x) as f_1, which is designed based on f_linear for showing the difference between linear and non linear phase function. 
    f_1=np.arctan(B*(f_linear/2))
    max_index_1 = np.argmax(f_1)
    max_value_1 = f_1[max_index_1]
    f_n=f_1/max_value_1     # Normalized f_1.
    
    # The phase profile which is calculated based on linear f_linear.
    PhaseProfile=np.arctan(A*np.cos(f_linear*np.pi))/(np.pi)
    
    # A nonlinear f(x) which is similar as f_1 but this one is for calculating nonlinear phase profile. Because the phase profile will not be symmetric this is for the negtive/left part of it.
    f_=np.arctan(B*(x_/p))
    max_index_ = np.argmax(f_)
    max_value_ = f_[max_index_]
    f_L=f_/max_value_
    
    # The positive/right part.            
    f_r=np.arctan(B*(-x_/p))
    min_index_r = np.argmin(f_r)
    min_value_r = f_r[min_index_r]
    f_R=f_/min_value_r
    
    Mc=[]   # An empty array to save the normalized intensities associated with different parameter M.
    # A loop that is for going through different parameter AL/AR.
    for n in range(len(AL)):
        # The negtive/left part of phase profile which is calculated based on nonlinear f_L.
        PhaseProfile_L=np.arctan(AL[n]*np.cos(f_L*np.pi))/(np.pi)
        max_index_L = np.argmax(PhaseProfile_L)
        max_value_L = PhaseProfile_L[max_index_L]
        PhaseProfile_L=PhaseProfile_L/max_value_L
        
        # The positive/right part.            
        PhaseProfile_R=np.arctan(AL[n]*np.cos(f_R*np.pi))/(np.pi)
        max_index_R = np.argmax(PhaseProfile_R)
        max_value_R = PhaseProfile_R[max_index_R]
        PhaseProfile_R=PhaseProfile_R/max_value_R
        
        # Combine left and right together.
        PhaseProfile_n=np.concatenate((PhaseProfile_L[::-1], PhaseProfile_R))
        
        # Simulate the binaty amplitude grating resembling the pixelation structure of SLM.
        pixelation=[0] * len(x)
        for i in range(len(x)):
            if x[i]<len(x)/2 and x[i]>a[a_]/2:
                pixelation[i]=0
            elif x[i]>-len(x)/2 and x[i]<-a[a_]/2:
                pixelation[i]=0
            else:
                pixelation[i]=1
                
        # The phase is resacled by parameter M.
        PhaseProfile_M=PhaseProfile_n*M
        
        mc=[]  # An empty array to save the normalized intensities associated with different diffraction orders m.      
        # A loop that is for going through different diffraction orders m.
        for k in range(len(m)): 
            # print(len(m))
            cc=[]    # An empty array to save the coefficients associated with different phase value gray.   
            for j in range(len(gray)):        
                gray_M=M*gray[j]/255    # The phase modulation range, which is manipulated by parameter M.
                phaseProfile_C=PhaseProfile_M*gray_M       
                ex=m[k]*np.pi*f_linear 
                # The integrand of Fourier series: the multiplication of SLM´s pixelation funtion, the nonliear phase profile, and an exponant functon.
                phase=pixelation*np.exp(1j*phaseProfile_C*np.pi)*np.exp(1j*ex)
                cm=np.sum(phase)  
                cc.append(cm)   # The coefficient value for one of the orders we requested based on the Fourier series integral.
            mc.append(cc)   # The coefficient value for all the orders we requested.
        Im=mc*np.conjugate(mc)/len(x)**2   # The normalized intesnities for all the orders we requested.
        Mc.append(Im)   # The normalized intesnities of all the orders for all the requested AL/AR.
    ac.append(Mc)   # The normalized intesnities of all the orders for all the requested AL/AR with different fill factors.

# Showing the normalized intesnities of all the orders for all the requested AL/AR with different fill factors in terms of phase values (gray level).
plt.figure()
plt.plot(gray,ac[0][0][0],label=f"A={AL[0]},a={a[0]},order={m[0]}")
plt.plot(gray,ac[0][0][1],label=f"A={AL[0]},a={a[0]},order={m[1]}")
plt.plot(gray,ac[0][0][2],label=f"A={AL[0]},a={a[0]},order={m[2]}")

plt.plot(gray,ac[0][1][0],label=f"A={AL[1]},a={a[0]},order={m[0]}",linestyle='--')
plt.plot(gray,ac[0][1][1],label=f"A={AL[1]},a={a[0]},order={m[1]}",linestyle='--')
plt.plot(gray,ac[0][1][2],label=f"A={AL[1]},a={a[0]},order={m[2]}",linestyle='--')
ax_n = plt.gca()
ax_n.yaxis.set_minor_locator(MultipleLocator(0.02))
plt.legend(loc="best")
plt.grid(True, linestyle='--')
plt.ylim([0,1])
plt.xlim([0,255])
plt.show()


plt.figure()
plt.plot(gray,ac[0][0][0],label=f"A={AL[0]},a={a[0]},order={m[0]}")
plt.plot(gray,ac[0][0][1],label=f"A={AL[0]},a={a[0]},order={m[1]}")
plt.plot(gray,ac[0][1][0],label=f"A={AL[1]},a={a[0]},order={m[0]}",linestyle=':')
plt.plot(gray,ac[0][1][1],label=f"A={AL[1]},a={a[0]},order={m[1]}",linestyle=':')
plt.plot(gray,ac[0][2][0],label=f"A={AL[2]},a={a[0]},order={m[0]}",linestyle='--')
plt.plot(gray,ac[0][2][1],label=f"A={AL[2]},a={a[0]},order={m[1]}",linestyle='--')
ax_n = plt.gca()
ax_n.yaxis.set_minor_locator(MultipleLocator(0.02))
plt.legend(loc="best")
plt.grid(True, linestyle='--')
plt.ylim([0,1])
plt.xlim([0,255])
plt.show()

plt.figure()
plt.plot(gray,ac[1][0][0],label=f"A={AL[0]},a={a[1]},order={m[0]}")
plt.plot(gray,ac[1][0][1],label=f"A={AL[0]},a={a[1]},order={m[1]}")
plt.plot(gray,ac[1][1][0],label=f"A={AL[1]},a={a[1]},order={m[0]}",linestyle=':')
plt.plot(gray,ac[1][1][1],label=f"A={AL[1]},a={a[1]},order={m[1]}",linestyle=':')

plt.plot(gray,ac[1][2][0],label=f"A={AL[2]},a={a[1]},order={m[0]}",linestyle='--')
plt.plot(gray,ac[1][2][1],label=f"A={AL[2]},a={a[1]},order={m[1]}",linestyle='--')

ax_n = plt.gca()
ax_n.yaxis.set_minor_locator(MultipleLocator(0.02))
plt.legend(loc="best")
plt.grid(True, linestyle='--')
plt.ylim([0,1])
plt.xlim([0,255])
plt.show()

plt.figure()
plt.plot(gray,ac[2][0][0],label=f"A={AL[0]},a={a[2]},order={m[0]}")
plt.plot(gray,ac[2][0][1],label=f"A={AL[0]},a={a[2]},order={m[1]}")
plt.plot(gray,ac[2][1][0],label=f"A={AL[1]},a={a[2]},order={m[0]}",linestyle=':')
plt.plot(gray,ac[2][1][1],label=f"A={AL[1]},a={a[2]},order={m[1]}",linestyle=':')

plt.plot(gray,ac[2][2][0],label=f"A={AL[2]},a={a[2]},order={m[0]}",linestyle='--')
plt.plot(gray,ac[2][2][1],label=f"A={AL[2]},a={a[2]},order={m[1]}",linestyle='--')

ax_n = plt.gca()
ax_n.yaxis.set_minor_locator(MultipleLocator(0.02))
plt.legend(loc="best")
plt.grid(True, linestyle='--')
plt.ylim([0,0.5])
plt.xlim([0,255])
plt.show()



# gray_1 = []  
# gray_2 = [] 
# index=  []         
# for i1 in range(len(ac)): 
#     for m_ in range(len(Mc)):
#         diff=ac[i1][m_][0]- ac[i1][m_][1]
#         # print(diff)
#         indexes = [i for i, x in enumerate(diff) if  abs(x)<=0.007]
#         print(indexes)
#         # for g_ in range(len(indexes[0]))
#         gray1=gray[indexes[0]] 
#         gray2=gray[indexes[1]] 
#         gray_1.append(gray1)    
#         gray_2.append(gray2)
# print("gray_1:", gray_1)
# print("gray_2:", gray_2)

# plt.figure(3)
# plt.plot(a,gray_1[i for in range(len(gray_1))],label=f"triplicator 1st")
# plt.plot(a,gray_2,label=f"triplicator 2nd")

# ax_n = plt.gca()
# ax_n.yaxis.set_minor_locator(MultipleLocator(10))
# plt.legend(loc="best")
# plt.grid(True, linestyle='--')
# plt.ylim([0,255])
# plt.xlim([0,max(AL)])
# plt.show()


# # Calculate the absolute differences
# differences_1 = np.abs(Mc[0][0] - Mc[0][1])
# differences_2 = np.abs(Mc[1][0] - Mc[1][1])
# # Find the indices of the two smallest differences
# indices1 = np.argmin(differences_1)[:2]
# indices2 = np.argmin(differences_2)[:2]
# gray1=gray[int(indices1)]
# gray2=gray[int(indices2)]
# print(f"M={AL[0]}: {indices1,gray1},M={AL[1]}: {indices2,gray2}")

# plt.figure(4)
# plt.plot(x, PhaseProfile / max_value,label="linear")
# plt.plot(x_,PhaseProfile_R,-x_,PhaseProfile_L,label="nonlinear_")
# plt.legend(loc="best")
# ax_n = plt.gca()
# ax_n.xaxis.set_major_locator(MultipleLocator(0.25))
# plt.minorticks_on()
# plt.grid(True, linestyle='--')
# plt.show()

# plt.figure(3)
# plt.plot(x, PhaseProfile / max_value,label="linear")
# plt.plot(x,PhaseProfile_n,label="nonlinear")
# plt.plot(x,phaseProfile_C,label="nonlinearM+g")
# plt.legend(loc="best")
# ax_n = plt.gca()
# ax_n.xaxis.set_major_locator(MultipleLocator(0.25))
# plt.minorticks_on()
# plt.grid(True, linestyle='--')
# plt.show()