# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:28:11 2024

@author: SG
"""

import numpy as np
import scipy as sy
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
# AL, AR, B, M, x,p=sp.symbols("Al AR B M x p")
x=np.linspace(-0.5, 0.5,640)  # linear
x_=np.linspace(0, 0.5,320)    #nonlinear

M=1.007
A=1000
AL=[1000,2,50]
AR=AL
m=[0,1,-1]
p=1
B=0.1
a=[0.964,1,0.7]
# gray=8

gray=np.arange(0,256,1)#[0,8,16,24,32,40,48,56,64,72,80,88,96,
      # 104,112,120,128,136,144,152,160,168,
      # 176,184,192,200,208,216,224,232,240,
      # 248,255]

ac=[]
for a_ in range (len(a)):
          
    # Phase profile for the linear
    f_linear=2*x/p
    # define nonlinear f_1, f is normalized, which affects phase profile a lot
    f_1=np.arctan(B*(f_linear/2))
    max_index_1 = np.argmax(f_1)
    max_value_1 = f_1[max_index_1]
    f_n=f_1/max_value_1
    # equation=M*sp.atan(AL*sp.cos(f) )
    # print(f"Equation: {equation}")
    
    PhaseProfile=np.arctan(A*np.cos(f_linear*np.pi))/(np.pi)
    max_index = np.argmax(PhaseProfile)
    max_value = PhaseProfile[max_index]
    
    # Phase profile for the positive/right
    f_=np.arctan(B*(x_/p))
    max_index_ = np.argmax(f_)
    max_value_ = f_[max_index_]
    f_L=f_/max_value_
    
    Mc=[]
    for n in range(len(AL)):
        PhaseProfile_L=np.arctan(AL[n]*np.cos(f_L*np.pi))/(np.pi)
        max_index_L = np.argmax(PhaseProfile_L)
        max_value_L = PhaseProfile_L[max_index_L]
        PhaseProfile_L=PhaseProfile_L/max_value_L
        
        # Phase profile for the negtive/left            
        f_r=np.arctan(B*(-x_/p))
        min_index_r = np.argmin(f_r)
        min_value_r = f_r[min_index_r]
        f_R=f_/min_value_r
        
        PhaseProfile_R=np.arctan(AL[n]*np.cos(f_R*np.pi))/(np.pi)
        max_index_R = np.argmax(PhaseProfile_R)
        max_value_R = PhaseProfile_R[max_index_R]
        PhaseProfile_R=PhaseProfile_R/max_value_R
        
        # combine L and R together
        # PhaseProfile_n=[0] * (len(x_)+1)
        # PhaseProfile_n[-1:] = PhaseProfile_L[::-1]
        PhaseProfile_n=np.concatenate((PhaseProfile_L[::-1], PhaseProfile_R))
        
    
           
        
        pixelation=[0] * len(x)
        for i in range(len(x)):
            if x[i]<len(x)/2 and x[i]>a[a_]/2:
                pixelation[i]=0
            elif x[i]>-len(x)/2 and x[i]<-a[a_]/2:
                pixelation[i]=0
            else:
                pixelation[i]=1
                
    
    
        PhaseProfile_M=PhaseProfile_n*M
        
        mc=[]        
        for k in range(len(m)): 
            # print(len(m))
            cc=[]       
            for j in range(len(gray)):        
                gray_M=M*gray[j]/255        
                phaseProfile_C=PhaseProfile_M*gray_M       
                ex=m[k]*np.pi*f_linear            
                phase=pixelation*np.exp(1j*phaseProfile_C*np.pi)*np.exp(1j*ex)
                cm=np.sum(phase)
                cc.append(cm)
            mc.append(cc)
        Im=mc*np.conjugate(mc)/len(x)**2 
        Mc.append(Im)
    ac.append(Mc)


plt.figure(2)
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


plt.figure(3)
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

plt.figure(4)
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

plt.figure(5)
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