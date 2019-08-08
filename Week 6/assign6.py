import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
import scipy


def plotter(x,y,title,xlabel,ylabel,show=True,legend =[],leg = False):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,y)
    if(leg):
        plt.legend(legend)
    if(show):
        plt.show()

#Question 1
#this function returns the transfer function of the forced oscillatory response the function represents 
def transfer_spring(frequency,decay):
    p= np.polymul([1.0,0,2.25],[1,-2*decay,frequency*frequency + decay*decay])
    return sp.lti([1,-1*decay],p)

t,x = sp.impulse(transfer_spring(1.5,-0.5),None,np.linspace(0,50,5001))
plotter(t,x,"Forced Damping Oscillator with decay = 0.5","t","x")

t,x = sp.impulse(transfer_spring(1.5,-0.05),None,np.linspace(0,50,5001))
plotter(t,x,"Forced Damping Oscillator with decay = 0.05","t","x")

freq = np.linspace(1.4,1.6,5)
plt.title("Forced Damping Oscillator with Different frequencies")
plt.xlabel("t")
plt.ylabel("x")
l = []
for f in freq:
    H = sp.lti([1],[1,0,2.25])
    t = np.linspace(0,150,5001)
    f_ = np.cos(f*t)*np.exp(-0.05*t)*(t>0)
    t,x,_ = sp.lsim(H,f_,t)
    plt.plot(t,x)
    l.append("freq =" + str(f))

plt.legend(l)
plt.show()


#solve for X in coupling equation
X = sp.lti([1,0,2],[1,0,3,0])
t,x = sp.impulse(X,None,np.linspace(0,50,5001))
plotter(t,x,"Coupled Oscilations: X and Y","t","x",show = False)
#solve for Y in coupling equation
Y = sp.lti([2],[1,0,3,0])
t,y = sp.impulse(Y,None,np.linspace(0,50,5001))
plotter(t,y,"Coupled Oscilations: X and Y","t","y",leg = 1,legend = ['x','y'])
#defines V_i
def func(t):
    return np.cos(1000*t) -np.cos(1e6*t)
#returns Low pass filter response for given input and time period
def RLC(time,inp=func,bode = False,R=100,L=1e-6,C=1e-6):    
    H = sp.lti([1],[L*C,R*C,1])
    if bode:
        w,S,phi = H.bode()
        fig,(ax1,ax2) = plt.subplots(2,1)
        ax1.set_title("Magnitude response")
        ax1.semilogx(w,S)
        ax2.set_title("Phase response")
        ax2.semilogx(w,phi)

        plt.show()
    return sp.lsim(H,inp(time),time)

t=np.linspace(0,30e-6,10000)
t,y,_ = RLC(t,bode = 1)
plotter(t,y,"Output of RLC for t<30u","t","x")

t=np.linspace(0,30e-3,10000)
t,y,_ = RLC(t)
plotter(t,y,"Output of RLC for t<30m","t","x")










#animation code
'''
import cv2
freq = np.linspace(1.4,1.6,50)
for f in freq:
    t,x = sp.impulse(transfer_spring(f,-0.05),None,np.linspace(0,150,5001))
    plt.plot(t,x)
    plt.savefig('a.png')
    plt.clf()
    plt.cla()
    plt.close()
    img = cv2.imread("a.png")
    cv2.imshow('fam',img)
    cv2.waitKey(1)
'''
