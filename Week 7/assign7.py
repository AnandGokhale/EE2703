import scipy.signal as sp
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import sympy
sympy.init_session


#plotting helper function
def plotter(x,y,title,xlabel,ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,y)
    plt.show()

#low pass filter definition function
def lowpass(R1,R2,C1,C2,G,Vi):
    s=  sympy.symbols("s")
    A = sympy.Matrix([[0,0,1,-1/G],\
            [-1/(1+s*R2*C2),1,0,0],\
            [0,-G,G,1],\
            [-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
    b=  sympy.Matrix([0,0,0,-Vi/R1])
    V = A.inv()*b
    return A,b,V

#high pass filter definition function
def highpass(R1,R3,C1,C2,G,Vi):
    s=  sympy.symbols("s")
    A=sympy.Matrix([[0,-1,0,1/G],
        [s*C2*R3/(s*C2*R3+1),0,-1,0],
        [0,G,-G,1],
        [-s*C2-1/R1-s*C1,0,s*C2,1/R1]])
    b=sympy.Matrix([0,0,0,-Vi*s*C1])
    V=A.inv()*b

    return (A,b,V)
#this is a function that converts a sympy function to a verrsion that is understood by scipy.signals 
def symToTransferFn(Y):
    Y = sympy.expand(sympy.simplify(Y))
    n,d = sympy.fraction(Y)
    n,d = sympy.Poly(n,s), sympy.Poly(d,s)
    num,den = n.all_coeffs(), d.all_coeffs()
    num,den = [float(f) for f in num], [float(f) for f in den]
    return num,den
#plots step response
def stepresponse(Y,title):
    num,den = symToTransferFn(Y)
    den.append(0)
    H = sp.lti(num,den)
    t,y=sp.impulse(H,T = np.linspace(0,1e-3,10000))
    
    plotter(t,y,title,"t","output")
    return
#sum of sinusoids
def inputs(t):
    return (np.sin(2000*np.pi*t)+np.cos(2e6*np.pi*t))
#function that calculates response for arbitrary function
def inp_response(Y,title,inp=inputs,tlim=1e-3):
    
    num,den = symToTransferFn(Y)
    H = sp.lti(num,den)
    t = np.linspace(0,tlim,100000)
    t,y,svec = sp.lsim(H,inp(t),t)
    plotter(t,y,title,"t","Vo")
    return

#High frequency damped sinusoids
def damped1(t,decay=3e3,freq=1e7):
    return np.cos(freq*t)*np.exp(-decay*t) * (t>0)
#Low frequency damped sinusoids
def damped2(t,decay=1e1,freq=1e3):
    return np.cos(freq*t)*np.exp(-decay*t) * (t>0)

#first we deal with lowpass filters

s =  sympy.symbols("s")
#defining the low pass transfer function
A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)
H=V[3]
print (H)
w=np.logspace(0,8,801)
ss=1j*w
hf=sympy.lambdify(s,H,"numpy")
v=hf(ss)
#plotting low pass Magnitude response
plt.title("Low pass Magnitude response")
plt.xlabel("w")
plt.loglog(w,abs(v),lw=2)
plt.grid(True)
plt.show()

#Problem 1, finding step response
stepresponse(H,"step response for Low pass filter")
#Problem 2
t = np.linspace(0,1e-3,1000000)
plotter(t,inputs(t),"sum of sinusoids","t","Vi(t)")
inp_response(H,"Response of Low pass Filter to sum of sinusoids",inputs)


#response to a damped function
inp_response(H,"Response of Low pass Filter to Damped High Frequency sinusoid",damped1)
inp_response(H,"Response of Low pass Filter to Damped Low Frequency sinusoid",damped2,tlim = .5)


#now for high pass filters
A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1)
H=V[3]
print (H)
#plotting high pass Magnitude response
w=np.logspace(0,8,801)
ss=1j*w
hf=sympy.lambdify(s,H,"numpy")
v=hf(ss)

plt.title("Magnitude response for high pass filter")
plt.xlabel("w")
plt.loglog(w,abs(v),lw=2)
plt.grid(True)
plt.show()

#plotting step response
stepresponse(H,"step response for high pass filter")
#plotting sum of sinusoids 
inp_response(H,"Response of High pass Filter to sum of sinusoids",inputs,tlim= 1e-5)
#plotting response to damped sinusoids
inp_response(H,"Response of High pass Filter to Damped High Frequency sinusoid",damped1)
inp_response(H,"Response of High pass Filter to Damped Low Frequency sinusoid",damped2,tlim = .5)

#Plotting high damped sinusoid
t = np.linspace(0,1e-3,1e5)
plt.title("High frequency damped sinusoid")
plt.xlabel("$t$")
plt.ylabel("$v_i(t)$")
plt.plot(t,damped1(t,decay=3e3,freq=1e7))
plt.grid()
plt.show()

#Plotting low damped sinusoid
t = np.linspace(0,.5,1e5)
plt.title("Low frequency damped sinusoid")
plt.xlabel("$t$")
plt.ylabel("$v_i(t)$")
plt.plot(t,damped2(t))
plt.grid()
plt.show()
