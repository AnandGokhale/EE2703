import numpy as np
import scipy
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math

#this function converts e^x into a periodic version with period 2pi
def helper(fun):
    def newfun(x):
        return fun(np.remainder(x,2*np.pi))
    return newfun
#defining functions using numpy
def e(x):
    return np.exp(x)

def coscos(x):
    return np.cos(np.cos(x))
#Plotting functions

def plot_e(upper_limit,lower_limit, N= 10000):
    x = np.linspace(lower_limit,upper_limit,N)
    plt.title(r'$e^x$ on a semilogy plot')
    plt.xlabel('x')
    plt.ylabel(r'$log(e^x)$')
    plt.grid(True)
    plt.semilogy(x,e(x))
    plt.show()

def plot_coscos(upper_limit,lower_limit, N= 10000):  
    x = np.linspace(lower_limit,upper_limit,N)
    plt.title(r'Plot of $cos(cos(x))$')
    plt.xlabel('x')
    plt.ylabel(r'$cos(cos(x))$')
    plt.grid(True)
    plt.plot(x,coscos(x))
    plt.show()
#this function returns the first n fourier coefficients of the function f using the integration method 
def FT(n,function):
    a = np.zeros(n)
    def fcos(x,k,f):
        return f(x)*np.cos(k*x)/np.pi
    def fsin(x,k,f):
        return f(x)*np.sin(k*x)/np.pi

    a[0] = integrate.quad(function,0,2*np.pi)[0]/(2*np.pi)
    for i in range(1,n):
        if(i%2==1):
            a[i] = integrate.quad(fcos,0,2*np.pi,args=(int(i/2)+1,function))[0]
        else:
            a[i] = integrate.quad(fsin,0,2*np.pi,args=(int(i/2),function))[0]
    return a
#This function plots eFT and cosFT in semilogy and the loglog scale
def plot4(eFT,cosFT,color = 'ro'):
    eFT = np.abs(eFT)
    
    cosFT = np.abs(cosFT)
    plt.title(r"Coefficients of fourier series of $e^x$ on a semilogy scale")
    plt.xlabel(r'$n$')
    plt.ylabel(r'$log(coeff)$')
    plt.semilogy(eFT,color)
    plt.grid(True)
    plt.show()
    plt.title(r"Coefficients of fourier series of $e^x$ on a loglog scale")
    plt.xlabel(r'$log(n)$')
    plt.ylabel(r'$log(coeff)$')
    plt.loglog(eFT,color)
    plt.grid(True)
    plt.show()

    plt.title(r"Coefficients of fourier series of $cos(cos(x))$ on a semilogy scale")
    plt.xlabel(r'$n$')
    plt.ylabel(r'$log(coeff)$')
    plt.semilogy(cosFT,color)
    plt.show()
    plt.title(r"Coefficients of fourier series of $cos(cos(x))$ on a loglog scale")
    plt.xlabel(r'$log(n)$')
    plt.ylabel(r'$log(coeff)$')
    plt.loglog(cosFT,color)
    plt.grid(True)
    plt.show()


#PART 1
plot_e(-2*np.pi,4*np.pi)
plot_coscos(-2*np.pi,4*np.pi)

#calculating fourier coefficients
cosFT = FT(51,coscos)
eFT = FT(51,e)

#Part 2
plot4(eFT,cosFT)

#This function Generates Matrices for a least squares approach
def generateAb(x,f):
    A = np.zeros((x.shape[0],51))
    A[:,0] = 1
    for i in range(1,26):
        A[:,2*i-1]=np.cos(i*x)
        A[:,2*i]=np.sin(i*x)
    return A,f(x)

#setting endpoint = False gives a larger error
x = np.linspace(0,2*np.pi,400,endpoint=True)

Acos,bcos = generateAb(x,coscos)
Ae,be = generateAb(x,e)
#Solving using lstsq
ccos = scipy.linalg.lstsq(Acos,bcos)[0]
ce = scipy.linalg.lstsq(Ae,be)[0]

#This function plots Fourier coefficients obtained by the two methods in semilogy and the loglog scale
def plot8(eFT,cosFT,ce,ccos,color = 'ro'):
    eFT = np.abs(eFT)
    cosFT = np.abs(cosFT)
    ce = np.abs(ce)
    ccos = np.abs(ccos)
    plt.title(r"Coefficients of fourier series of $e^x$ on a semilogy scale")
    plt.xlabel(r'$n$')
    plt.ylabel(r'$log(coeff)$')
    plt.semilogy(eFT,'ro')
    plt.semilogy(ce,color)
    plt.legend(["true","pred"])
    plt.grid(True)
    plt.show()
    plt.title(r"Coefficients of fourier series of $e^x$ on a loglog scale")
    plt.xlabel(r'$log(n)$')
    plt.ylabel(r'$log(coeff)$')
    plt.loglog(eFT,'ro')
    plt.semilogy(ce,color)
    plt.legend(["true","pred"])
    plt.grid(True)
    plt.show()

    plt.title(r"Coefficients of fourier series of $cos(cos(x))$ on a semilogy scale")
    plt.xlabel(r'$n$')
    plt.ylabel(r'$log(coeff)$')
    plt.semilogy(cosFT,'ro')
    plt.semilogy(ccos,color)
    plt.legend(["true","pred"])
    plt.grid(True)
    plt.show()
    plt.title(r"Coefficients of fourier series of $cos(cos(x))$ on a loglog scale")
    plt.xlabel(r'$log(n)$')
    plt.ylabel(r'$log(coeff)$')
    plt.loglog(cosFT,'ro')
    plt.semilogy(ccos,color)
    plt.legend(["true","pred"])
    plt.grid(True)
    plt.show()

plot8(eFT,cosFT,ce,ccos,'go')

#measuring absolute error
print("The error in Coefficients of e^x =",np.amax(np.abs(eFT -ce)))
print("The error in Coefficients of cos(cos(x)) =",np.amax(np.abs(cosFT -ccos)))



ce = np.reshape(ce,(51,1))
#Finding values of the function from the Coefficients obtained using lstsq
TTT = np.matmul(Ae,ce)
#plotting results
x = np.linspace(0,2*np.pi,400,endpoint=True)
plt.title(r"Plot of $e^x$")
t = np.linspace(-2*np.pi,4*np.pi,10000,endpoint=True)
plt.semilogy(t,e(t))
l = helper(e)
plt.semilogy(t,l(t))
plt.semilogy(x,TTT,'go')
plt.xlabel('x')
plt.ylabel(r'$e^x$')
plt.legend(["true","expected","pred"])
plt.grid(True)
plt.show()

ccos = np.reshape(ccos,(51,1))
#Finding values of the function from the Coefficients obtained using lstsq
TTT = np.matmul(Acos,ccos)
#plotting results
x = np.linspace(0,2*np.pi,400,endpoint=True)
plt.title(r"Plot of $cos(cos(x))$")
t = np.linspace(-2*np.pi,4*np.pi,10000,endpoint=True)
plt.plot(x,TTT,'ro')
plt.plot(t,coscos(t))
plt.xlabel('x')
plt.ylabel(r'$cos(cos(x))$')
plt.legend(["prediction","true"])
plt.grid(True)
plt.show()
