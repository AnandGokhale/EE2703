import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.special as sp

def load(FILENAME):
    data = np.loadtxt(FILENAME)
    x = data[:,0]
    y = data[:,1:]
    return x,y

def plot_with_legend(x,y):
    plt.plot(x,y)
    scl=np.logspace(-1,-3,9)
    plt.xlabel(r'$t$',size=20)
    plt.ylabel(r'$f(t)+n$',size=20)
    plt.legend(scl)
    plt.show()

def g(t,A=1.05,B=-0.105):
    return A*sp.jn(2,t)+B*t

def plot_g(t):
    plt.figure(0)
    plt.plot(x,g(x))
    plt.show()

def plot_errorbar(x,y,i):
    y_true = g(x)
    sigma = np.std(y[:,i]-y_true)
    plt.plot(x,y_true)
    plt.errorbar(x[::5],y[::5,i],sigma,fmt='ro')
    plt.show()

def generateM(x):
    M = np.zeros((x.shape[0],2))
    M[:,0] = sp.jn(2,x)
    M[:,1] = x
    return M

def error(x,AB):
    M = generateM(x)
    y_true = np.reshape(g(x),(101,1))
    y_pred = np.matmul(M,AB)
    return (np.square(y_pred - y_true)).mean()

def generateAB(i,j,step1 = 0.1,step2 = 0.01,Amin=0,Bmin = -0.2):
    AB = np.zeros((2,1))
    AB[0][0] = Amin +  step1 * i
    AB[1][0] = Bmin +step2 * j
    return AB
def find_error_matrix(x,y,noise_index):
    try:
        y_noisy = np.reshape(y[:,noise_index],(101,1))
    except:
        y_noisy =np.reshape(g(x),(101,1))
    error = np.zeros((21,21))
    M = generateM(x)
    for i in range(21):
        for j in range(21):
            error[i,j] = np.square( np.matmul(M,generateAB(i,j)) - y_noisy).mean()
    return error


def plot_contour(x,y):
    a = np.linspace(0,2,21)
    b = np.linspace(-0.2,0,21)
    X, Y = np.meshgrid(a,b)
    error = find_error_matrix(x,y,0)
    CS = plt.contour(X,Y,error,[0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5])
    plt.clabel(CS,CS.levels[:3], inline=1, fontsize=10)
    plt.show()
    return

def estimateAB(M,b):
    return scipy.linalg.lstsq(M,b)

def error_pred(pred,true):
    return np.square(pred[0]-true[0]),np.square(pred[1]-true[1])


x,y = load("fitting.dat")
plot_with_legend(x,y)
plot_g(x)
plot_errorbar(x,y,8)

AB = np.zeros((2,1))
AB[0][0] = 1.05
AB[1][0] = -0.105
print(error(x,AB))

plot_contour(x,y)


prediction,_,_,_ = estimateAB(generateM(x),y[:,1])

print(error_pred(prediction,AB))

scl=np.logspace(-1,-3,9)
error_a = np.zeros(9)
error_b = np.zeros(9)
for i in range(9):
    prediction,_,_,_ = estimateAB(generateM(x),y[:,i])
    error_a[i],error_b[i] = error_pred(prediction,AB)

'''basex = 10,'''
plt.loglog(scl,error_a,'r--',basex = 10)
plt.scatter(scl,error_a)
plt.loglog(scl,error_b, 'b--',basex = 10)
plt.scatter(scl,error_b)
plt.legend(["A","B"])
plt.show()











