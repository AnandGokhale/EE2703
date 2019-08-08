from pylab import *

Nx = 31
Ny = 31

#def function(N=200):

x = linspace(-1,1,Nx+2)[2:]
y = linspace(-1,1,Ny+2)[2:]
X,Y = meshgrid(x,y)

ii = where((X**2 + Y**2)>(0.75)**2)
jj = where((abs(X)<0.5)* (abs(Y)<0.125))

phi = 0.5 * ones((Nx,Ny))
phi[ii] = 1
phi[jj] = 0 

def solve(phi,phiold):
    phi[1:-1,1:-1] = 0.25 * (phiold[:-2,1:-1] + phiold[2:,1:-1] + phiold[1:-1,2:] + phiold[1:-1,:-2])
    return phi

N = 200
err = zeros(N)

for i in range(N):
    phiold = phi.copy()
    phi = solve(phi,phiold)
    phi[ii] = 1
    phi[jj] = 0
    err[i] = np.sum(np.abs(phi-phiold))
contourf(X,Y,phi)
colorbar()
show()

#return phi

#function()