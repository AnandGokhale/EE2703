from pylab import*

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def examples():
    t=linspace(-pi,pi,65)[:-1]
    dt=t[1]-t[0];fmax=1/dt
    y=sin(sqrt(2)*t)
    y[0]=0 # the sample corresponding to -tmax should be set zero
    y=fftshift(y) # make y start with y(t=0)
    Y=fftshift(fft(y))/64.0
    w=linspace(-pi*fmax,pi*fmax,65)[:-1]
    figure()
    subplot(2,1,1)
    plot(w,abs(Y),lw=2)
    xlim([-10,10])
    ylabel(r"$|Y|$",size=16)
    title(r"Spectrum of $\sin\left(\sqrt{2}t\right)$")
    grid(True)
    subplot(2,1,2)
    plot(w,angle(Y),'ro',lw=2)
    xlim([-10,10])
    ylabel(r"Phase of $Y$",size=16)
    xlabel(r"$\omega$",size=16)
    grid(True)
    savefig("fig10-1.png")
    show()
    t1=linspace(-pi,pi,65)[:-1]
    t2=linspace(-3*pi,-pi,65)[:-1]
    t3=linspace(pi,3*pi,65)[:-1]
    # y=sin(sqrt(2)*t)
    figure(2)
    plot(t1,sin(sqrt(2)*t1),'b',lw=2)
    plot(t2,sin(sqrt(2)*t2),'r',lw=2)
    plot(t3,sin(sqrt(2)*t3),'r',lw=2)
    ylabel(r"$y$",size=16)
    xlabel(r"$t$",size=16)
    title(r"$\sin\left(\sqrt{2}t\right)$")
    grid(True)
    savefig("fig10-2.png")
    show()


    t1=linspace(-pi,pi,65)[:-1]
    t2=linspace(-3*pi,-pi,65)[:-1]
    t3=linspace(pi,3*pi,65)[:-1]
    y=sin(sqrt(2)*t1)
    figure(3)
    plot(t1,y,'bo',lw=2)
    plot(t2,y,'ro',lw=2)
    plot(t3,y,'ro',lw=2)
    ylabel(r"$y$",size=16)
    xlabel(r"$t$",size=16)
    title(r"$\sin\left(\sqrt{2}t\right)$ with $t$ wrapping every $2\pi$ ")
    grid(True)
    savefig("fig10-3.png")
    show()

    t=linspace(-pi,pi,65)[:-1]
    dt=t[1]-t[0]
    fmax=1/dt
    y=t
    y[0]=0 # the sample corresponding to -tmax should be set zeroo
    y=fftshift(y) # make y start with y(t=0)
    Y=fftshift(fft(y))/64.0
    w=linspace(-pi*fmax,pi*fmax,65)[:-1]
    figure()
    semilogx(abs(w),20*log10(abs(Y)),lw=2)
    xlim([1,10])
    ylim([-20,0])
    xticks([1,2,5,10],["1","2","5","10"],size=16)
    ylabel(r"$|Y|$ (dB)",size=16)
    title(r"Spectrum of a digital ramp")
    xlabel(r"$\omega$",size=16)
    grid(True)
    savefig("fig10-4.png")
    show()


    t1=linspace(-pi,pi,65)[:-1]
    t2=linspace(-3*pi,-pi,65)[:-1]
    t3=linspace(pi,3*pi,65)[:-1]
    n=arange(64)
    wnd=fftshift(0.54+0.46*cos(2*pi*n/63))
    y=sin(sqrt(2)*t1)*wnd
    figure(3)
    plot(t1,y,'bo',lw=2)
    plot(t2,y,'ro',lw=2)
    plot(t3,y,'ro',lw=2)
    ylabel(r"$y$",size=16)
    xlabel(r"$t$",size=16)
    title(r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$ ")
    grid(True)
    savefig("fig10-5.png")
    show()


    t=linspace(-pi,pi,65)[:-1]
    dt=t[1]-t[0];
    fmax=1/dt
    n=arange(64)
    wnd=fftshift(0.54+0.46*cos(2*pi*n/63))
    y=sin(sqrt(2)*t)*wnd
    y[0]=0 # the sample corresponding to -tmax should be set zeroo
    y=fftshift(y) # make y start with y(t=0)
    Y=fftshift(fft(y))/64.0
    w=linspace(-pi*fmax,pi*fmax,65)[:-1]
    figure()
    subplot(2,1,1)
    plot(w,abs(Y),lw=2)
    xlim([-8,8])
    ylabel(r"$|Y|$",size=16)
    title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
    grid(True)
    subplot(2,1,2)
    plot(w,angle(Y),'ro',lw=2)
    xlim([-8,8])
    ylabel(r"Phase of $Y$",size=16)
    xlabel(r"$\omega$",size=16)
    grid(True)
    savefig("fig10-6.png")
    show()

    t=linspace(-4*pi,4*pi,257)[:-1]
    dt=t[1]-t[0];fmax=1/dt
    n=arange(256)
    wnd=fftshift(0.54+0.46*cos(2*pi*n/256))
    y=sin(sqrt(2)*t)
    y=y*wnd
    y[0]=0 # the sample corresponding to -tmax should be set zeroo
    y=fftshift(y) # make y start with y(t=0)
    Y=fftshift(fft(y))/256.0
    w=linspace(-pi*fmax,pi*fmax,257)[:-1]
    figure()
    subplot(2,1,1)
    plot(w,abs(Y),lw=2)
    xlim([-8,8])
    ylabel(r"$|Y|$",size=16)
    title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
    grid(True)
    subplot(2,1,2)
    plot(w,angle(Y),'ro',lw=2)
    xlim([-8,8])
    ylabel(r"Phase of $Y$",size=16)
    xlabel(r"$\omega$",size=16)
    grid(True)
    savefig("fig10-7.png")
    show()

examples()
#helper functions
def spectrum(lim,n,f,t_=0,show_ = True,t_lims = False,windowing= False,xlim1=10,title1 = r"Spectrum of $\sin\left(\sqrt{2}t\right)$",xlabel1 = r"$\omega$",ylabel1= r"$|Y|$", ylabel2 = r"Phase of $Y$",savename = "abc.png"):
    if(t_lims):
        t = t_
    else:
        t=linspace(-lim,lim,n+1)[:-1]
    dt=t[1]-t[0];
    fmax=1/dt
    y = f(t)
    if (windowing):
        m=arange(n)
        wnd=fftshift(0.54+0.46*cos(2*pi*m/n))
        y = y*wnd
    y[0]=0 # the sample corresponding to -tmax should be set zeroo
    y=fftshift(y) # make y start with y(t=0)
    Y=fftshift(fft(y))/float(n)
    w=linspace(-pi*fmax,pi*fmax,n+1)[:-1]
    
    mag = abs(Y)
    ph = angle(Y)
    if(show_):
        figure()
        subplot(2,1,1)
        plot(w,mag,lw=2)
        xlim([-xlim1,xlim1])
        ylabel(ylabel1,size=16)
        title(title1)
        grid(True)
        subplot(2,1,2)
        ph[where(mag<3e-3)] = 0
        plot(w,ph,'ro',lw=2)
        xlim([-xlim1,xlim1])
        ylabel(ylabel2,size=16)
        xlabel(xlabel1,size=16)
        grid(True)
        savefig(savename)
        show()
    return w,Y

#defining cos^3
def cos3(t,w0=0.86):
    return (cos(w0*t))**3


def cosine(t,w0=1.5,delta=0.5):
    return cos(w0*t + delta)

#FFT of cos^3 windowed and unwindowed
a,b = spectrum(4*pi,64*4,cos3,xlim1= 3,windowing=False, title1 = r"Spectrum of $cos^3(w_0t)$",savename = '10-8.png')
a,b = spectrum(4*pi,64*4,cos3,xlim1= 3,windowing=True, title1 = r"Spectrum of $cos^3(w_0t)$",savename = '10-8.png')

#question 3
#FFT of cos(wt+delta) windowed windowed to estimate w, delta
w,Y = spectrum(pi,128,cosine,xlim1= 3,windowing=True, title1 = r"Spectrum of $cos(w_0t + \delta)$",savename = '10-8.png')

def est_omega(w,Y):
    ii = where(w>0)
    omega = (sum(abs(Y[ii])**2*w[ii])/sum(abs(Y[ii])**2))#weighted average
    print ("omega = ", omega)

def est_delta(w,Y,sup = 1e-4,window = 1):
    ii_1=np.where(np.logical_and(np.abs(Y)>sup, w>0))[0]
    np.sort(ii_1)
    points=ii_1[1:window+1]
    print (np.sum(np.angle(Y[points]))/len(points))#weighted average for first 2 points

est_omega(w,Y)
est_delta(w,Y)

#question 4
def noisycosine(t,w0=1.5,delta=0.5):
    return cos(w0*t + delta) + 0.1*np.random.randn(128)

w,Y = spectrum(pi,128,noisycosine,xlim1= 3,windowing=True, title1 = r"Spectrum of $cos(w_0t + \delta)$")

est_omega(w,Y)
est_delta(w,Y)

#question 5
 
def chirp(t):
    return cos(16*(1.5 + t/(2*pi))*t) 

w,Y = spectrum(pi,1024,chirp,xlim1= 60,windowing=True, title1 = r"Spectrum of chirp function")
w,Y = spectrum(pi,1024,chirp,xlim1= 60,windowing=False, title1 = r"Spectrum of chirp function")

#question 6

t=np.linspace(-np.pi,np.pi,1025);t=t[:-1]
t_arrays=np.split(t,16)

Y_mags=np.zeros((16,64))
Y_angles=np.zeros((16,64))
#splitting array and doing fft
for i in range(len(t_arrays)):
    w,Y = spectrum(lim = 10,t_ = t_arrays[i],t_lims=True,n = 64,f = chirp,xlim1= 60,windowing=False, title1 = r"Spectrum of chirp function",show_ = False)
    Y_mags[i] =  abs(Y)
    Y_angles[i] =  angle(Y)
#plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

t=np.linspace(-np.pi,np.pi,1025);t=t[:-1]
fmax = 1/(t[1]-t[0])
t=t[::64]
w=np.linspace(-fmax*np.pi,fmax*np.pi,64+1);w=w[:-1]
t,w=np.meshgrid(t,w)

surf=ax.plot_surface(w,t,Y_mags.T,cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.ylabel("Frequency")
plt.xlabel("time")

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(w,t,Y_angles.T,cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.ylabel("Frequency")
plt.xlabel("time")

plt.show()
