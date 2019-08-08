import numpy as np
import scipy.signal as sig 
from pylab import *
import csv
import time  

# csv file name 
#h.csv contains the FIR Filter coeffs, and x1.csv contains the zadoff-chu sequence
filename = "h.csv"
  
b =np.zeros(12)
i = 0
with open(filename, 'r') as filehandle:  
    for line in filehandle:
        b[i] = float(line)
        i+=1

w,h = sig.freqz(b)


def plotter(x,y,x1='w',y1='Magnitude',t1='xyz',s1='x.png'):
    plot(x,y)
    xlabel(x1)
    ylabel(y1)
    title(t1)
    grid(True)
    savefig(s1)
    show()

plotter(w,abs(h),'w','Magnitude','Low pass filter',"plot1.png")

plotter(w,angle(h),'w','Phase','Low pass filter',"plot2.png")


n = arange(2**10)
x = cos(0.2*pi*n) + cos(0.85*pi*n)
plotter(n,x,'n','Amplitude','Input signal',"plot3.png")

y = np.zeros(len(x))
for i in arange(len(x)):
    for k in arange(len(b)):
        y[i]+=x[i-k]*b[k]
    
    
plotter(n,y,'n','Amplitude','Output of linear convolution',"plot4.png")


def showFFT(w,Y,xlim1=pi,ylabel1= "|Y|",ylabel2 = "phi",titl = "",xlabel1 = "w"):
        figure()
        subplot(2,1,1)
        plot(w,abs(Y),lw=2)
        xlim([-xlim1,xlim1])
        ylabel(ylabel1,size=16)
        title(titl)
        grid(True)
        subplot(2,1,2)
        ro = True
        if (ro):
            plot(w,angle(Y),'ro',lw=2)
        if(0):
            ii=where(abs(Y)>1e-3)
            plot(w[ii],angle(Y[ii]),'go',lw=2)
        xlim([-xlim1,xlim1])
        ylabel(ylabel2,size=16)
        xlabel(xlabel1,size=16)
        grid(True)
        show()

y=ifft(fft(x)*fft(concatenate((b,zeros(len(x)-len(b))))))
plotter(n,real(y),'n','Amplitude','Output of circular convolution',"plot5.png")

def circular_conv(x,h):
    P = len(h)
    n_ = int(ceil(log2(P)))
    h_ = np.concatenate((h,np.zeros(int(2**n_)-P)))
    P = len(h_)
    n1 = int(ceil(len(x)/2**n_))
    x_ = np.concatenate((x,np.zeros(n1*(int(2**n_))-len(x))))
    y = np.zeros(len(x_)+len(h_)-1)
    for i in range(n1):
        temp = np.concatenate((x_[i*P:(i+1)*P],np.zeros(P-1)))
        y[i*P:(i+1)*P+P-1] += np.fft.ifft(np.fft.fft(temp) * np.fft.fft( np.concatenate( (h_,np.zeros(len(temp)-len(h_))) ))).real
    return y

y = circular_conv(x,b)
plotter(n,real(y[:1024]),'n','Amplitude','Output of circular convolution using linear convolution',"plot6.png")

file2 = "x1.csv"

lines = []
with open(file2,'r') as file2:
    csvreader = csv.reader(file2)
    for row in csvreader:
        lines.append(row)
lines2 = []
for line in lines:
    line = list(line[0])
    try :
        line[line.index('i')]='j'
        lines2.append(line)
    except ValueError:
        lines2.append(line)
        continue
x = [complex(''.join(line)) for line in lines2]

X = np.fft.fft(x)
x2 = np.roll(x,5)
cor = np.fft.ifftshift(np.correlate(x2,x,'full'))
print(len(cor))
figure()
xlim(0,20)
plotter(linspace(0,len(cor)-1,len(cor)),abs(cor),'t','Correlation','auto-correlation',"plot7.png",)



