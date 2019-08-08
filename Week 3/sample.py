\documentclass{article}
\usepackage[utf8]{inputenc}

\title{Assignment No 3}
\author{Anand Uday Gokhale}
\date{10th February 2019}

\usepackage{natbib}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{listings}

\begin{document}

\maketitle

\section{Abstract}

This week’s Python assignment will focus on the following topics:

\begin{itemize}
  \item Reading data from files and parsing them
  \item Analysing the data to extract information
  \item Study the effect of noise on the fitting process
  \item Plotting graphs
\end{itemize}


\section{Introduction}
This weeks assignment starts off with generating data as a linear combination of the Bessel Function and $y = x$. Varying amounts  of Noise is added to this combination. \newline
\begin{align*}
f(t) = A*J_2(t) - B*t + n(t)\newline 
\end{align*}
Where :
\begin{itemize}
  \item $A = 1.05$  
  \item $B = 0.105$
  \item $n(t) =$ noise function
  \item $J_2(t) =$ Bessel function
\end{itemize}
            
We have to study the relation between the error of our estimation of A and B and the standard deviation of the noise that was added. 


\section{Part 1 : Generation Of Data}
Data is generated using the script provided as a part of the assignment. It utilizes the pylab library to find the value of the Bessel function and different values of $t$ and adds noise for 9 values of standard deviation.
\section{Part 2 : Importing the Data}
Numpy's loadtxt function was used to import data from the file where it was previously stored. The data consists of 10 columns.  The first column is time, while the remaining columns are data with varying noise levels. 
\lstset{language=Python}
\lstset{frame=lines}
\lstset{label={lst:code_direct}}
\lstset{basicstyle=\footnotesize}
\begin{lstlisting}
def load(FILENAME):
    data = np.loadtxt(FILENAME)
    x = data[:,0]
    y = data[:,1:]
    return x,y
\end{lstlisting}

\section{Part 3 : Plotting the Data}
This is done using Matplotlib's pyplot Library

\lstset{language=Python}
\lstset{frame=lines}
\lstset{label={lst:code_direct}}
\lstset{basicstyle=\footnotesize}
\begin{lstlisting}
def plot_with_legend(x,y):
    plt.plot(x,y)
    scl=np.logspace(-1,-3,9)
    plt.xlabel(r'$t$',size=20)
    plt.ylabel(r'$f(t)+n$',size=20)
    plt.legend(scl)
    plt.show() 
\end{lstlisting}


\begin{figure}[h!]
\centering
\includegraphics[scale=0.6]{part3.png}
\caption{Part 3 : Varying Noise Levels}
\label{:part3}
\end{figure}
\newpage
\section{Plotting the Original function}
With $A = 1.05$ and $B = -0.105$ the original function is plotted as follows:
\lstset{language=Python}
\lstset{frame=lines}
\lstset{label={lst:code_direct}} 
\lstset{basicstyle=\footnotesize}
\begin{lstlisting}
def g(t,A=1.05,B=-0.105):
    return A*sp.jn(2,t)+B*t
def plot_g(t):
    plt.figure(0)
    plt.plot(x,g(x))
    plt.show()
\end{lstlisting}
\begin{figure}[h!]
\centering
\includegraphics[scale=0.6]{part4.png}
\caption{The Original function}
\label{:part4}
\end{figure}
\newpage
\section{Part 5: Plotting Error Bars}
\lstset{language=Python}
\lstset{frame=lines}
\lstset{label={lst:code_direct}} 
\lstset{basicstyle=\footnotesize}
\begin{lstlisting}
def plot_errorbar(x,y,i):
    y_true = g(x)
    sigma = np.std(y[:,i]-y_true)
    plt.plot(x,y_true)
    plt.errorbar(x[::5],y[::5,i],sigma,fmt='ro')
    plt.show()
\end{lstlisting}
\begin{figure}[h!]
\centering
\includegraphics[scale=0.6]{part5.png}
\caption{Part 5: Error bars}
\label{:part5}
\end{figure}

\section{Part 6: Generating Matrix}
To generate the Matrix M:
\begin{center}
$M = 
\begin{bmatrix}
    J_2(t_1) & t_1 \\
    J_2(t_2) & t_2 \\
    \dots & \dots \\
    J_2(t_m) & t_m \\
\end{bmatrix}$
\end{center}
\begin{lstlisting}
def generateM(x):
    M = np.zeros((x.shape[0],2))
    M[:,0] = sp.jn(2,x)
    M[:,1] = x
    return M
\end{lstlisting}
To Find Mean Square Error Between the function generated and the original function:
\begin{lstlisting}
def error(x,AB):
    M = generateM(x)
    y_true = np.reshape(g(x),(101,1))
    y_pred = np.matmul(M,AB)
    return (np.square(y_pred - y_true)).mean()
\end{lstlisting}



\section{Part 7: Generating Error For Different Values of A and B}

We can find the MSE(Mean square error) between our predicted AB vs Actual AB using this formula:  

\begin{equation}
\epsilon_{ij} = \sum_{k=0}^{101}(f_k - g(t_k,A_i,B_j))^2 
\end{equation}
where $f_k = (k+1)^{th}$ column of the data

\begin{lstlisting}
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
\end{lstlisting}
\newpage

\section{Part 8: Plotting Contour of Error}

\begin{lstlisting}
def plot_contour(x,y):
    a = np.linspace(0,2,21)
    b = np.linspace(-0.2,0,21)
    X, Y = np.meshgrid(a,b)
    error = find_error_matrix(x,y,0)
    CS = plt.contour(X,Y,error)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.show()
    return
\end{lstlisting}
\begin{figure}[h!]
\centering
\includegraphics[scale=0.6]{part7.png}
\caption{Part 7: Contour Plot}
\label{:part7}
\end{figure}

\section{Part 9 : Estimating A and B}
AB is estimated by minimizing $|M*(AB) - b|$\newline
where $b =$ one of the columns of the data\newline
This can be done using the scipy.linalg.lstsq function 
\begin{lstlisting}
def estimateAB(M,b):
    return scipy.linalg.lstsq(M,b)
prediction,_,_,_ = estimateAB(generateM(x),y[:,1])
\end{lstlisting}
\newpage
\section{Part 10 : Error of A and B on a linear Scale}
\begin{figure}[h!]
\centering
\includegraphics[scale=0.6]{part10.png}
\caption{Error of A and B on a linear Scale}
\label{:part10}
\end{figure}

\section{Part 11 : Error of A and B on a loglog Scale}
\begin{figure}[h!]
\centering
\includegraphics[scale=0.55]{part11.png}
\caption{Error of A and B on a LogLog Scale}
\label{:part11}
\end{figure}



\end{document}

