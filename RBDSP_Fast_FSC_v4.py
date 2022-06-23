import matplotlib
# matplotlib.use('GTKAgg')

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from scipy import io

def Fast_SC(x,Nw,alpha_max,Fs,opt, WindowType = 'hanning'):

    # -->===================================
    # -->Check inputs
    # -->===================================
    if (alpha_max > Fs/2):
        print('\'alpha_max\' must be smaller than Fs/2!')
    
    if (alpha_max < 0):
        print('\'alpha_max\' must be non-negative!')
 
    # -->===================================
    # -->Set value of overlap
    # -->===================================
    Nv,dt,da,df = param_Fast_SC( len(x), Nw, alpha_max,Fs)
    
    # -->===================================
    # -->Computation of short-time Fourier transform 
    # -->===================================
  
    # STFT,f,t = LiteSpectrogram(x,Nw,Nv,Nw,Fs)
    STFT,f,t = LiteSpectrogram(x = x, Window = Nw, Noverlap = Nv, Nfft= Nw,Fs = Fs, WinType = WindowType)

    # -->===================================
    # -->Fast spectral correlation/coherence
    # -->===================================
    S, alpha, __ ,__ =   Fast_SC_STFT( STFT = STFT, Dt = dt, Wind = Nw, opt = opt, Fs = Fs,  Nfft = np.array([]), WinType = WindowType)

    I = np.where(alpha <= alpha_max)
    I = []
    I.extend( i for i in range(0,len(alpha)) if(alpha[i] <= alpha_max) )
    alpha = alpha[I[:]]
    S = S[:,I]  

    return(S,alpha,f,STFT,t,Nv)

def param_Fast_SC(L,Nw,alpha_max,Fs):

    # -->===================================    
    # -->block shift
    # -->===================================
    
    R = np.fix(Fs/2/alpha_max)
    R = np.maximum(1, np.minimum(R, np.fix(.25*Nw) ) )

    # -->===================================
    # -->block overlap
    # -->===================================
    Nv = Nw - R
    
    # -->===================================
    # -->time resolution of STFT (in s)
    # -->===================================    
    dt = R/Fs

    # -->===================================    
    # -->cyclic frequency resolution (in Hz)
    # -->===================================    
    da = Fs/L

    # -->===================================    
    # -->carrier frequency resolution (in Hz)
    # -->===================================
    df = Fs/Nw*np.sum(Nw**2)/np.mean(Nw)**2
    
    
    return(Nv,dt,da,df)

def Fast_SC_STFT( STFT, Dt, Wind, opt, Fs = 1, Nfft = None, WinType = 'hanning' ):

    NF = STFT.shape[0]    
    Nw = 2*(NF-1)      # -->window length
    
    flag = 0
    
    if(Nfft == None):
        Nfft = STFT.shape[1]
    else:
        if (Nfft.size == 0):
            Nfft = STFT.shape[1]
    
    
    if (opt == None):
        opt = { "abs": 0, "calib": 1, "coh": 0}
    else:
        if not("abs" in opt):
            opt["abs"] = 0
    
        if not("calib" in opt):
            opt["calib"] = 1
    
        if not("coh" in opt):
            opt["coh"] = 0
    
    if(np.size(Wind) == 1):
        Wind = GetWindow(WinType, Wind)



        # Window = io.loadmat("D:\GDRB_COC\codes\src_python\lib\matlab_hanning_window.mat")
        # Wind = Window['Window']
    # -->===================================
    # -->Whitening the STFT for computing the spectral coherence
    # -->===================================
    if(opt["coh"] == 1):
        Sx = np.mean( abs(STFT)**2, axis=1)  # -->Mean power spectral density
        Sx = Sx.reshape((Sx.shape[0],1))
        termo1 = 1/np.sqrt(Sx)
        termo2 = np.tile( termo1, (1, STFT.shape[1]) )
        STFT = np.multiply(STFT,termo2)
        # STFT = np.multiply( STFT, np.tile( 1/np.sqrt(Sx), (1, STFT.shape[1]) ) )
    # -->===================================
    # -->Computation of the cyclic modulation spectrum
    # -->===================================
    S,alpha, __,__,__ = CPS_STFT_zoom(0,STFT,Dt,Wind,Fs,Nfft,flag)
    W0,__, __ = Window_STFT_zoom(alpha,0,Dt,Wind,Nfft,'full', Fs = Fs)
    # W0,__, __ = Window_STFT_zoom(alpha,0,Dt,Wind,Nfft,'trunc', Fs = Fs)
    
    if (opt["abs"] == 1):
        S = np.abs(S)
        W = np.array(W0, copy=True)
        W = np.abs(W)
    else:
        W = np.array(W0, copy=True)
  
    # for i in range(int(np.ceil(Nfft/2)+1),Nfft):
    #     aux = W0[i]
    #     W[i] = 0
    #     W0[i]=aux
    W[int(np.ceil(Nfft/2)+1):Nfft] = 0 # -->truncate negative frequencies
    
    # -->===================================
    # -->Number of scans
    # -->===================================
    Fa = 1/Dt               # -->cyclic sampling frequency in Hz
    K = np.fix(Nw/2*Fa/Fs)


    
    for k in range(1,int(K)): 

        # -->===================================
        # -->positive cyclic frequencies
        # -->===================================
    
        Stemp,alpha,alpha0,__,__ = CPS_STFT_zoom(k/Nw*Fs,STFT,Dt,Wind,Fs,Nfft,flag)
            
        Wtemp = Shift_Window_STFT_zoom(W0, alpha0/Fa*Nfft,'trunc')
        # -->Wtemp = Window_STFT_zoom(alpha,alpha0,Dt,Wind,Nfft,Fs,'trunc')
        if(opt["abs"] == 1):
            S[:,2:Nfft] = S[:,2:Nfft] + abs(Stemp[:,2:Nfft])
            W[2:Nfft] = W[2:Nfft] + abs(Wtemp[2:Nfft])
        else:

            # W = W.reshape((W.shape[1],1))
            S[:,2:Nfft] = S[:,2:Nfft] + Stemp[:,2:Nfft]
            W[2:Nfft] = W[2:Nfft] + Wtemp[2:Nfft]
        # -->plot(abs(Wtemp),':'),plot(W)
        
        # -->negative cyclic frequencies
        Stemp,alpha,alpha0,__,__ = CPS_STFT_zoom(-k/Nw*Fs,STFT,Dt,Wind,Fs,Nfft,flag)
        Wtemp = Shift_Window_STFT_zoom(W0,alpha0/Fa*Nfft,'trunc')
        # -->Wtemp = Window_STFT_zoom(alpha,alpha0,Dt,Wind,Nfft,Fs,'trunc')
        if (opt["abs"] == 1):
            S[:,2:Nfft] = S[:,2:Nfft] + abs(Stemp[:,2:Nfft])
            W[2:Nfft] = W[2:Nfft] + abs(Wtemp[2:Nfft])
        else:
            S[:,2:Nfft] = S[:,2:Nfft] + Stemp[:,2:Nfft]
            W[2:Nfft] = W[2:Nfft] + Wtemp[2:Nfft]

    # -->===================================
    # -->Calibration
    # -->===================================
            
    if (opt["calib"] == 1):
        Winv = np.ones((Nfft,1))
        I,__ = np.where(W < .5*W[0])
        Winv[0:I[0]] = 1/W[0:I[0]]
        Winv[I[0]+1:Nfft] = 1/W[I[0]+1]
        valor_aux1 = np.sum(Wind[:]**2)
        valor_aux2 = np.tile(Winv.transpose(),(NF,1))
         
        S = np.multiply( S, valor_aux2*valor_aux1)
        # S = S*np.tile(Winv.transpose(),(NF,1))*np.sum(Wind[:]**2)
        
    else:
        Winv = 1/W(0)
        S = S*Winv*sum(Wind[:]**2)
    # -->===================================
    # -->Impose real values at zero cyclic frequency
    # -->===================================
    # -->S(:,1) = real(S(:,1))
    if(opt["coh"]== 1):
        valor_aux3 = np.mean(S[:,0])
        S = S/valor_aux3
        

    return(S,alpha,W,Winv)
    # -->Subroutines of Fast_SC_STFT.m
    
def CPS_STFT_zoom(alpha0, STFT, Dt, Window, Fs = 1, Nfft = [], flag = 0):


    
    # [NF,NT,N3] = STFT.shape
    NF,NT = STFT.shape
    N3 = 1
    # [NF,NT,N3] = np.size(STFT)
    Nw = 2*(NF-1)          # --> window length
    Fa = 1/Dt    # --> cyclic sampling frequency in Hz

    if(np.size(Nfft) == 0):
        Nfft = NT
    else:
        if(Nfft < NT):
            error('Nfft must be greater than or equal to the number of time samples in STFT!')
    
    # -->===================================
    # --> Check for aliasing
    # -->===================================
            
    if(flag == 0):
        if (np.abs(alpha0) > Fa/2):
            disp(['|alpha0| must be selected smaller than ',num2str(Fa/2),'!!'])


    # -->===================================
    # --> Vector of cyclic frequencies
    # -->===================================

    alpha = np.arange(0,Nfft)/Nfft*Fa
 
    # -->===================================
    # --> Computation "cross-frequency" cyclic modulation spectrum
    # -->===================================
    fk = int(np.round(alpha0/Fs*Nw))
    alpha0 = fk/Nw*Fs

    if(N3 == 1):
        if(fk >= 0):
            S = np.multiply( 
                                np.vstack( (STFT[fk:NF , : ] ,  np.zeros((fk,NT))) ),
                                STFT.conjugate()
                            )
        else:
            S = np.multiply(
                                np.vstack( (STFT[-fk:NF , : ].conjugate() , np.zeros((-fk,NT)))  ),
                                STFT
                            )

    else:
        if( fk >= 0):
            S = np.multiply(
                                np.vstack( ( np.squeeze( STFT[fk:NF,:,1]) , zeros(fk,NT) ) ),
                                np.squeeze(STFT[:,:,2]).conjugate()
                            )
        else:
            S = np.multiply(
                                np.vstack( ( np.squeeze(STFT[-fk:NF,:,1]).conjugate() , np.zeros(-fk,NT) ) ),
                                np.squeeze(STFT[:,:,2])
                            )
    # S = np.fft.fft(S, n=Nfft)/NT
    S = np.fft.fft(S, n=Nfft, axis=1)/NT

    # -->===================================
    # --> Calibration
    # -->===================================
    valor_aux = np.sum(Window[:]**2)
    valor_aux = valor_aux/Fs
    S = S/valor_aux
    # S = S/np.sum(Window[:]**2)/Fs

    # -->===================================
    # --> Removal of aliased cyclic frequencies
    # -->===================================
    ak = np.round(alpha0/Fa*Nfft)
    valor_aux = np.arange(int(np.ceil(Nfft/2)+1+ak),Nfft)
    S[:,int(np.ceil(Nfft/2)+1+ak):Nfft] = 0
    # S[:,int(np.ceil(Nfft/2)+ak):Nfft] = 0

    # -->===================================
    # --> Phase correction
    # -->===================================

    Iw = np.argmax(Window)

    a2 = alpha - alpha0
    a2 = a2/Fs
    a2 = -2j*np.pi*Iw*a2
    a2 = np.exp(a2)
    S = np.multiply(    
                        S,
                        np.tile(a2, (NF,1))
                    )

    return(S,alpha,alpha0,fk,Fa)

def Window_STFT_zoom(alpha,alpha0,Dt,Window,Nfft, opt, Fs = 1):

    Fa = 1/Dt    # --> cyclic sampling frequency in Hz

    # -->===================================
    # --> Computation the "zooming" window
    # -->===================================
    WSquared = Window[:]**2
    Iw = np.argmax(Window) # --> set origin of time to the centre of symmetry (maximum value) of the window
    W1 = np.zeros((Nfft,1))
    W2 = np.zeros((Nfft,1))
    
    # n = np.arange(0,Iw)
    n = np.arange(1,Iw+1).reshape((Iw,1))
    n = n/Fs


    # plt.figure()
    # plt.plot(WSquared)
    
    # fig, ([ax1,ax2],[ax3,ax4], [ax5, ax6]) = plt.subplots(3,2)

    for k in range(0,Nfft):

        # -->===================================
        # --> "positive" frequencies
        # -->===================================
        T =  WSquared[Iw:0:-1].reshape((WSquared[Iw:0:-1].shape[0],1))
        # valor1 = (2*np.pi*n*(alpha[k]-alpha0))
        valor1 = np.cos(2*np.pi*n*(alpha[k]-alpha0))
        valor2 = ( np.multiply( T, valor1 ))
        # valor2 = ( np.dot( WSquared[Iw:0:-1], valor1 ))
        # valor2 = ( WSquared[Iw:0:-1]*valor1 )
        valor3 = 2*np.sum( valor2)
        # valor2 = 2*np.sum( WSquared[Iw:0:-1]*valor1)
        W1[k] = WSquared[Iw] + valor3

        # W1[k] = WSquared[Iw] + 2*np.sum( np.multiply( WSquared[Iw:0:-1], np.cos(2*np.pi*n*(alpha[k]-alpha0)) ))

        # -->===================================
        # --> "negative" frequencies (aliased)
        # -->===================================
        valor4 = np.cos(2*np.pi*n*(alpha[k]-alpha0-Fa))
        valor5 =  np.multiply(T,valor4 )
        # valor5 =  np.dot(WSquared[Iw:0:-1],valor4 )
        valor6 = 2*np.sum(valor5 )
        W2[k] = WSquared[Iw] + valor6
        # W2[k] = WSquared[Iw] + 2*np.sum( np.multiply( WSquared[Iw:0:-1], np.cos(2*np.pi*n*(alpha[k]-alpha0-Fa))) )
        
        # W2(k) = WSquared(Iw) + 2*sum(WSquared(Iw-1:-1:1).*cos(2*pi*n*(alpha(k)-alpha0-Fa)));

        # ax1.plot(W1)
        # ax3.plot(valor1)
        # ax5.plot(valor2)
        # # ax3.plot(valor3)
        # ax2.plot(W2)
        # ax4.plot(valor4)
        # ax6.plot(valor5)
        
        
    W = W1 + W2
    # -->===================================
    # --> Note: sum(W2) = max(W)
    # -->===================================

    # -->===================================
    # --> Removal aliased cyclic frequencies
    # -->===================================

    if 'trunc' in opt:
        ak = np.round(alpha0/Fa*Nfft)
        W[int(np.ceil(Nfft/2)+1+ak):Nfft] = 0

    return (W,W1,W2)

def Shift_Window_STFT_zoom(W0,a0,opt):

    Nfft = len(W0)
    
    # -->===================================
    # --> Circular shift with linear interpolation for non-integer shifts
    # -->===================================
    a1 = int(np.floor(a0))
    a2 = int(np.ceil(a0))

    if (a1 == a2):
        # W = np.roll(W0,a0,axis=0)
        W = np.roll(W0,int(a0),axis=0)
    else:
        valor_aux = a0-a1
        valor_aux = np.roll(W0,a2,axis=0)*valor_aux
        W = np.roll(W0,a1,axis=0)*(1-(a0-a1)) + valor_aux
        

        # W = np.roll(W0,a1)*(1-(a0-a1)) + np.roll(W0,a2)*(a0-a1)
    
    # -->===================================
    # --> Removal of aliased cyclic frequencies
    # -->===================================
    if 'trunc' in opt:
        W[ int(np.ceil(Nfft/2)+1+round(a0)):Nfft] = 0
    
    return(W)

def LiteSpectrogram(x, Window, Noverlap,Nfft,Fs = 1, WinType = 'hanning'):

    from scipy import signal
    
    if (np.size(Window) == 1):
        Window = GetWindow(WinType, Window)

        
    Window = Window[:]
    n = len(x)                  # --> Number of data points
    nwind = len(Window)         # --> length of window
    R = nwind - Noverlap        # --> block shift
    x = x[:]		
    K = np.fix((n-Noverlap)/(nwind-Noverlap))	# --> Number of windows
                                                
    # -->===================================
    # --> compute STFT
    # -->===================================
    index0 = 0
    index1 = nwind
   
    X = np.zeros( (int((Nfft/2)+1),int(K)), dtype='complex128' )
    for k in range(0, int(K)):

        
        Xw = np.fft.fft( np.multiply(Window, x[int(index0):int(index1)]), n=Nfft)		# --> Xw(f-a/2) or Xw(f-a)

        X[:,k] = Xw[0:int(Nfft/2+1)]
        index0 = index0+R
        index1 = index1+R
        
    f = np.arange(0,int(Nfft/2)+1)/Nfft*Fs
    t = np.arange(nwind/2, nwind/2+(K-1)*R, R)/Fs

    return (X, f, t)




def GetWindow(window_type, window_size):
    if window_type == 'hanning':
        Window = signal.get_window("hanning", window_size)
    elif window_type == 'hamming':
        Window = signal.get_window("hamming", window_size)
    elif window_type == 'blackman':
        Window = signal.get_window("blackman", window_size)
    elif window_type == 'kaiser':
        Window = signal.get_window(("kaiser", 15), window_size )
    elif window_type == 'gaussian':
         Window = signal.get_window(("gaussian", 80), window_size)
    elif window_type == 'chebwin':
        Window = signal.get_window(("chebwin", 80), window_size)
    else:
        Window = signal.get_window("hanning", window_size)
    return(Window)