#
#RBDSP_demo_RBFastSC.py
#


#=============================================================================
#Autor:         Rodrigo Barros
#Data:          09/01/2019
#Descrição:     Programa para ler o logo do saass e gerar relatorio

#=============================================================================

DESCRICAO =  """
\t=============================================================================
\tAutor:         Rodrigo Barros
\tData:          06/04/2021
\tVersao:        v1
\n      

\tDescrição:      
\t Esse script calcula Correlação Espectral ou a Coerência Espectral utilizando algoritmo FastSC.
\t É possível alterar a janela da STFT modificando o parâmetro WindowType = 'hanning'.
\t Valore possíveis: ['hanning', 'hamming', 'blackman', 'kaiser', 'gaussian', 'chebwin']
\t=============================================================================
"""


print(DESCRICAO)

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import sys
sys.path.append("./")


import RBDSP_Fast_FSC_v4 as RBDSP_Ciclo 



 
def plotSCoData(sco_alpha, sco_freq, sco_data, title = "", img = False):
    
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    
   
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    sco_alpha, sco_freq = np.meshgrid(sco_alpha, sco_freq)
    surf = ax.plot_surface( sco_alpha, sco_freq, sco_data,
                            cmap=cm.coolwarm,
                            rstride = 5,
                            cstride = 5,
                            linewidth=0,
                            antialiased=True)
   
    if(title == ""):
        ax.set_title('Cyclic Coherence Spectral')          # add a title    
    else:
        ax.set_title(title)          # add a title    
        
    ax.set(xlabel = 'Cyclic freq [Hz]', ylabel = 'Spectral freq [Hz]')
    
    ax.grid()                          # ligar a linhas de grid
    
    plt.show()




opt         = { "abs": 0, "calib": 1, "coh": 0 }
alpha_max   = 51.9      # --> maximum cyclic frequency to scan (in Hz)
Nw          = 1024               # --> window length (number of samples)
f_sample    = 8000
n_samples   = 10*f_sample
tempo       = n_samples/f_sample
f_shaft     = 5;
t           = np.arange(0,n_samples,1)/f_sample;

base1       = np.cos(2*np.pi*20*t)
base2       = np.cos(2*np.pi*5*t)

carrier1    = np.cos(2*np.pi*1000*t)
carrier2    = np.cos(2*np.pi*500*t)
carrier3    = np.cos(2*np.pi*1500*t)
carrier4    = np.cos(2*np.pi*2500*t)

signal = base1*(carrier1+carrier2+carrier3+carrier4)

carrier1    = np.cos(2*np.pi*1200*t)
carrier2    = np.cos(2*np.pi*700*t)
carrier3    = np.cos(2*np.pi*1700*t)
carrier4    = np.cos(2*np.pi*2700*t)

signal = signal + base2*(carrier1+carrier2+carrier3+carrier4) 


opt["coh"]  = 1             # -->compute sepctral coherence? (yes=1, no=0)

sco_data, sco_alpha, sco_freq, __, __, Nv = RBDSP_Ciclo.Fast_SC(signal, Nw, alpha_max, f_sample, opt)

plotSCoData(sco_alpha[2:], sco_freq, (np.abs(sco_data[:,2:])) )

