# RBFastSC
FastSC algorithm in python

Provides a fast algorithm for estimating the spectral correlation (or spectral coherence).
To be used for the detection and analysis of cyclostationary signals.
This algorithm was ported from the original matlab scripts developed by Jerome Antoni (link in reference).

It's possible to change the STFT window by modifing the parameter WindowType (default/original = 'hanning').

Possible values for WindowType: ['hanning', 'hamming', 'blackman', 'kaiser', 'gaussian', 'chebwin']

References:

Jerome Antoni (2022). Fast_SC(x,Nw,alpha_max,Fs,opt) (https://www.mathworks.com/matlabcentral/fileexchange/60561-fast_sc-x-nw-alpha_max-fs-opt), MATLAB Central File Exchange. Retrieved June 23, 2022. 
