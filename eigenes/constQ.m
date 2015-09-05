function cq= constQ(x, sparKernel)
fft_res = fft(x,size(sparKernel,1));
cq = fft_res * sparKernel;
end