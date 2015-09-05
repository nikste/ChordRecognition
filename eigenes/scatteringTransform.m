function output = scatteringTransform(filename)



filename = '/media/owner/Festpladde/uni/masterAI/MasterThesis/Datasets/beatles/AbbeyRoad/wav/1.wav'
sounddata = wavread(filename)
%strcat
ls = length(sounddata);

N = length(sounddata); %length of the signal
T = 2^15;%length(dat); % pow(2,15) step size?! averaging window (set to 4096?! and then load frames progressively)
fftLen = N;
filt_opt = default_filter_options('audio', T);

Wop = wavelet_factory_1d(N, filt_opt);

% sounddata = [sounddata(1:fftLen/2,1); sounddata; sounddata(ls - fftLen/2: ls,1)];

%output = []
output = scat(sounddata,Wop);
% for f=1:2048:2%length(sounddata)-fftLen
%     
%     complete = f/length(sounddata) * 100.0
%     output = scat(sounddata(f:f+fftLen),Wop);
%     %output = [output S];
% 
% end
end
