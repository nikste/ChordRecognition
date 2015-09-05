function executeCQraw()

minFreq =  32,703;%220;%27.5;174,6%
maxFreq = 1975,53;%880*2;
fs = 44100;
bins = 12;
Q = 1/(2^(1/bins)-1)
fftlen = 2^nextpow2( ceil(Q*fs/minFreq) )
sparKernel = sparseKernel(minFreq,maxFreq,bins,44100,0.00005);



'wavread!'
lab = importdata('/media/owner/Festpladde/uni/masterAI/MasterThesis/Datasets/beatles/AbbeyRoad/lab/1.lab');
'labread'

l = importdata('/media/owner/Festpladde/uni/masterAI/MasterThesis/Datasets/beatles/AbbeyRoad/train/1.dataP');
'old pcp read'

description = cell(length(lab),3);

for i = 1:length(lab)

    aux = strread(lab{i},'%s','delimiter',' ');
    for j=1:3
        description(i,j) = aux(j);
    end
end

end