function executeCQ(wavfolder)


minFreq =  32,703;%220;%27.5;
maxFreq = 1975,53;%880*2;
fs = 44100;
bins = 12;
sparKernel = sparseKernel(minFreq,maxFreq,bins,44100,0.00005);

Q= 1/(2^(1/bins)-1); 
fftlen = 2^nextpow2( ceil(Q*fs/minFreq) )
x = wavread('/media/owner/Festpladde/uni/masterAI/MasterThesis/Datasets/beatles/AbbeyRoad/wav/1.wav');
filler = zeros(32768/2);

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


small_legend = [1:12];
legend = [1:72];
y = [1:72];
%p = plot(legend,y)
%set(p,'YDataSource','y');
plot(legend,y)
framectr = 1;
linecounter = 1;
secondscounter = 32768/2/44100;%half of window
for i=1:2048:length(x)-32768 
    %i
    if(secondscounter > str2double(description(linecounter,2)))
        linecounter = linecounter+1;
    end
    current = x(i:i+32768);
    current = current';
    y = constQ(current,sparKernel);
    %refreshdata
    %plot(legend,abs(y));
    % add bins every 12th bin together
    pcp = zeros(12);
    for base_bin=1:12
        for bin = base_bin:12:72
        pcp(base_bin) = pcp(base_bin) + abs(y(base_bin));
        end
    end
    
    %plot(small_legend,bar(pcp));
    pcp = pcp/sum(pcp);
    
    
    cont = [pcp l(ceil((i+32768/2)/2048),:)'];
    bar(cont);
    set(gca,'XTickLabel',{'A','A#','B','C','C#','D','D#','E','F','F#','G','G#'})
    time = strcat(num2str(secondscounter),'<',description(linecounter,2));
    tit = strcat(description(linecounter,3),'   \t  ',time)
    title(tit);
    drawnow;
    pause(0.5);
    framectr = framectr+1;
    secondscounter = secondscounter + 2048/44100;
end
end