function [ res ,res_end, more_ffts,more_gts] = computePureChordsActivation( nn )
%VISUALIZEPURECHORDS Summary of this function goes here
%   Detailed explanation goes here
input_folder_c = '/home/nsteen/onlychords/Guitar_Only_11025/a/Guitar_Only/c';
input_folder_d = '/home/nsteen/onlychords/Guitar_Only_11025/a/Guitar_Only/d';
input_folder_e = '/home/nsteen/onlychords/Guitar_Only_11025/a/Guitar_Only/e';
input_folder_f = '/home/nsteen/onlychords/Guitar_Only_11025/a/Guitar_Only/f';
input_folder_g = '/home/nsteen/onlychords/Guitar_Only_11025/a/Guitar_Only/g';
input_folder_a = '/home/nsteen/onlychords/Guitar_Only_11025/a/Guitar_Only/a';

input_folder_dm= '/home/nsteen/onlychords/Guitar_Only_11025/a/Guitar_Only/dm';
input_folder_em= '/home/nsteen/onlychords/Guitar_Only_11025/a/Guitar_Only/em';
input_folder_am = '/home/nsteen/onlychords/Guitar_Only_11025/a/Guitar_Only/am'; 
input_folder_bm = '/home/nsteen/onlychords/Guitar_Only_11025/a/Guitar_Only/bm'; 


files_c = myls(strcat(input_folder_c,'/*.wav'));
files_d = myls(strcat(input_folder_d,'/*.wav'));
files_e = myls(strcat(input_folder_e,'/*.wav'));
files_f = myls(strcat(input_folder_f,'/*.wav'));
files_g = myls(strcat(input_folder_g,'/*.wav'));
files_a = myls(strcat(input_folder_a,'/*.wav'));

files_dm = myls(strcat(input_folder_dm,'/*.wav'));
files_em = myls(strcat(input_folder_em,'/*.wav'));
files_am = myls(strcat(input_folder_am,'/*.wav'));
files_bm = myls(strcat(input_folder_bm,'/*.wav'));

%copy first layer of neural network:

%nn2 = struct('size',[nn.size(1) nn.size(2)],'n',2,'activation_function',nn.activation_function,'learningRate',nn.learningRate,'momentum',nn.momentum,'scaling_learningRate',nn.scaling_learningRate,'weightPenaltyL2',nn.weightPenaltyL2,'nonSparsityPenalty',nn.nonSparsityPenalty,'sparsityTarget',nn.sparsityTarget,'inputZeroMaskedFraction',nn.inputZeroMaskedFraction,'dropoutFraction',nn.dropoutFraction,'testing',nn.testing,'W',nn.W{1},'vW',nn.vW{1},'p',{nn.p{1} nn.p{2}},'a',{nn.a{1} nn.a{2}},'L',nn.L,'dW',{nn.dW{1}})
nn2 = nnsetup([nn.size(1) nn.size(2)]);
nn2.W{1} = nn.W{1};
nn2.activation_function = nn.activation_function
%nn2.a{1} = nn.a{1};
%nn2.e = nn.e;
%output,nn.output

files_cont = {};
files_cont{1} = files_c;
files_cont{2} = files_d;
files_cont{3} = files_e;
files_cont{4} = files_f;
files_cont{5} = files_g;
files_cont{6} = files_a;

files_cont{7} = files_dm;
files_cont{8} = files_em;
files_cont{9} = files_am;
files_cont{10} = files_bm;

% l = length(files_c)+length(files_d)+length(files_e)+length(files_f)+length(files_g)+length(files_a);
% l = l+length(files_dm)+length(files_em)+length(files_am)+length(files_bm);
l = 0;
for i=1:length(files_cont)
    l = l + length(files_cont(i));
end
more_gts = zeros(l,1);
gt = [1 3 5 6 8 10 15 17 22 24];
ctr = 1;
for i=1:length(files_cont)
    for j=1:length(files_cont(i))
        more_gts(ctr,1) = gt(i);
        ctr = ctr+1;
    end
end
res = zeros(l,200);
res_end = zeros(l,25);
counter = 1;
more_ffts = zeros(l,1500);
disp('compputing first layer')
for chord_type = 1:length(files_cont)
    chord_files = files_cont{chord_type};
    disp(strcat(num2str(chord_type),' of  ',num2str(length(files_cont))));
   for chord = 1: length(chord_files)
       d = wavread(chord_files{chord}); 
       f = fft(d,8192);
       f = abs(f(1:1500,:));
       f = f';
       f = normr(sqrt(f));
       more_ffts(counter,:) = f;
       %disp(files_a{as})
       %size(f)
       %f(1501,:) = 0;
       %disp(strcat('size f:',num2str(size(f))));
       %disp(strcat('size nn.W{1}:',num2str(size(nn.W{1}'))));
       %res(as,:) = f * nn.W{1}';
       nn2.testing = 1;
       nn2 = nnff(nn2, f, zeros(size(f,1), nn2.size(end)));
       nn2.testing = 0;
       res(counter,:) = nn2.a{end};
       counter = counter +1;
   end
end

disp('computing complete activation!')
counter = 1;
for chord_type = 1:length(files_cont)
    chord_files = files_cont{chord_type};
    
    disp(strcat(num2str(chord_type),' of  ',num2str(length(files_cont))));
   for chord = 1: length(chord_files)
       d = wavread(chord_files{chord}); 
       f = fft(d,8192);
       f = abs(f(1:1500,:));
       f = f'; 
       f = normr(sqrt(f));

       %disp(files_a{as})
       %size(f)
       %f(1501,:) = 0;
       %disp(strcat('size f:',num2str(size(f))));
       %disp(strcat('size nn.W{1}:',num2str(size(nn.W{1}'))));
       %res(as,:) = f * nn.W{1}';
       nn.testing = 1;
       nn = nnff(nn, f, zeros(size(f,1), nn.size(end)));
       nn.testing = 0;
       res_end(counter,:) = nn.a{end};
       counter = counter +1;
   end
end


end

