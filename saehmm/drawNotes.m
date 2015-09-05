function [ w_notes ] = drawNotes( w )
%drawNotes(w): draws lines with note boundaries


%note to frequency: f_0 a = 440 hz, n half steps away from where you are,
% a = 2^1/12
a = 2^(1/12);
f_0 = 440;

Fs = 11025; % sample rate
N = 8192;

w_notes = w;

for n = -100:100
    % compute frequency
    f_n = f_0 * a^n;
    % compute bin
    bin_num = floor(f_n * 8192/11025); 
    % draw line if between 0 and 1500
    if(bin_num > 50 && bin_num < 1500)
        w_notes(:,bin_num) = max(max(w));
    end
    
end





end

