function [chordsAsString] = Num2Chord(nums)
chordsAsString = {};

% from to number
% chord = h + 12 * mult
% 1-12 maj
for i=1:size(nums,1)
   chordsAsString = [chordsAsString;convertToString(nums(i,:))] ;
end

end



function row = convertToString(rowIn)

chordNum = rowIn(3);

if chordNum == 0
    row = [num2str(rowIn(1)) '  ' num2str(rowIn(2)) '  ' 'N'];
    return
end


c = mod(rowIn(3) - 1 ,12);
s = '';
switch c
    case 0
        s = 'C';
    case 1
        s = 'C#';
    case 2
        s = 'D';
    case 3
        s = 'D#';
    case 4
        s = 'E';
    case 5
        s = 'F';
    case 6
        s = 'F#';
    case 7    
        s = 'G';
    case 8    
        s = 'G#';
    case 9    
        s = 'A';
    case 10    
        s = 'A#';
    case 11    
        s = 'B';
end       

modifier = floor((rowIn(3) - 1)/12);

m = findExtension(modifier);
row = [num2str(rowIn(1)) '  ' num2str(rowIn(2)) '  ' strcat(s,':',m)];
end


function mult = findExtension(s)
mult = '';
%% major minor
% maj
if(s == 0)
    mult = 'maj';
end

% min
if(s == 1)
    mult = 'min';
end

%% 7th
% maj7
if(s == 2)
    mult = 'maj7';
end
% min7
if(s == 3)
    mult = 'min7';
end
% 7
if(s == 4)
    mult = '7';
end

%% inverse chords
% maj/3
if(s == 5)
    mult = 'maj/3';
end
% min/b3
if(s == 6)
    mult = 'min/b3';
end
% maj/5
if(s == 7)
    mult = 'maj/5';
end
% min/5
if(s == 8)
    mult = 'min/5';
end
%% inverse 7th
% maj7/3
if(s == 9)
    mult = 'maj7/3';
end
% min7/b3 
if(s == 10)
    mult = 'min7/b3';
end
% 7/3
if(s == 11)
    mult = '7/3';
end
% maj7/5
if(s == 12)
    mult = 'maj7/5';
end
% min7/5
if(s == 13)
    mult = 'min7/5';
end
% 7/5
if(s == 14)
    mult = '7/5';
end
% maj7/7
if(s == 15)
    mult = 'maj7/7';
end
% min7/b7
if(s == 16)
    mult = 'min7/b7';
end
% 7/b7
if(s == 17)
    mult = '7/b7';
end


if(mult == -1)
    error(strcat('SOMETHING WENT WRONG:',s));
end

end