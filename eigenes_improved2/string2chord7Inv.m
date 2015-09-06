function chord = string2chord7Inv(s)
chord = -1;
note = s;
ext = 'none';

% easy major chord
if(length(s) == 1)
    h = height(s);
    chord = h;
end
% major chord with deviation
if(length(s) == 2)
    h = height(s);
    chord = h;
end

%more complicated
if(length(s) > 2)

    if(s(2) == ':')
        h = height(s(1));
        mult = getMultiplier(s(3:end));
    end
    if(s(3) == ':')
            h = height(s(1:2));
        mult = getMultiplier(s(4:end));
    end
    chord = h + mult * 12;
end
end


%%% if s is 1 or 2 long it has to be a major (1..12) or none chord (0)
function h = height(s)

h = -1;
switch s
    case 'X' %mcgill dataset, for (voice) or something else i guess
        h = 0;
    case 'N'
        h = 0;
    case 'Cb'
        h = 12;
    case 'C'
        h = 1;
    case 'C#'
        h = 2;
        
        
    case 'Db'
        h = 2;
    case 'D'
        h = 3;
    case 'D#'
        h = 4;
        
    case 'Eb'
        h = 4;
    case 'E'
        h = 5;
    case 'E#'
        h = 6;
        
    case 'Fb'
        h = 5;
    case 'F'
        h = 6;
    case 'F#'
        h = 7;
        
    case 'Gb'
        h = 7;
    case 'G'
        h = 8;
    case 'G#'
        h = 9;
        
        
    case 'Ab'
        h = 9;
    case 'A'
        h = 10;
    case 'A#'
        h = 11;
        
    case 'Bb'
        h = 11;
    case 'B'
        h = 12;
    case 'B#'
        h = 1;
    otherwise
        h =-1;
        warning('unknown note!')
        s
  
end

end


function mult = getMultiplier(s)
mult = -1;
%% major minor
% maj
if(strcmp(s,'maj') == 1)
    mult = 0;
end

% min
if(strcmp(s,'min') == 1)
    mult = 1;
end

%% 7th
% maj7
if(strcmp(s,'maj7') == 1)
    mult = 2;
end
% min7
if(strcmp(s,'min7') == 1)
    mult = 3;
end
% 7
if(strcmp(s,'7') == 1)
    mult = 4;
end

%% inverse chords
% maj/3
if(strcmp(s,'maj/3') == 1)
    mult = 5;
end
% min/b3
if(strcmp(s,'min/b3') == 1)
    mult = 6;
end
% maj/5
if(strcmp(s,'maj/5') == 1)
    mult = 7;
end
% min/5
if(strcmp(s,'min/5') == 1)
    mult = 8;
end
%% inverse 7th
% maj7/3
if(strcmp(s,'maj7/3') == 1)
    mult = 9;
end
% min7/b3 
if(strcmp(s,'min7/b3') == 1)
    mult = 10;
end
% 7/3
if(strcmp(s,'7/3') == 1)
    mult = 11;
end
% maj7/5
if(strcmp(s,'maj7/5') == 1)
    mult = 12;
end
% min7/5
if(strcmp(s,'min7/5') == 1)
    mult = 13;
end
% 7/5
if(strcmp(s,'7/5') == 1)
    mult = 14;
end
% maj7/7
if(strcmp(s,'maj7/7') == 1)
    mult = 15;
end
% min7/b7
if(strcmp(s,'min7/b7') == 1)
    mult = 16;
end
% 7/b7
if(strcmp(s,'7/b7') == 1)
    mult = 17;
end


if(mult == -1)
    error(strcat('SOMETHING WENT WRONG:',s));
end

end






% %%%
% % will determine which chord type it is, in case there is a : after the
% % first element of the string
function type = checkMinor(s)
%possibilities 
%:min
%:maj(somenumber)
%:somenumber
%:dim
%:aug
%:sus
%:add
%(something)
%:hdim (whatever this is)
type = 100;
%min
if(s(1) == 'm' && s(2) == 'a' && s(3) == 'j')
    type = 0;
end
%maj
if(s(1) == 'm' && s(2) == 'i' && s(3) == 'n')
    type = 1;
end
%somenumber
if(s(1) == '1' || s(1) == '2' || s(1) == '3' || s(1) == '4' || s(1) == '5' || s(1) == '6' || s(1) == '7' || s(1) == '8' || s(1) =='9' || s(1) =='0')
    type = 0;
end
%dim
if(s(1) == 'd' && s(2) == 'i' && s(3) == 'm')
    type = -1; %none
end
%aug
if(s(1) == 'a' && s(2) == 'u' && s(3) == 'g')
    type = -1; %none
end
%sus
if(s(1) == 's' && s(2) == 'u' && s(3) == 's')
    type = -1; %none
end
%add
if(s(1) == 'a' && s(2) == 'd' && s(3) == 'd')
    type = -1; %none
end
if(s(1) == '(')
    type = -1; %none
end
if(s(1) == 'h' && s(2) == 'd' && s(3) == 'i' && s(4) =='m')
    type = -1;
end
end
% type = -2;
% if(s(1) == 'm' && s(2) == 'a' && s(3) =='j')
%     type = 0;
% elseif(s(1) == 'm' && s(2) == 'i' && s(3) =='n')
%     type = 1;
% elseif(s(1) == '1' ||s(1) == '2' ||s(1) == '3' ||s(1) == '4' ||s(1) == '5' || s(1) == '6' || s(1) == '7' || s(1) == '8' || s(1) == '9' )
%     type = 0;
% end
% if(s(1) == 's' && s(2) == 'u' && s(3) =='s')
%     type = -1;
% elseif(s(1) == 'a' && s(2) == 'u' && s(3) =='g')
%     type = -1;
% elseif(s(1) == 'd' && s(2) == 'i' && s(3) =='m')
%     type = -1;
% elseif(s(1) == '(')
%     type = -1;
% elseif(s(1) == 'h' && s(2) == 'd' && s(3) =='i' && s(4) == 'm')
%     type = -1;
% elseif(s(1) == 'a' && s(2) == 'd' && s(3) =='d')
%     type = 0;
% end

% if(length(s) > 1)
%     if(s(2) == 'i' || s(2) == 'I') % minor or diminshed
%         type = 1;
%     elseif(s(2) == 'a' || s(2) == 'A' || s(2) == 'u' || s(2) == 'U') % major or augmented
%         type = 0;
%     end
% end
% 
% if(s(1) == '1' ||s(1) == '2' ||s(1) == '3' ||s(1) == '4' ||s(1) == '5' || s(1) == '6' || s(1) == '7' || s(1) == '8' || s(1) == '9' )
%     type = 0;
% end
% 
% end