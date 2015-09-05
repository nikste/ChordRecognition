function chord = string2chord(s)
chord = -1;
note = s;
ext = 'none';



%with addition e.g. Eb:something or Eb(something)

%examples:
%   E
%   Eb
%   E:something
%   E/something
%   E(something)
%   Eb:something
%   Eb(something)
%   Eb/something

if(length(s) > 2)
%Eb:something || Eb(something) || Eb/something
    if(s(3) == ':')
        note = s(1:2);
        ext = s(4:length(s));
        is_minor = checkMinor(ext);
        if(is_minor < 0)
            chord = 0;
        else
            h = height(note);
            chord = h + is_minor*12;
        end
    elseif(s(3) == '(')% this is non chord
%         note = s(1:2);
%         ext = s(4:length(s));
%         is_minor = checkMinor(ext);
%         h = height(note);
%         chord = h + is_minor*12;
        chord = 0;
    elseif(s(3) == '/') %should this be major?
        note = s(1:2);
        %ext = s(4:length(s));
        %is_minor = checkMinor(ext);
        h = height(note);
        chord = h ;%+ is_minor*12;      
        
        
%E:something || E(something) || E/something
    
    elseif(s(2) == ':')
        note = s(1);
        ext = s(3:length(s));
        is_minor = checkMinor(ext);    
        
        if(is_minor < 0)
            chord = 0;
        else
            h = height(note);
            chord = h + is_minor*12;
        end
        
    elseif(s(2) == '(')
        %note = s(1);
        %ext = s(3:length(s));
        chord = 0;
    elseif(s(2) == '/')
        note = s(1);
        %ext = s(3:length(s));
        h = height(note);
        chord = h;
    end
%E || Eb
else
    chord = height(s);
end
%check 
end

    



% if(length(s) > 2)
%     if(s(3) == ':')
%          note = s(1:2);
%          ext = s(4:length(s));
%          is_minor = checkMinor(ext);
%          h = height(note);
%          if(is_minor >= 0)
%             chord = height(note) + is_minor*12;
%          elseif(is_minor == -1)
%             chord = 0;
%          end
%             %check major
%      elseif(s(3) == '/')
%         note = s(1:2);
%         ext = s(4:length(s));
%         %check major
%         is_minor = 0; %checkMinor(ext);
%         if(is_minor >= 0)
%             chord = height(note) + is_minor*12;
%         elseif(is_minor == -1)
%             chord = 0;
%         end
%     elseif(s(3) == '(')
%         note = s(1:2);
%         ext = s(4:length(s));
%         %check major
%         is_minor = 0; %checkMinor(ext);
%         if(is_minor >= 0)
%            chord = height(note) + is_minor*12;
%         elseif(is_minor == -1)
%            chord = 0;
%         end
%     end
% end
% if(length(s) > 1)
%     if(s(2) == ':')
%         note = s(1);
%         ext = s(3:length(s));
%         %check major
%         is_minor = checkMinor(ext);
%         if(is_minor >= 0)
%             chord = height(note) + is_minor*12;
%         elseif(is_minor == -1)
%             chord = 0;
%         end
%     elseif(s(2) == '/')
%         note = s(1);
%         ext = s(3:length(s));
%         %check major
%         is_minor = 0; %checkMinor(ext);        
%         if(is_minor >= 0)
%             chord = height(note) + is_minor*12;
%         elseif(is_minor == -1)
%             chord = 0;
%         end
%     elseif(s(2) == '(')
%         note = s(1)
%         ext = s(3:length(s));
%         %check major
%         is_minor = 0; %checkMinor(ext);
%         if(is_minor >= 0)
%             chord = height(note) + is_minor*12;
%         elseif(is_minor == -1)
%             chord = 0;
%         end
%     else % E#
%         chord = height(note);
%     end
% else
%     chord = height(s);
% end
%check height 0 for none, otherwise 1..12 for major




%note 
%ext

%'classified as'
%chord %something went wrong!




% switch length(s)
%     case 1
%         chord = whichMajor(s)
%     case 2
%         chord  = whichMajor(s)
% 
%     otherwise
%         chord = -1
%         warning('no chord type found');
%         s
% end





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