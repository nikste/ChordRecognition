function res = convertTo1K7Inv(input)
res = zeros(length(input),12*18+1);

for f = 1:length(input)
    %input(f)+1
    res(f,input(f)+1) = 1; 
end

end