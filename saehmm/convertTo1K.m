function res = convertTo1K(input)
res = zeros(length(input),25);

for f = 1:length(input)
    %input(f)+1
    res(f,int8(input(f)+1)) = 1; 
end

end