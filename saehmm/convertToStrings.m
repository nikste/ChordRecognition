function [res] = convertToStrings(input_number_mat)
%converts from mats number to cell strings


res = {};

for i=1:size(input_number_mat,1)    
    
    if(input_number_mat(i) >= 1000)
        res{i} = num2str(input_number_mat(i));
    end
    
    if(input_number_mat(i)  < 1000)
        res{i} = strcat('0',num2str(input_number_mat(i)));
    end
    
    if(input_number_mat(i)  < 100)
        res{i} = strcat('00',num2str(input_number_mat(i)));
    end
    
    
    if(input_number_mat(i) < 10)
        %'1234'
        % 1 -> 0001
        res{i} = strcat('000',num2str(input_number_mat(i)));
    end

end
end