function res = checkDiv(no,no2)

res = [];
for j = 2:no2
    if(rem(no, j)==0)
        j_is_divider = j
        res = [res j];
    end
        
    
end