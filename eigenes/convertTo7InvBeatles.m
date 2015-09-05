function [chord7Inv_string] = convertTo7InvBeatles(chord_string)

c_t = createConversionTable();




ind = find(strcmp(chord_string, {c_t{:,1}}));

chord7Inv_string = c_t{ind,2};
if(~strcmp(chord7Inv_string,chord_string))
disp(strcat(chord_string, ' mapped to: ', chord7Inv_string));
end
end