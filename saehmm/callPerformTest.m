function [] = callPerformTest(first_layer,second_layer,multi_res,pcp_pretrain,id)
	%display command line arguments:
	disp(strcat('first layer:',first_layer));
	disp(strcat('second layer:',second_layer));
	disp(strcat('multi resolution SDAE:',multi_res));
	disp(strcat('pcp intermediate targe:',pcp_pretrain));


	%convert command line arguments:

	first_layer = str2num(first_layer);
	second_layer = str2num(second_layer);
	multi_res = str2num(multi_res);
	pcp_pretrain = str2num(pcp_pretrain);



	%first_layer = 200;
	%second_layer = 0;
	%multi_res = 0;
	%pcp_pretrain = 0;
	[ output_args ] = performTest(first_layer,second_layer, multi_res, pcp_pretrain,id)

	disp(mat2str(output_args));


end
