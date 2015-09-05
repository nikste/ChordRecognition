function callCreateFullDataset()
%matlabpool('open',6);
% createFullDataset('AbbeyRoad');
% createFullDataset('forSaleRe');
% createFullDataset('hardDaysNight');
% createFullDataset('help');
% createFullDataset('letItBe');
% createFullDataset('magicalMysteryTourRE');
% createFullDataset('revolver');
% createFullDataset('rubberSoul');
% createFullDataset('stPepper');
% createFullDataset('zweieck');
%matlabpool('close');


tic
createFullDatasetNew27Inv('01_-_Please_Please_Me')
toc

tic
createFullDatasetNew27Inv('02_-_With_the_Beatles')
toc

tic
createFullDatasetNew27Inv(strcat('03_-_A_Hard_Day',char(39),'s_Night'))
toc

tic
createFullDatasetNew27Inv('04_-_Beatles_for_Sale')
toc

tic
createFullDatasetNew27Inv('05_-_Help!')
toc

tic
createFullDatasetNew27Inv('06_-_Rubber_Soul')
toc

tic
createFullDatasetNew27Inv('07_-_Revolver')
toc

tic
createFullDatasetNew27Inv(strcat('08_-_Sgt._Pepper',char(39),'s_Lonely_Hearts_Club_Band'))
toc

tic
createFullDatasetNew27Inv('09_-_Magical_Mystery_Tour')
toc

tic
createFullDatasetNew27Inv('10CD1_-_The_Beatles')
toc

tic
createFullDatasetNew27Inv('10CD2_-_The_Beatles')
toc

tic
createFullDatasetNew27Inv('11_-_Abbey_Road')
toc

tic
createFullDatasetNew27Inv('12_-_Let_It_Be')
toc
end