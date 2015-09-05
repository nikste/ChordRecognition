function alignAllBeatles()
%output = '/home/nsteen/beatles/wav_aligned'
%output = '/home/nsteen/beatles_new/wav_aligned'
% fn = '/home/nsteen/beatles/AbbeyRoad/wav';
% TT = myls(strcat(fn,'/*.wav'));
% [T,O,S] = find_beatles_timeskew(TT,0,0,output);
% %skew_audiofiles()
% 
% fn = '/home/nsteen/beatles/forSaleRe/wav';
% TT = myls(strcat(fn,'/*.wav'));
% [T,O,S] = find_beatles_timeskew(TT,0,0,output);

%fn = '/home/nsteen/beatles/hardDaysNight/wav';
%TT = myls(strcat(fn,'/*.wav'));
%[T,O,S] = find_beatles_timeskew(TT,0,0,output);
% 
% fn = '/home/nsteen/beatles/help/wav';
% TT = myls(strcat(fn,'/*.wav'));
% [T,O,S] = find_beatles_timeskew(TT,0,0,output);
% 
% fn = '/home/nsteen/beatles/letItBe/wav';
% TT = myls(strcat(fn,'/*.wav'));
% [T,O,S] = find_beatles_timeskew(TT,0,0,output);
% 
% fn = '/home/nsteen/beatles/magicalMysteryTourRE/wav';
% TT = myls(strcat(fn,'/*.wav'));
% [T,O,S] = find_beatles_timeskew(TT,0,0,output);
% 
% fn = '/home/nsteen/beatles/revolver/wav';
% TT = myls(strcat(fn,'/*.wav'));
% [T,O,S] = find_beatles_timeskew(TT,0,0,output);
% 
% fn = '/home/nsteen/beatles/rubberSoul/wav';
% TT = myls(strcat(fn,'/*.wav'));
% [T,O,S] = find_beatles_timeskew(TT,0,0,output);
% 
% fn = '/home/nsteen/beatles/stPepper/wav';
% TT = myls(strcat(fn,'/*.wav'));
% [T,O,S] = find_beatles_timeskew(TT,0,0,output);
% 
% 
% fn = '/home/nsteen/beatles/zweieck/wav';
% TT = myls(strcat(fn,'/*.wav'));
% [T,O,S] = find_beatles_timeskew(TT,0,0,output);






%%%%%%new version!%%%%%

output = '/home/nsteen/beatles_new/wav_aligned'
fn = '/home/nsteen/beatles_new/orig/beatles_rubber_soul'
TT = myls(strcat(fn,'/*.wav'));
[T,O,S] = find_beatles_timeskew(TT,0,0,output);

doplot = 1;
find_beatles_timeskew(fullfile(output, [T, '.wav']),0,doplot);
drawnow;

fn = '/home/nsteen/beatles_new/orig/the beatles - 2009 - revolver (stereo remaster) [flac]'
TT = myls(strcat(fn,'/*.wav'));
[T,O,S] = find_beatles_timeskew(TT,0,0,output);

fn = '/home/nsteen/beatles_new/orig/the_beatles _2009_please_please_me'
TT = myls(strcat(fn,'/*.wav'));
[T,O,S] = find_beatles_timeskew(TT,0,0,output);

fn = '/home/nsteen/beatles_new/orig/the_beatles_2009_beatles_for_sale'
TT = myls(strcat(fn,'/*.wav'));
[T,O,S] = find_beatles_timeskew(TT,0,0,output);
fn = '/home/nsteen/beatles_new/orig/The Beatles - 2009 - Abbey Road (Stereo Remaster) [FLAC]'
TT = myls(strcat(fn,'/*.wav'));
[T,O,S] = find_beatles_timeskew(TT,0,0,output);
fn = '/home/nsteen/beatles_new/orig/The Beatles - 2009 - Help! (Stereo Remaster) [FLAC]'
TT = myls(strcat(fn,'/*.wav'));
[T,O,S] = find_beatles_timeskew(TT,0,0,output);
fn = '/home/nsteen/beatles_new/orig/The Beatles - 2009 - Sgt. Peppers Lonely Hearts Club Band (Stereo Remaster) [FLAC]'
TT = myls(strcat(fn,'/*.wav'));
[T,O,S] = find_beatles_timeskew(TT,0,0,output);
fn = '/home/nsteen/beatles_new/orig/the_beatles-a_hard_days_night-(remastered)-2009-sire'
TT = myls(strcat(fn,'/*.wav'));
[T,O,S] = find_beatles_timeskew(TT,0,0,output);
fn = '/home/nsteen/beatles_new/orig/The Beatles - Let It Be (1970)'
TT = myls(strcat(fn,'/*.wav'));
[T,O,S] = find_beatles_timeskew(TT,0,0,output);
fn = '/home/nsteen/beatles_new/orig/The Beatles - Magical Mystery Tour (1967)'
TT = myls(strcat(fn,'/*.wav'));
[T,O,S] = find_beatles_timeskew(TT,0,0,output);
fn = '/home/nsteen/beatles_new/orig/The_Beatles_White_Albums_Vinyl_Rip_Blackbird'
TT = myls(strcat(fn,'/*.wav'));
[T,O,S] = find_beatles_timeskew(TT,0,0,output);
fn = '/home/nsteen/beatles_new/orig/with_the_beatles'
TT = myls(strcat(fn,'/*.wav'));
[T,O,S] = find_beatles_timeskew(TT,0,0,output);


end