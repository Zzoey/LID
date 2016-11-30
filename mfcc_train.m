%Author : Vinay Chandragiri
%Clean-up
  clc;


F = dir('train_data/*');

for i=3:length(F)
     A = dir(fullfile('test_data',F(i).name,'*.wav'));
         for j=1:length(A)
             file_name=fullfile('test_data',F(i).name,A(j).name);
             %disp(file_name);
             [y,fs]= audioread(file_name);
             clear file_name;
             y = y/max(abs(y));
             y1 = mfcc_rasta_delta_pkm_v1(y,16000,13,26,20,10,0,0,5);
             save(fullfile('mfcc_test',F(i).name),'y1');
             clear y;
         end;
end;
