%%

f1D=fopen('Data/2dseq','r');
data_mag_1D =fread(f1D,'int16');
fclose(f1D);

repetitions = 100;
nTIs = 64;
nTEs = 2048;

data_mag = reshape(data_mag_1D, nTEs, nTIs, repetitions);

slice = data_mag(:,:,1);

avg_slice = mean(data_mag,3);
max_slice = max(data_mag,[],3);
min_slice = min(data_mag,[],3);
range = max_slice-min_slice;

figure
plot(1:1:2048,avg_slice(:,:))
ylim([0,10000])
xlim([0,200])

%%

f1D=fopen('Data/ser','r');
data_C_1D=fread(f1D,'int16');
fclose(f1D);

repetitions = 100;
nTIs = 64;
nTEs = 2048;

data_C = reshape(data_C_1D, 4, nTEs, nTIs, repetitions);

slice_C = data_C(:,1,:,1);
