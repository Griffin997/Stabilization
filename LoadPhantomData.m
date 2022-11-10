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
data_C_2D=fread(f1D,[2,13107200],'int32');
fclose(f1D);

data_C_1D = complex(data_C_2D(1,:), data_C_2D(2,:));

repetitions = 100;
nTIs = 64;
nTEs = 2048;

unphased_dataset = reshape(data_C_1D, nTEs, nTIs, repetitions);
calc_phase = angle(unphased_dataset)*180/pi;
% calc_phase = reshape(data_phase, nTEs, nTIs, repetitions);

figure;
subplot(1,2,1)
plot(1:1:2048, calc_phase(:,1,1), 'k-')
title('Echos of First TI')
subplot(1,2,2)
plot(1:1:2048, calc_phase(:,end,1), 'k-')
title('Echos of Last TI')

%We choose 300 because the lower signal-to-noise ratio occurs after 300 so
%that we wouldn't get a good average
mean_calc_phase = squeeze(mean(calc_phase(1:300,:,:)));

phase_shift = -1*mean(mean_calc_phase(end,:));

phased_complex_ser = data_C_1D*complex(cos(phase_shift*pi/180),sin(phase_shift*pi/180));

figure;
subplot(2,2,1)
plot(real(data_C_1D(1:(2048*64))));
title('Real Unphased Data - Rep 1')
xlabel('Echo Intensity')

subplot(2,2,2)
plot(real(phased_complex_ser(1:(2048*64))));
title('Real Phased Data - Rep 1')
xlabel('Echo Intensity')

subplot(2,2,3)
plot(imag(data_C_1D(1:(2048*64))));
title('Imaginary Unphased Data - Rep 1')
xlabel('Echo Intensity')

subplot(2,2,4)
plot(imag(phased_complex_ser(1:(2048*64))));
title('Imaginary Phased Data - Rep 1')
xlabel('Echo Intensity')

phased_dataset=reshape(phased_complex_ser, nTEs, nTIs, repetitions);

signal_dataset = real(phased_dataset);

avg_signal = mean(signal_dataset,3);

figure
plot(1:1:2048, avg_signal,'k-')
xlabel('TE')
ylabel('Signal')
title("Phased")

% save("Data/phased_dataset.mat",'phased_dataset')

signal_dataset = real(unphased_dataset);

avg_signal = mean(signal_dataset,3);

figure
plot(1:1:2048, avg_signal,'k-')
xlabel('TE')
ylabel('Signal')
title("Unphased")