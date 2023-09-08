clc
clear
close all

%% Loading in TI values
TI_vals = importdata('Data/TI_phantom_nullExp.txt');

%% Phasing with more accurate phase shift values

f1D=fopen('Data/ser','r');
data_C_2D=fread(f1D,[2,13107200],'int32');
fclose(f1D);

data_C_1D = complex(data_C_2D(1,:), data_C_2D(2,:));

TI_index = 55;
TI_oi = 64-TI_index+1;

repetitions = 100;
nTIs = 64;
nTEs = 2048;

TE_array = (1:1:2048)*0.4;
TE_array = TE_array(1:2048/2);

unphased_dataset = reshape(data_C_1D, nTEs, nTIs, repetitions);
unphased_dataset = unphased_dataset(1:2048/2,:,:);
one_signal = unphased_dataset(:,TI_oi,25);
TI_signal_used = TI_vals(TI_index);
first_point = one_signal(1);

square_signal = real(one_signal).^2 + imag(one_signal).^2;
figure;
plot(TE_array, square_signal)
title("Magnitudes")

theta = atan(imag(first_point)/real(first_point))
angles = imag(one_signal)./real(one_signal);

one_signal(1:20)

figure;
plot(angles)
% yline(theta)
title("Ratio of Imag to Real")

%We choose only the first signal
% first_angle = angles(1);
% 
% angle_diff = first_angle-theta
% if (first_angle-theta)<1e-5
%     warning("The angles are close")
% end

phased_data = one_signal*exp(-1i*theta);

figure;
subplot(2,2,1)
plot(TE_array, real(one_signal));
title('Real Unphased Data - Rep 1')
xlabel('Echo (ms)')

subplot(2,2,2)
plot(TE_array, real(phased_data));
title('Real Phased Data - Rep 1')
xlabel('Echo (ms)')

subplot(2,2,3)
plot(TE_array, imag(one_signal));
title('Imaginary Unphased Data - Rep 1')
xlabel('Echo (ms)')

subplot(2,2,4)
plot(TE_array, imag(phased_data));
title('Imaginary Phased Data - Rep 1')
xlabel('Echo (ms)')

%% Workspace

figure
plot(real(one_signal),imag(one_signal),'r*')
grid on
xlabel('Real')
ylabel('Imaginary')
title(string(TI_vals(TI_index)))

%%

figure
subplot(3,1,1)
plot(TE_array,imag(one_signal),'r*')
grid on
xlabel('TE')
xlim([350,400])
ylabel('Signed Magnitude')
title(strcat("Imaginary :: ", string(TI_vals(TI_index))))

subplot(3,1,2)
plot(TE_array,real(one_signal),'r*')
grid on
xlabel('TE')
xlim([350,400])
ylabel('Signed Magnitude')
title(strcat("Real :: ", string(TI_vals(TI_index))))

subplot(3,1,3)
plot(TE_array,real(one_signal)./imag(one_signal),'r*')
grid on
xlabel('TE')
xlim([350,400])
ylabel('Signed Ratio :: Real/Imag')
ylim([-2,2])
title(strcat("Ratio :: ", string(TI_vals(TI_index))))
