clc
clear
close all

%% Loading in TI values
TI_vals = importdata('Data/TI_phantom_nullExp.txt');

%% Loading and Shaping Data

f1D=fopen('Data/ser','r');
data_C_2D=fread(f1D,[2,13107200],'int32');
fclose(f1D);

data_C_1D = complex(data_C_2D(1,:), data_C_2D(2,:));

repetitions = 100;
nTIs = 64;
nTEs = 2048;

unphased_dataset = reshape(data_C_1D, nTEs, nTIs, repetitions);

%Removing the first noise realization at every combo - shows TR
%interactions
unphased_dataset = unphased_dataset(:,:,2:end);

repetitions = repetitions - 1;

real_unphased_dataset = real(unphased_dataset);
imag_unphased_dataset = imag(unphased_dataset);

save("Data/unphased_dataset.mat",'unphased_dataset')

save("Data/real_unphased_dataset.mat",'real_unphased_dataset')
save("Data/imag_unphased_dataset.mat",'imag_unphased_dataset')

starting_points = squeeze(unphased_dataset(1,:,:));
start_point_mag = sqrt(real(starting_points).^2 + imag(starting_points).^2);

for iter_TI = 28:1:30
    figure;
    
    plot(1:1:repetitions,start_point_mag(iter_TI,:),'k-o')

    title(strcat("Starting Point Noise Repetitions for TI = ", string(TI_vals(nTIs - iter_TI + 1))))
    xlabel("Noise Realization")
    ylabel('Initial Magnitude')
end


%% Processing Data

phased_dataset = zeros(size(unphased_dataset));
magnitudes = zeros(size(unphased_dataset));
theta_mat = zeros(nTIs, repetitions);

for iter_TI = 1:nTIs
    for iter_rep = 1:repetitions
        one_signal = unphased_dataset(:,iter_TI,iter_rep);
        [abs_max, max_index] = max(abs(one_signal));
        max_point = one_signal(max_index);

        square_signal = sqrt(real(one_signal).^2 + imag(one_signal).^2);
        theta = atan(imag(max_point)/real(max_point));

        phased_signal = one_signal*exp(-1i*theta);

        magnitudes(:,iter_TI, iter_rep) = square_signal;
        theta_mat(iter_TI, iter_rep) = theta;
        phased_dataset(:,iter_TI, iter_rep) = phased_signal;
    end
end

real_phased_dataset = real(phased_dataset);

%% Plotting Data

for iter_TI = 28:1:29
    figure;
    for iter_rep = 1:repetitions
        plot(1:1:2048,real_phased_dataset(:,iter_TI,iter_rep),'-')
        hold on
    end
    grid on
    title(strcat("100 Noise Repetitions for TI = ", string(TI_vals(nTIs - iter_TI + 1))))
    xlabel("TE")
    ylabel('Signal')

    figure;
    for iter_rep = 1:repetitions
        plot(1:1:2048,magnitudes(:,iter_TI,iter_rep),'-')
        hold on
    end
    grid on
    title(strcat("100 Noise Repetitions for TI = ", string(TI_vals(nTIs - iter_TI + 1))))
    xlabel("TE")
    ylabel('Magnitude')

end

%% All Angles

figure
imagesc(theta_mat*180/pi)
xlabel("Num Repetitions")
ylabel("TIs")
title("Theta (degrees)")
colorbar

%% Saving Data

save("Data/real_phased_dataset.mat",'real_phased_dataset')