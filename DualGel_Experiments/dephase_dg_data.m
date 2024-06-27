clc;
clear all;
close all;

%% Options

figure_opt = false;
save_opt = true;

%% Loading in TI values
TI_vals = importdata('dualGel_TI.csv');

%% Loading and Shaping Data

repetitions = 100;
nTIs = length(TI_vals);
nTEs = 2048;

run_number = 86;

n_data = repetitions*nTIs*nTEs;

if run_number == 78
    file_add = 'fid';
else
    file_add = 'ser';
end

formatted_string = sprintf('run%d_%s',run_number,file_add);
f1D=fopen(formatted_string,'r');
data_C_2D=fread(f1D,[2,n_data],'int32');
fclose(f1D);

data_C_1D = complex(data_C_2D(1,:), data_C_2D(2,:));

unphased_dataset = reshape(data_C_1D, nTEs, nTIs, repetitions);

%% Clipping Bad Runs

unphased_dataset = unphased_dataset(:,:,2:end);

repetitions = repetitions - 1;

%% Calculating Phase Angle

calc_phase = angle(unphased_dataset)*180/pi;

if figure_opt
    figure
    subplot(1,2,1)
    plot(1:1:2048, calc_phase(:,1,1), 'k-')
    title('Echos of First TI')
    subplot(1,2,2)
    plot(1:1:2048, calc_phase(:,end,1), 'k-')
    title('Echos of Last TI')
end

%% Real and Imaginary Split

real_unphased_dataset = real(unphased_dataset);
imag_unphased_dataset = imag(unphased_dataset);

%% Plotting Initial Signals Across Noise Realizations

starting_points = squeeze(unphased_dataset(1,:,:));
start_point_mag = sqrt(real(starting_points).^2 + imag(starting_points).^2);

%This is the code to see the initial signal for each curve across all reps
if figure_opt    
    for iter_TI = 10:4:20
        figure;
    
        plot(1:1:repetitions,start_point_mag(iter_TI,:),'k-o')
    
        title(strcat("Starting Point Noise Repetitions for TI = ", string(TI_vals(nTIs - iter_TI + 1))))
        xlabel("Noise Realization")
        ylabel('Initial Magnitude')
    end
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

%% TI initial values

TI_initial_avg_values = mean(real_phased_dataset(1,:,:), 3);

figure;
plot(TI_vals, TI_initial_avg_values,'o-')
title("Average Initial Signal for Each TI")
xlabel("TI (ms)")
ylabel('Initial Signal')

%% Plotting Data
if figure_opt
    for iter_TI = 28:1:29
        figure;
        for iter_rep = 1:repetitions
            plot(1:1:2048,real_phased_dataset(:,iter_TI,iter_rep),'-')
            hold on
        end
        grid on
        title(strcat("100 Noise Repetitions for TI = ", string(TI_vals(iter_TI))))
        xlabel("TE")
        ylabel('Signal')
    
        figure;
        for iter_rep = 1:repetitions
            plot(1:1:2048,magnitudes(:,iter_TI,iter_rep),'-')
            hold on
        end
        grid on
        title(strcat("100 Noise Repetitions for TI = ", string(TI_vals(iter_TI))))
        xlabel("TE")
        ylabel('Magnitude')
    end
end
%% Average Signals and All Curves

average_signal = mean(real_phased_dataset, 3);

if figure_opt
    figure;
    for iTI = 1:length(TI_vals)
        plot(1:1:2048, average_signal(:,iTI),'-')
        hold on
    end
    grid on
    title(strcat("Avg Signal for Each TI"))
    xlabel("TE")
    ylabel('Signal')
end

%% Specific TI

TI_choice = 60;

figure;

plot(1:1:2048, average_signal(:,TI_choice),'-')

grid on
title(strcat("Avg Signal for TI = ", string(TI_vals(TI_choice))))
xlabel("TE")
ylabel('Signal')


%% All Angles
if figure_opt
    figure
    imagesc(theta_mat*180/pi)
    xlabel("Num Repetitions")
    ylabel("TIs")
    title("Theta (degrees)")
    colorbar
end
%% File Output

if save_opt
    output_fstring = sprintf('real_phased_run%d.mat', run_number);
    save(output_fstring,'real_phased_dataset')
end