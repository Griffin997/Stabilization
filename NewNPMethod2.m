%function NewNPMethod2

clear all;

NumTrials = 5000;
NumBins = 100;
PLOT = 0; 
% PLOT = 1 --> plot exponential curves along with data and display individual parameter sets
% PLOT = 0 --> suppress all screen output for each trial

% bin limits. Mean values are taken only for parameters within these limits.
lower = 2;
upper = 150;

c1_m = zeros(1,NumTrials);
c2_m = zeros(1,NumTrials);
T21_m = zeros(1,NumTrials);
T22_m = zeros(1,NumTrials);
T11_m = zeros(1,NumTrials);
T12_m = zeros(1,NumTrials);

c1_b = zeros(1,NumTrials);
c2_b = zeros(1,NumTrials);
T21_b = zeros(1,NumTrials);
T22_b = zeros(1,NumTrials);
T11_b = zeros(1,NumTrials);
T12_b = zeros(1,NumTrials);

for i=1:NumTrials
    sprintf('i = %d/%d',i,NumTrials)
    [X_mono,X_bi,c1,c2,T11,T21,T12,T22] = getX(PLOT);
    c1_m(i) = X_mono(1); 
    c2_m(i) = X_mono(2); 
    T21_m(i) = X_mono(3); 
    T22_m(i) = X_mono(4); 
    T11_m(i) = X_mono(5); 
    T12_m(i) = X_mono(6);
    
    c1_b(i) = X_bi(1); 
    c2_b(i) = X_bi(2); 
    T21_b(i) = X_bi(3); 
    T22_b(i) = X_bi(4); 
    T11_b(i) = X_bi(5); 
    T12_b(i) = X_bi(6);
end

% clean data: only consider data points in [0 150]
T21_m = T21_m( T21_m >= lower & T21_m <= upper );
T21_b = T21_b( T21_b >= lower & T21_b <= upper );
T22_m = T22_m( T22_m >= lower & T22_m <= upper );
T22_b = T22_b( T22_b >= lower & T22_b <= upper );

subplot(2,2,1);
histogram(T21_m,NumBins,'BinLimits',[lower upper]); xlabel('T21'); ylabel('Counts');
mu = mean(T21_m); line(T21*[1 1],ylim,'Color','red'); line(mu*[1 1],ylim);
tit = sprintf('BIC: Exact T21 = %2.3f, mean = %2.3f',T21,mu); title(tit);
xlim([lower upper]);

subplot(2,2,2);
histogram(T21_b,NumBins,'BinLimits',[lower upper]); xlabel('T21'); ylabel('Counts');
mu = mean(T21_b); line(T21*[1 1],ylim,'Color','red'); line(mu*[1 1],ylim);
tit = sprintf('Conventional: Exact T21 = %2.3f, mean = %2.3f',T21,mu); title(tit);
title(tit);
xlim([lower upper]);

subplot(2,2,3);
histogram(T22_m,NumBins,'BinLimits',[lower upper]); xlabel('T22'); ylabel('Counts');
mu = mean(T22_m); line(T22*[1 1],ylim,'Color','red'); line(mu*[1 1],ylim);
tit = sprintf('BIC: Exact T22 = %2.3f, mean = %2.3f',T22,mu); title(tit);
xlim([lower upper]);

subplot(2,2,4);
histogram(T22_b,NumBins,'BinLimits',[lower upper]); xlabel('T22'); ylabel('Counts');
mu = mean(T22_b); line(T22*[1 1],ylim,'Color','red'); line(mu*[1 1],ylim);
tit = sprintf('Conventional: Exact T22 = %2.3f, mean = %2.3f',T22,mu); title(tit);
xlim([lower upper]);


legend('Data','Exact','Sample Mean');

%end

function[X_mono,X_bi,c1,c2,T11,T21,T12,T22] = getX(PLOT)



c1 = 0.3; % first exponential
T11 = 30;
T21 = 60;

c2 = 0.6; % second exponential
T12 = 45;
T22 = 70;

% nullpoints
TI_1_star = T11*log(2);
TI_2_star = T12*log(2);

% sampled TI values (one for each data set)
TI1 = TI_1_star;
TI2 = TI_2_star;
TI3 = 0;

SNR = 100;
N = 30;
t = linspace(0.01,200,N);

S1 = Master_biexp(t,c1,c2,T11,T12,T21,T22,TI1) + 1/SNR * randn(1,N); % data1
S2 = Master_biexp(t,c1,c2,T11,T12,T21,T22,TI2) + 1/SNR * randn(1,N); % data2
S3 = Master_biexp(t,c1,c2,T11,T12,T21,T22,TI3) + 1/SNR * randn(1,N); % data3

% RSS for MONO exponential fits to the data
% T21vec = linspace(0,200,201);
% T22vec = linspace(50,300,251);
% RSS = zeros(100,100);
% for i = 1:length(T21vec)
%     for j = 1:length(T22vec)
%         d2_star = c2*(1-2*exp(-TI_1_star/T12));
%         d1_star = c1*(1-2*exp(-TI_2_star/T11));
%         RSS(i,j) = objective_function1([d2_star d1_star T22vec(j) T21vec(i)],S1,S2,t);
%     end
% end


c10 = rand(1,1);
c20 = rand(1,1);
T210 = 2 + 148*rand(1,1);
T220 = 2 + 148*rand(1,1);
T110 = 2 + 148*rand(1,1);
T120 = 2 + 148*rand(1,1);

X0 = [c10 c20 T210 T220 T110 T120];

options = optimset('MaxFunEvals',20000,'MaxIter',20000);

X = fminsearch(@(X) objective_function1(X,S1,S2,S3,t,TI1,TI2,TI3), X0,options);
c1_est = X(1); c2_est = X(2); T21_est = X(3); T22_est = X(4); T11_est = X(5); T12_est = X(6);
X_mono = X;

if PLOT == 1

    subplot(1,2,1); 
    plot(t,S1,'rs',t,S2,'bs',t,S3,'gs'); hold on;
    plot(t,biexp(t,     0*(1-2*exp(-TI1/T11_est)), c2_est*(1-2*exp(-TI1/T12_est)),T21_est,T22_est),'r-',...
        t,biexp(t,c1_est*(1-2*exp(-TI2/T11_est)),      0*(1-2*exp(-TI2/T12_est)),T21_est,T22_est),'b-',...
        t,biexp(t,c1_est*(1-2*exp(-TI3/T11_est)), c2_est*(1-2*exp(-TI3/T12_est)),T21_est,T22_est),'g-');
    title('BIC');
end

X = fminsearch(@(X) objective_function2(X,S1,S2,S3,t,TI1,TI2,TI3), X0, options);
c1_est = X(1); c2_est = X(2); T21_est = X(3); T22_est = X(4); T11_est = X(5); T12_est = X(6);
X_bi = X;

if PLOT == 1
    subplot(1,2,2); 
    plot(t,S1,'rs',t,S2,'bs',t,S3,'gs'); hold on;
    plot(t,biexp(t,c1_est*(1-2*exp(-TI1/T11_est)), c2_est*(1-2*exp(-TI1/T12_est)),T21_est,T22_est),'r-',...
        t,biexp(t,c1_est*(1-2*exp(-TI2/T11_est)), c2_est*(1-2*exp(-TI2/T12_est)),T21_est,T22_est),'b-',...
        t,biexp(t,c1_est*(1-2*exp(-TI3/T11_est)), c2_est*(1-2*exp(-TI3/T12_est)),T21_est,T22_est),'g-');
    title('Conventional');
    sprintf('Exact: %1.4f  %1.4f  %1.4f  %1.4f %1.4f  %1.4f',c1,c2,T21,T22,T11,T12)
end

end


function out = objective_function1(X,S1,S2,S3,t,TI1,TI2,TI3)

c1 = X(1); c2 = X(2); T21 = X(3); T22 = X(4); T11 = X(5); T12 = X(6);
out = norm( biexp(t,0*(1-2*exp(-TI1/T11)), c2*(1-2*exp(-TI1/T12)),T21,T22) - S1, 2)^2 + ...
      norm( biexp(t,c1*(1-2*exp(-TI2/T11)), 0*(1-2*exp(-TI2/T12)),T21,T22) - S2, 2)^2 + ...
      norm( biexp(t,c1*(1-2*exp(-TI3/T11)),c2*(1-2*exp(-TI3/T12)),T21,T22) - S3, 2)^2;

end


function out = objective_function2(X,S1,S2,S3,t,TI1,TI2,TI3)

c1 = X(1); c2 = X(2); T21 = X(3); T22 = X(4); T11 = X(5); T12 = X(6);
out = norm( biexp(t,c1*(1-2*exp(-TI1/T11)), c2*(1-2*exp(-TI1/T12)),T21,T22) - S1, 2)^2 + ...
      norm( biexp(t,c1*(1-2*exp(-TI2/T11)), c2*(1-2*exp(-TI2/T12)),T21,T22) - S2, 2)^2 + ...
      norm( biexp(t,c1*(1-2*exp(-TI3/T11)), c2*(1-2*exp(-TI3/T12)),T21,T22) - S3, 2)^2;

end

function out = monoexp(t,c,T2)
    out = c*exp(-t/T2);
end

function out = biexp(t,d1,d2,T21,T22)
    out = d1*exp(-t/T21) + d2*exp(-t/T22);
end

function out = Master_biexp(t,c1,c2,T11,T12,T21,T22,TI)

out = c1 * ( 1- 2*exp(-TI/T11) ) * exp(-t/T21)   + c2 * ( 1- 2*exp(-TI/T12) ) * exp(-t/T22);

end

