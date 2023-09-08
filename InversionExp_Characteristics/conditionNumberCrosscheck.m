%%%%%%%%%%%%%
%This script repeats the TI varied condition number with matlab tools to
%see if there is any difference presented in a Matlab Figure

clc
clear
close all

%%%%%%%%%%%%%
%% Options

intEnds = 1;

%% Initialize Parameters

c1 = 0.5;
c2 = 0.5;
T21 = 45;
T22 = 200;
T11 = 600;
T12 = 1200;

TE_series = 8:8:512;

nullRadius = 30;
nullResolution = 0.1;
TI1star = log(2)*T11;

standard_array = -nullRadius:nullResolution:nullRadius;

if intEnds
    TI_array = floor(TI1star) + standard_array;
else
    TI_array = TI1star + standard_array;
end
paramList = {'T_{11}','T_{12}','c_1','c_2','T_{21}','T_{22}'};

paramString = strcat(paramList(1), ":", string(T11), "::",...
                    paramList(2), ":", string(T12), "::",...
                    paramList(3), ":", string(c1), "::" ,...
                    paramList(4), ":", string(c2), "::" ,...
                    paramList(5), ":", string(T21), "::", ...
                    paramList(6), ":", string(T22), "::");


%% Generate Data

reduced_6p = 1;
if reduced_6p
    columns_kept = [1,2,3,4,5,6];
    final_size = length(columns_kept);
    CN_addendum = "\nclip T2s";
else
    CN_addendum = "";
end

condition_6p_B = zeros(length(TI_array),1);
if reduced_6p
    condition_6pList_BTB = zeros(length(TI_array),final_size);
else
    condition_6pList_BTB = zeros(length(TI_array),6);
end
condition_6p_BTB = zeros(length(TI_array),1);

eigenBig_6p = zeros(length(TI_array),2);
eigenLil_6p = zeros(length(TI_array),2);

condition_4p_B = zeros(length(TI_array),1);
condition_4pList_BTB = zeros(length(TI_array),4);
condition_4p_BTB = zeros(length(TI_array),1);

eigenBig_4p = zeros(length(TI_array),2);
eigenLil_4p = zeros(length(TI_array),2);

for iTI = 1:length(TI_array)

    TI = TI_array(iTI);

    B_mat = gen_Jac_6p(TE_series, TI, T11, T12, c1, c2, T21, T22);
    if reduced_6p
        B_mat = B_mat(:,columns_kept);
        if size(B_mat,2) ~= final_size
            error("Mismatch")
        end
    end
    B_svd = svd(B_mat);
    eigenBig_6p(iTI,1) = B_svd(1);
    eigenLil_6p(iTI,1) = B_svd(end);
    condition_6p_B(iTI) = cond(B_mat);
    covP = B_mat'*B_mat;
    B_svd = svd(covP);
    eigenBig_6p(iTI,2) = B_svd(1);
    eigenLil_6p(iTI,2) = B_svd(end);
    condition_6p_BTB(iTI) = cond(covP);
    CN_params = reshape(diag(covP),1,final_size);
    condition_6pList_BTB(iTI,:) = CN_params;

    d1 = get_dval(TI,c1,T11);
    d2 = get_dval(TI,c2,T12);
    B_mat = gen_Jac_4p(TE_series, d1, d2, T21, T22);
    B_svd = svd(B_mat);
    eigenBig_4p(iTI,1) = B_svd(1);
    eigenLil_4p(iTI,1) = B_svd(end);
    condition_4p_B(iTI) = cond(B_mat);
    covP = B_mat'*B_mat;
    B_svd = svd(covP);
    eigenBig_4p(iTI,2) = B_svd(1);
    eigenLil_4p(iTI,2) = B_svd(end);
    condition_4p_BTB(iTI) = cond(covP);
    CN_params = reshape(diag(covP),1,4);
    condition_4pList_BTB(iTI,:) = CN_params;
end

%% Plot Data

figure;
sgtitle(paramString)
set(gcf,'Position',[100 100 1100 400])
subplot(1,2,1)
plot(TI_array,condition_6p_B,'b-x')
xline(TI1star,'k','LineWidth',1)
xlabel('TI')
ylabel('Condition Number')
title('B matrix - 6 params')

subplot(1,2,2)
plot(TI_array,condition_6p_BTB,'b-x')
xline(TI1star,'k','LineWidth',1)
xlabel('TI')
ylabel('Condition Number')
title('BTB matrix - 6 params')

figure;
sgtitle(paramString)
set(gcf,'Position',[100 100 1100 400])
subplot(1,2,1)
plot(TI_array,condition_4p_B,'b-x')
xline(TI1star,'k','LineWidth',1)
xlabel('TI')
ylabel('Condition Number')
title('B matrix - 4 params')

subplot(1,2,2)
plot(TI_array,condition_4p_BTB,'b-x')
xline(TI1star,'k','LineWidth',1)
xlabel('TI')
ylabel('Condition Number')
title('BTB matrix - 4 params')


figure;
sgtitle(paramString)
set(gcf,'Position',[100 100 1100 800])
subplot(2,2,1)
plot(TI_array,eigenBig_6p(:,1),'b-x')
xline(TI1star,'k','LineWidth',1)
xlabel('TI')
ylabel('Singular Value')
legend('B matrix','BTB matrix')
title('B - Largest Singular Value - 6 params')

subplot(2,2,2)
plot(TI_array,eigenBig_6p(:,2),'b-x')
xline(TI1star,'k','LineWidth',1)
xlabel('TI')
ylabel('Singular Value')
title('BTB - Largest Singular Value - 6 params')

subplot(2,2,3)
plot(TI_array,eigenLil_6p(:,1),'b-x')
xline(TI1star,'k','LineWidth',1)
xlabel('TI')
ylabel('Singular Value')
legend('B matrix','BTB matrix')
title('B - Smallest Singular Value - 6 params')

subplot(2,2,4)
plot(TI_array,eigenLil_6p(:,2),'b-x')
xline(TI1star,'k','LineWidth',1)
xlabel('TI')
ylabel('Singular Value')
title('BTB - Smallest Singular Value - 6 params')

figure;
sgtitle(paramString)
set(gcf,'Position',[100 100 1100 800])
subplot(2,2,1)
plot(TI_array,eigenBig_4p(:,1),'b-x')
xline(TI1star,'k','LineWidth',1)
xlabel('TI')
ylabel('Singular Value')
legend('B matrix','BTB matrix')
title('B - Largest Singular Value - 4 params')

subplot(2,2,2)
plot(TI_array,eigenBig_4p(:,2),'b-x')
xline(TI1star,'k','LineWidth',1)
xlabel('TI')
ylabel('Singular Value')
title('BTB - Largest Singular Value - 4 params')

subplot(2,2,3)
plot(TI_array,eigenLil_4p(:,1),'b-x')
xline(TI1star,'k','LineWidth',1)
xlabel('TI')
ylabel('Singular Value')
legend('B matrix','BTB matrix')
title('B - Smallest Singular Value - 4 params')

subplot(2,2,4)
plot(TI_array,eigenLil_4p(:,2),'b-x')
xline(TI1star,'k','LineWidth',1)
xlabel('TI')
ylabel('Singular Value')
title('BTB - Smallest Singular Value - 4 params')

%% Vary T1 Experiment Initialization

TI_hold = 800;

T12_array = 800:2:1200;

T11 = 1000;

c1 = 0.3;
c2 = 0.7;
T21 = 60;
T22 = 45;

paramList = {'T_{11}','T_{12}','c_1','c_2','T_{21}','T_{22}'};
paramString = strcat(paramList(1), ":", string(T11), "::",...
                    paramList(3), ":", string(c1), "::" ,...
                    paramList(4), ":", string(c2), "::" ,...
                    paramList(5), ":", string(T21), "::", ...
                    paramList(6), ":", string(T22), "::");

%% Generate Data Vay T1 Experiment

condition_6p_B = zeros(length(T12_array),1);
if reduced_6p
    condition_6pList_BTB = zeros(length(T12_array),final_size);
else
    condition_6pList_BTB = zeros(length(T12_array),6);
end
condition_6p_BTB = zeros(length(T12_array),1);

eigenBig_6p = zeros(length(T12_array),2);
eigenLil_6p = zeros(length(T12_array),2);

condition_4p_B = zeros(length(T12_array),1);
condition_4pList_BTB = zeros(length(T12_array),4);
condition_4p_BTB = zeros(length(T12_array),1);

eigenBig_4p = zeros(length(T12_array),2);
eigenLil_4p = zeros(length(T12_array),2);

for iT12 = 1:length(T12_array)

    T12 = T12_array(iT12);

    B_mat = gen_Jac_6p(TE_series, TI_hold, T11, T12, c1, c2, T21, T22);
    if reduced_6p
        B_mat = B_mat(:,columns_kept);
        if size(B_mat,2) ~= final_size
            error("Mismatch")
        end
    end
    B_svd = svd(B_mat);
    eigenBig_6p(iT12,1) = B_svd(1);
    eigenLil_6p(iT12,1) = B_svd(end);
    condition_6p_B(iT12) = cond(B_mat);
    covP = B_mat'*B_mat;
    B_svd = svd(covP);
    eigenBig_6p(iT12,2) = B_svd(1);
    eigenLil_6p(iT12,2) = B_svd(end);
    condition_6p_BTB(iT12) = cond(covP);
    CN_params = reshape(diag(covP),1,final_size);
    condition_6pList_BTB(iT12,:) = CN_params;

    d1 = get_dval(TI_hold,c1,T11);
    d2 = get_dval(TI_hold,c2,T12);
    B_mat = gen_Jac_4p(TE_series, d1, d2, T21, T22);
    B_svd = svd(B_mat);
    eigenBig_4p(iT12,1) = B_svd(1);
    eigenLil_4p(iT12,1) = B_svd(end);
    condition_4p_B(iT12) = cond(B_mat);
    covP = B_mat'*B_mat;
    B_svd = svd(covP);
    eigenBig_4p(iT12,2) = B_svd(1);
    eigenLil_4p(iT12,2) = B_svd(end);
    condition_4p_BTB(iT12) = cond(covP);
    CN_params = reshape(diag(covP),1,4);
    condition_4pList_BTB(iT12,:) = CN_params;
end

%% Plot T1 Related Data

figure;
sgtitle(paramString)
set(gcf,'Position',[100 100 1100 400])
subplot(1,2,1)
plot(T12_array,condition_6p_B,'b-x')
xline(T11,'k','LineWidth',1)
xlabel('T12')
ylabel('Condition Number')
title('B matrix - 6 params')

subplot(1,2,2)
plot(T12_array,condition_6p_BTB,'b-x')
xline(T11,'k','LineWidth',1)
xlabel('T12')
ylabel('Condition Number')
title('BTB matrix - 6 params')

figure;
sgtitle(paramString)
set(gcf,'Position',[100 100 1100 400])
subplot(1,2,1)
plot(T12_array,condition_4p_B,'b-x')
xline(T11,'k','LineWidth',1)
xlabel('T12')
ylabel('Condition Number')
title('B matrix - 4 params')

subplot(1,2,2)
plot(T12_array,condition_4p_BTB,'b-x')
xline(T11,'k','LineWidth',1)
xlabel('T12')
ylabel('Condition Number')
title('BTB matrix - 4 params')


figure;
sgtitle(paramString)
set(gcf,'Position',[100 100 1100 800])
subplot(2,2,1)
plot(T12_array,eigenBig_6p(:,1),'b-x')
xline(T11,'k','LineWidth',1)
xlabel('T12')
ylabel('Singular Value')
legend('B matrix','BTB matrix')
title('B - Largest Singular Value - 6 params')

subplot(2,2,2)
plot(T12_array,eigenBig_6p(:,2),'b-x')
xline(T11,'k','LineWidth',1)
xlabel('T12')
ylabel('Singular Value')
title('BTB - Largest Singular Value - 6 params')

subplot(2,2,3)
plot(T12_array,eigenLil_6p(:,1),'b-x')
xline(T11,'k','LineWidth',1)
xlabel('T12')
ylabel('Singular Value')
legend('B matrix','BTB matrix')
title('B - Smallest Singular Value - 6 params')

subplot(2,2,4)
plot(T12_array,eigenLil_6p(:,2),'b-x')
xline(T11,'k','LineWidth',1)
xlabel('T12')
ylabel('Singular Value')
title('BTB - Smallest Singular Value - 6 params')

figure;
sgtitle(paramString)
set(gcf,'Position',[100 100 1100 800])
subplot(2,2,1)
plot(T12_array,eigenBig_4p(:,1),'b-x')
xline(T11,'k','LineWidth',1)
xlabel('T12')
ylabel('Singular Value')
legend('B matrix','BTB matrix')
title('B - Largest Singular Value - 4 params')

subplot(2,2,2)
plot(T12_array,eigenBig_4p(:,2),'b-x')
xline(T11,'k','LineWidth',1)
xlabel('T12')
ylabel('Singular Value')
title('BTB - Largest Singular Value - 4 params')

subplot(2,2,3)
plot(T12_array,eigenLil_4p(:,1),'b-x')
xline(T11,'k','LineWidth',1)
xlabel('T12')
ylabel('Singular Value')
legend('B matrix','BTB matrix')
title('B - Smallest Singular Value - 4 params')

subplot(2,2,4)
plot(T12_array,eigenLil_4p(:,2),'b-x')
xline(T11,'k','LineWidth',1)
xlabel('T12')
ylabel('Singular Value')
title('BTB - Smallest Singular Value - 4 params')

%% Vary T1 and TI Experiment Initialization

TI_array = 600:4:800;

T12_array = 800:4:1200;

T11 = 1000;

T1star = log(2)*T11;

T2star_low = log(2)*800;
T2star_high = log(2)*1200;

c1 = 0.3;
c2 = 0.7;
T21 = 60;
T22 = 45;

paramList = {'T_{11}','T_{12}','c_1','c_2','T_{21}','T_{22}'};
paramString = strcat(paramList(1), ":", string(T11), "::",...
                    paramList(3), ":", string(c1), "::" ,...
                    paramList(4), ":", string(c2), "::" ,...
                    paramList(5), ":", string(T21), "::", ...
                    paramList(6), ":", string(T22), "::");

%% Generate Data Vay T1 and TI Experiment

condition_6p_B = zeros(length(TI_array)*length(T12_array),1);
condition_6pList_BTB = zeros(length(TI_array)*length(T12_array),6);
condition_6p_BTB = zeros(length(TI_array)*length(T12_array),1);

condition_4p_B = zeros(length(TI_array)*length(T12_array),1);
condition_4pList_BTB = zeros(length(TI_array)*length(T12_array),4);
condition_4p_BTB = zeros(length(TI_array)*length(T12_array),1);

[TI_mesh, T12_mesh] = meshgrid(TI_array,T12_array);
TI_mesh = TI_mesh(:);
T12_mesh = T12_mesh(:);

for iT = 1:length(TI_mesh)

    TI = TI_mesh(iT);
    T12 = T12_mesh(iT);

    
    B_mat = gen_Jac_6p(TE_series, TI, T11, T12, c1, c2, T21, T22);
    condition_6p_B(iT) = cond(B_mat);
    covP = B_mat'*B_mat;
    condition_6p_BTB(iT) = cond(covP);
    CN_params = reshape(diag(covP),1,final_size);
    condition_6pList_BTB(iT,:) = CN_params;
    d1 = get_dval(TI,c1,T11);
    d2 = get_dval(TI,c2,T12);

    B_mat = gen_Jac_4p(TE_series, d1, d2, T21, T22);
    condition_4p_B(iT) = cond(B_mat);
    covP = B_mat'*B_mat;
    condition_4p_BTB(iT) = cond(covP);
    CN_params = reshape(diag(covP),1,4);
    condition_4pList_BTB(iT,:) = CN_params;
end

%% Plot T1 and TI Related Data

% figure;
% sgtitle(paramString)
% set(gcf,'Position',[100 100 1100 400])
% subplot(1,2,1)
% scatter3(TI_mesh(:),T12_mesh(:),condition_6p_B(:))
% xlabel('TI')
% ylabel('T12')
% zlabel('Condition Number')
% title('B matrix - 6 params')
% 
% subplot(1,2,2)
% scatter3(TI_mesh(:),T12_mesh(:),condition_6p_BTB(:))
% xlabel('TI')
% ylabel('T12')
% zlabel('Condition Number')
% title('BTB matrix - 6 params')

figure;
sgtitle(paramString)
set(gcf,'Position',[100 100 1100 400])
subplot(1,2,1)
scatter3(TI_mesh(:),T12_mesh(:),condition_4p_B(:))
zlim([0,0.05*max(condition_4p_B)])
line([T1star;T1star], [mean(T12_array),mean(T12_array)], zlim, 'LineWidth', 2, 'Color', 'k');
line([T2star_low;T2star_high], [T12_array(1);T12_array(end)], [mean(zlim),mean(zlim)], 'LineWidth', 2, 'Color', 'r'); 
xlabel('TI')
ylabel('T12')
zlabel('Condition Number') 

title('B matrix - 4 params')

subplot(1,2,2)
scatter3(TI_mesh(:),T12_mesh(:),condition_4p_BTB(:))
zlim([0,0.001*max(condition_4p_BTB)])
line([T1star;T1star], [mean(T12_array),mean(T12_array)], zlim, 'LineWidth', 2, 'Color', 'k');  
line([T2star_low;T2star_high], [T12_array(1);T12_array(end)], [mean(zlim),mean(zlim)], 'LineWidth', 2, 'Color', 'r');
xlabel('TI')
ylabel('T12')
zlabel('Condition Number')
title('BTB matrix - 4 params')


%% Functions

function d_val = get_dval(TI,c,T1)
    d_val = c*(1-2*exp(-TI/T1));
end

function Jacobian = gen_Jac_6p(TE,TI,T11,T12,c1,c2,T21,T22)
    dT11 = (-2*c1*TI/T11^2).*exp(-(TI/T11 + TE./T21));
    dT12 = (-2*c2*TI/T12^2).*exp(-(TI/T12 + TE./T22));
    dc1 = (1-2*exp(-TI/T11)).*exp(-TE./T21);
    dc2 = (1-2*exp(-TI/T12)).*exp(-TE./T22);
    dT21 = (c1*TE/T21^2).*(1-2*exp(-TI/T11)).*exp(-TE./T21);
    dT22 = (c2*TE/T22^2).*(1-2*exp(-TI/T12)).*exp(-TE./T22);

    Jacobian = [dT11(:), dT12(:), dc1(:), dc2(:), dT21(:), dT22(:)];
end

function Jacobian = gen_Jac_4p(TE,d1,d2,T21,T22)
    dd1 = exp(-TE./T21);
    dd2 = exp(-TE./T22);
    dT21 = (d1*TE/T21^2).*exp(-TE./T21);
    dT22 = (d2*TE/T22^2).*exp(-TE./T22);
    
    Jacobian = [dd1(:), dd2(:), dT21(:), dT22(:)];
end