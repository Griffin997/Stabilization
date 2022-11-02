% the final version
% 08

clc;clear;close all;
ROI_len = 100;
nTIs = 24;
nTEs1 = 2048;
nTEs_start = 1;
nTEs_end = 2048;
nTEs = nTEs_end - nTEs_start + 1;
TI = 1000*[0.015,0.03,0.04,0.05,0.06,0.08,0.1,0.130000000000000,0.150000000000000,0.170000000000000,0.200000000000000,0.250000000000000,0.300000000000000,0.350000000000000,0.400000000000000,0.500000000000000,0.600000000000000,0.700000000000000,0.800000000000000,0.900000000000000,1,1.20000000000000,1.50000000000000,2];
TI = TI(7:end);
TE = (0.4*nTEs_start:0.4:0.4*nTEs_end)';
%%
% measured reference values:
% get reflection point
opt = optimoptions(@lsqnonlin,'Display','off');

% load 1D data
f1D=fopen('1D_28avg/2dseq','r');
data_1Dvec=fread(f1D,'int16');
fclose(f1D);
data1D_ROI = reshape(data_1Dvec,nTEs1,ROI_len);
% load 2D data
f2D=fopen('2D_1avg/2dseq','r');
data_2Dvec=fread(f2D,'int16');
fclose(f2D);
data2D_ROI = reshape(data_2Dvec,nTEs1,nTIs,ROI_len);
%% SNR info
% std_2d = std(data2D_ROI,[],3);
% SNR_2D = max(max(max(data2D_ROI)))./std_2d;
% aa1D = std(data1D_ROI,[],2);
% SNR_1D = max(max(data1D_ROI))./aa1D;
%%
[X,Y] = meshgrid(TI,TE);
XY(:,:,1) = X;
XY(:,:,2) = Y;

data1D_ROI = data1D_ROI(nTEs_start:nTEs_end,:);
data2D_ROI = data2D_ROI(nTEs_start:nTEs_end,7:end,:);
nTIs = length(TI);
ROI_len = 100;

%%
baseline_1D = mean(data1D_ROI(end,:));
% baseline_1D = 0;

%% do 1D inversion
data1D_ROI = data1D_ROI - baseline_1D;
BN_Exp_1D = zeros(size(data1D_ROI,2),4);
oneD_ROI = zeros(nTEs,ROI_len);
res_1D = zeros(nTEs,ROI_len);
res_1D_true = zeros(nTEs,ROI_len);

%%
parfor i = 1:size(data1D_ROI,2)
    ZN = data1D_ROI(:,i);
    c0_ini = (0.4+0.1*rand())*max(ZN);
    x0_ini = [c0_ini max(ZN)-c0_ini 30+10*rand() 40+10*rand()];

    surfit = @(B) B(1)*exp(-TE/B(3))+ B(2)*exp(-TE/B(4)) - ZN;

    [BN,resnorm,residual] = lsqnonlin(surfit, x0_ini,[],[],opt);

    res_1D(:,i) = residual;
    oneD_ROI(:,i) = ZN;
    BN_Exp_1D(i,:) = [BN(1)/(BN(1) + BN(2)) BN(2)/(BN(1) + BN(2)) BN(3) BN(4)];
end
for i = 1:ROI_len
    if BN_Exp_1D(i,2) < BN_Exp_1D(i,1)
        a = BN_Exp_1D(i,1);
        BN_Exp_1D(i,1) = BN_Exp_1D(i,2);
        BN_Exp_1D(i,2) = a;
        b = BN_Exp_1D(i,3);
        BN_Exp_1D(i,3) = BN_Exp_1D(i,4);
        BN_Exp_1D(i,4) = b;
    end
end

x0 = mean(BN_Exp_1D);


%%
baseline_2D = zeros(nTEs,1);
prev_idx = 0;
prev_case = 0;
for i = 1:nTEs
    
    if i <= 400
        temp_TE_line_A = reshape(data2D_ROI(i,:,:),[],ROI_len);
        temp_TE_line = mean(temp_TE_line_A,2);
        [center,center_idx] = min(temp_TE_line);
        left = temp_TE_line(center_idx-1);
        right = temp_TE_line(center_idx+1);
        left_dist = left - center;
        right_dist = right - center;
    
        if left_dist <= right_dist % reflect left node case 0
            left_left = temp_TE_line(center_idx-2);
            left_left_dist = left_left - left;
            theoretical_dist = (left_left_dist + right_dist)/2;
            offset = (left + center - theoretical_dist)/2;
%             offset = 0;
            temp_TE_line_A = temp_TE_line_A - offset;
            temp_TE_line_A(1:center_idx-1,:) = -temp_TE_line_A(1:center_idx-1,:);
            data2D_ROI(i,:,:) = temp_TE_line_A;
            prev_case = 0;
        else % reflect center node case 1
            right_right = temp_TE_line(center_idx + 2);
            right_right_dist = right_right - right;
            theoretical_dist = (left_dist + right_right_dist)/2;
            offset = (right + center - theoretical_dist)/2;
%             offset = 0;
            temp_TE_line_A = temp_TE_line_A - offset;
            temp_TE_line_A(1:center_idx,:) = -temp_TE_line_A(1:center_idx,:);
            data2D_ROI(i,:,:) = temp_TE_line_A;
            prev_case = 1;
        end
    else
        temp_TE_line_A = reshape(data2D_ROI(i,:,:),[],ROI_len);
        temp_TE_line = mean(temp_TE_line_A,2);
        center_idx = 10;
        center = temp_TE_line(center_idx);
        left = temp_TE_line(center_idx-1);
        right = temp_TE_line(center_idx+1);
        left_dist = left - center;
        right_dist = right - center;
        left_left = temp_TE_line(center_idx-2);
        left_left_dist = left_left - left;
        theoretical_dist = (left_left_dist + right_dist)/2;
        offset = (left + center - theoretical_dist)/2;
%         offset = 0;
        temp_TE_line_A = temp_TE_line_A - offset;
        temp_TE_line_A(1:center_idx-1,:) = -temp_TE_line_A(1:center_idx-1,:);
        data2D_ROI(i,:,:) = temp_TE_line_A;
        prev_case = 0;
    end
        
    prev_idx = center_idx;
    baseline_2D(i) = offset;
        

        
    
end

% [X,Y] = meshgrid(TI,TE);
% XY(:,:,1) = X;
% XY(:,:,2) = Y;
%% 2D inversion
% surfit = @(B,XY) B(5)*exp(-TE/B(1)) .* (1-2.*exp((-TI/B(2))))'+ B(6)*exp(-TE/B(3)) .* (1-2.*exp((-TI/B(4))))';
BN_Exp_2D = zeros(ROI_len,6);
% x0 = [15.7 1650 100 555 0.505 0.495];
res_2D = zeros(ROI_len,nTEs,nTIs);
fitted_2D = zeros(ROI_len,nTEs,nTIs);
parfor i = 1:ROI_len
    ZN = reshape(data2D_ROI(:,:,i),nTEs,nTIs);
    maxZN = max(max(ZN));
    c0_ini = (0.4+0.1*rand())*maxZN;
%     x0_ini = [c0_ini max(ZN)-c0_ini 100+10*rand() 500+100*rand()];

    x0_ini = [30+10*rand() 200+50*rand() 40+10*rand() 400+100*rand() c0_ini maxZN-c0_ini];
    surfit = @(B) (B(5)*exp(-TE/B(1)) .* (1-2.*exp((-TI/B(2))))+ B(6)*exp(-TE/B(3)) .* (1-2.*exp((-TI/B(4)))) - ZN);
    surfit_1 = @(B) B(5)*exp(-TE/B(1)) .* (1-2.*exp((-TI/B(2))))+ B(6)*exp(-TE/B(3)) .* (1-2.*exp((-TI/B(4))));
    BN = lsqnonlin(surfit,x0_ini,[],[],opt);
%     BN = lsqnonlin(surfit,x0_ini,[],[],opt);

    % BN_Exp_2D = [c1 c2 T21 T22 T11 T12]
    BN_Exp_2D(i,:) = [BN(5)/(BN(5) + BN(6)) BN(6)/(BN(5) + BN(6)) BN(1) BN(3) BN(2) BN(4)];
    res_2D(i,:,:) = surfit(BN);
    fitted_2D(i,:,:) = ZN - surfit(BN);

end

for i = 1:ROI_len
    if BN_Exp_2D(i,5) > BN_Exp_2D(i,6)
        a = BN_Exp_2D(i,1);
        BN_Exp_2D(i,1) = BN_Exp_2D(i,2);
        BN_Exp_2D(i,2) = a;
        b = BN_Exp_2D(i,3);
        BN_Exp_2D(i,3) = BN_Exp_2D(i,4);
        BN_Exp_2D(i,4) = b;
        c = BN_Exp_2D(i,5);
        BN_Exp_2D(i,5) = BN_Exp_2D(i,6);
        BN_Exp_2D(i,6) = c;
    end
end

% idx_2D = BN_Exp_2D(:,3) <= 100;
% BN_Exp_2D(idx_2D,:) = [];
% 
% idx_1D = BN_Exp_1D(:,2) <= 0.1;
% BN_Exp_1D(idx_1D,:) = [];

%% Get histogram
edges_T21 = linspace(32,48,400);
edges_T22 = linspace(34,50,400);
edges_c1 = linspace(0.32,0.52,160);
edges_c2 = linspace(0.48,0.68,160);
edges_T11 = linspace(130,160,200);
edges_T12 = linspace(340,440,200);
h1 = figure;
tiledlayout(3,2)
ax1 = nexttile;
h = histogram(BN_Exp_1D(:,1),edges_c1,'EdgeColor','blue','FaceColor','blue');
% % h.Normalization = 'countdensity';
hold on
h = histogram(BN_Exp_2D(:,1),edges_c1,'EdgeColor','red','FaceColor','red');
% % h.Normalization = 'countdensity';
legend({'1D: Distribution of c_1','2D: Distribution of c_1'},'FontSize',30,'location','northeast')
xlabel('c_1','FontSize',48,'FontWeight','bold')

ax2= nexttile;
h = histogram(BN_Exp_1D(:,2),edges_c2,'EdgeColor','blue','FaceColor','blue');
% h.Normalization = 'countdensity';

hold on
h = histogram(BN_Exp_2D(:,2),edges_c2,'EdgeColor','red','FaceColor','red');
% h.Normalization = 'countdensity';

legend({'1D: Distribution of c_2','2D: Distribution of c_2'},'FontSize',30,'location','northwest')
xlabel('c_2','FontSize',48,'FontWeight','bold')


ax3 = nexttile;
h = histogram(BN_Exp_1D(:,3),edges_T21,'EdgeColor','blue','FaceColor','blue');
% h.Normalization = 'countdensity';
hold on
h = histogram(BN_Exp_2D(:,3),edges_T21,'EdgeColor','red','FaceColor','red');
% h.Normalization = 'countdensity';
legend({'1D: Distribution of T_{2,1}','2D: Distribution of T_{2,1}'},'FontSize',30,'location','northeast')
xlabel('T_{2,1}','FontSize',48,'FontWeight','bold')


ax4 = nexttile;
h = histogram(BN_Exp_1D(:,4),edges_T22,'EdgeColor','blue','FaceColor','blue');
% h.Normalization = 'countdensity';
hold on
h = histogram(BN_Exp_2D(:,4),edges_T22,'EdgeColor','red','FaceColor','red');
% h.Normalization = 'countdensity';
legend({'1D: Distribution of T_{2,2}','2D: Distribution of T_{2,2}'},'FontSize',30,'location','northwest')
xlabel('T_{2,2}','FontSize',48,'FontWeight','bold')

ax5 = nexttile;
h = histogram(BN_Exp_2D(:,5),edges_T11,'EdgeColor','red','FaceColor','red');
% h.Normalization = 'countdensity';
legend('2D: Distribution of T_{1,1}','FontSize',30,'location','northeast')
xlabel('T_{1,1}','FontSize',48,'FontWeight','bold')


ax6 = nexttile;
h = histogram(BN_Exp_2D(:,6),edges_T12,'EdgeColor','red','FaceColor','red');
% h.Normalization = 'countdensity';
legend('2D: Distribution of T_{1,2}','FontSize',30,'location','northwest')
xlabel('T_{1,2}','FontSize',48,'FontWeight','bold')
sgtitle( {'1D Versus 2D Results from Experimental Data'},'FontSize',44,'FontWeight','bold'); 


% linkaxes([ax1,ax2],'xy')
% linkaxes([ax5,ax6],'xy')

% linkaxes([ax3,ax4],'xy')
set(gcf,'position',[1616        -176        1598         976])
%% save fig
saveas(h1,'exp_dualgel','epsc')



%%
format long
% mean(BN_Exp_1D)
std(BN_Exp_1D)
% mean(BN_Exp_2D)
std(BN_Exp_2D)
%%
mean(BN_Exp_1D)
mean(BN_Exp_2D)