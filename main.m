%% 初始化
clear
close all
clc
format shortg
warning off
addpath('func_defined')

%% 数据读取
data = xlsread('新数据.xlsx','Sheet1','A1:N342');  
opts.PreserveVariableNames = true;
% 输入属性列
input = data(:,[1,2,3,4,5,7,8,9,10,11]);
% 标记输出属性
output = data(:,6); 

% 全部样本数目
N = length(output);   
testNum = 15;   %设定测试样本数目 15
% 计算训练样本数目
trainNum = N-testNum;    

%% 划分训练集、测试集
input_train = input(1:trainNum,:)';
output_train = output(1:trainNum)';
input_test = input(trainNum + 1:trainNum+testNum,:)';
output_test = output(trainNum + 1:trainNum+testNum)';

%% 数据归一化
[inputn,inputps] = mapminmax(input_train,-1,1);
[outputn,outputps] = mapminmax(output_train);
inputn_test = mapminmax('apply',input_test,inputps);

%% 获取输入层节点、输出层节点个数
inputnum = size(input,2);
outputnum = size(output,2);
disp('/////////////////////////////////')
disp('极限学习机ELM结构...')
disp(['输入层的节点数为：',num2str(inputnum)])
disp(['输出层的节点数为：',num2str(outputnum)])
disp(' ')
disp('隐含层节点的确定过程...')

% 初始化最小误差
MSE = 1e+5;
% 隐含层数 20~50
for hiddennum = 20:50                      
    % 用训练数据训练极限学习机模型
   [IW0,B0,LW0,TF,TYPE] = elmtrain(inputn,outputn,hiddennum);
   
    % 对训练集仿真
    an0 = elmpredict(inputn,IW0,B0,LW0,TF,TYPE);  % 仿真结果
    mse0 = mse(outputn,an0);  % 仿真的均方误差
    disp(['隐含层节点数为',num2str(hiddennum),'时，训练集的均方误差为：',num2str(mse0)])

    % 更新最佳的隐含层节点
    if mse0<MSE
        MSE=mse0;
        hiddennum_best=hiddennum;
    end
end
disp(['最佳的隐含层节点数为：',num2str(hiddennum_best),'，相应的均方误差为：',num2str(MSE)])

save para hiddennum_best inputn outputn output_train inputn_test outputps output_test

%% 训练最佳隐含层节点的极限学习机模型
disp(' ')
disp('ELM极限学习机：')
[IW0,B0,LW0,TF,TYPE] = elmtrain(inputn,outputn,hiddennum_best);

%% 模型测试
an0=elmpredict(inputn_test,IW0,B0,LW0,TF,TYPE); % 用训练好的模型进行仿真
test_simu0=mapminmax('reverse',an0,outputps); % 预测结果反归一化
% 误差指标
[mae0,mse0,rmse0,mape0,error0,errorPercent0]=calc_error(output_test,test_simu0);

%% 遗传算法寻最优权值阈值
disp(' ')
disp('GA优化ELM极限学习机：')
% 初始化ga参数
PopulationSize_Data=30;   % 初始种群规模
MaxGenerations_Data=80;   % 最大进化代数
CrossoverFraction_Data=0.8;  % 交叉概率
MigrationFraction_Data=0.2;   % 变异概率
nvars=inputnum*hiddennum_best+hiddennum_best;    % 自变量个数
% 自变量下限
lb=[-ones(inputnum*hiddennum_best,1)          % 输入层到隐含层的连接权值范围是[-1 1]    下限为-1
    zeros(hiddennum_best,1)];              % 隐含层阈值范围是[0 1]  下限为0
% 自变量上限
ub=ones(nvars,1);


% 调用遗传算法函数
options = optimoptions('ga');
options = optimoptions(options,'PopulationSize', PopulationSize_Data);
options = optimoptions(options,'CrossoverFraction', CrossoverFraction_Data);
options = optimoptions(options,'MigrationFraction', MigrationFraction_Data);
options = optimoptions(options,'MaxGenerations', MaxGenerations_Data);
options = optimoptions(options,'SelectionFcn', @selectionroulette);   %轮盘赌选择
options = optimoptions(options,'CrossoverFcn', @crossovertwopoint);   %两点交叉
options = optimoptions(options,'MutationFcn', {  @mutationgaussian [] [] });   %高斯变异
options = optimoptions(options,'Display', 'off');    %‘off’为不显示迭代过程，‘iter’为显示迭代过程
options = optimoptions(options,'PlotFcn', { @gaplotbestf });    %最佳适应度作图

% 求解
[bestx,fval] = ga(@fitness,nvars,[],[],[],[],lb,ub,[],[],options);
delete('para.mat')

%% 优化后的参数训练ELM极限学习机模型
[IW1,B1,LW1,TF,TYPE] = elmtrain(inputn,outputn,hiddennum_best,bestx);          %IW1   B1  LW1为优化后的ELM求得的训练参数

%% 优化后的ELM模型测试
an1=elmpredict(inputn_test,IW1,B1,LW1,TF,TYPE); 
test_simu1=mapminmax('reverse',an1,outputps);

% 误差指标
[mae1,mse1,rmse1,mape1,error1,errorPercent1]=calc_error(output_test,test_simu1);

%% 作图
figure
plot(output_test,'g-.o','linewidth',1)
hold on
plot(test_simu0,'b-*','linewidth',1)
hold on
plot(test_simu1,'r-v','linewidth',1)
legend('真实值','ELM预测值','GA-ELM预测值')
xlabel('测试样本编号')
ylabel('指标值')
title('优化前后的ELM模型预测值和真实值对比图')

figure
plot(error0,'b-*','markerfacecolor','r')
hold on
plot(error1,'r-v','markerfacecolor','r')
legend('ELM预测误差','GA-ELM预测误差')
xlabel('测试样本编号')
ylabel('预测偏差')
title('优化前后的ELM模型预测值和真实值误差对比图')

disp(' ')
disp('/////////////////////////////////')
disp('打印结果表格')
disp('样本序号     实测值      ELM预测值  GA-ELM值   ELM误差   GA-ELM误差')
for i=1:testNum
    disp([i output_test(i),test_simu0(i),test_simu1(i),error0(i),error1(i)])
end
